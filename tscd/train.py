"""
TSCD Training Pipeline — Algorithm 1 from the paper.

Algorithm 1  TSCD Training
----------------------------------------------------------------------
Require: Dataset D, epochs E, fusion interval T, nudging gamma,
         window size M
 1: Initialize W^P, W^N, W^C
 2: for epoch e = 1 to E do
 3:   for each mini-batch B do
 4:     // Parallel stream training
 5:     Compute dyadic states (u^P, v^P), (u^N, v^N) via Eqs. (8)
 6:     Compute gradient proxies nabla_D L^P, nabla_D L^N
 7:     // TF-GVS: Temporal averaging
 8:     Update sliding window, compute nabla_bar_D L^P, nabla_bar_D L^N
 9:     // MP-GBS: Multi-plane bias
10:     Compute consensus direction d* from multi-norm perturbations
11:     Update W^P, W^N using nabla_bar_D L|_{W+d*}
12:   end for
13:   if e mod T = 0 then
14:     // Cross-stream fusion
15:     Transplant: u^C <- u^P, v^C <- v^N
16:     Fine-tune W^C for T_fine steps
17:     Transfer: W^{P,N} <- 0.5 * W^{P,N} + 0.5 * Delta W^C
18:   end if
19: end for
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .models.tscd_network import TSCDFramework
from .optimizers.mp_gbs import MPGBS
from .optimizers.tf_gvs import BatchGroupTFGVS
from .data.negative_sampling import (
    create_positive_samples,
    create_negative_samples,
)


def train_tscd(
    model: TSCDFramework,
    train_loader,
    test_loader,
    num_classes: int,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    # TSCD hyper-parameters
    fusion_interval: int = 100,      # T
    fusion_steps: int = 10,          # T_fine
    gamma: float = 0.1,
    # Optimization strategy flags
    use_mp_gbs: bool = True,
    mp_gbs_rho: float = 0.05,
    use_tf_gvs: bool = True,
    tf_gvs_window: int = 3,         # M
    # Misc
    log_interval: int = 10,
    save_path: str = None,
    goodness_threshold: float = 2.0,
    use_fp16: bool = False,
):
    """
    Train a TSCD model end-to-end following Algorithm 1.

    Key differences from a vanilla FF implementation:
      - Separate per-layer energy losses for positive & negative streams (Eq 6-7)
      - Label embedding into images (Hinton 2022) for pos/neg samples
      - TF-GVS temporal gradient averaging (Eq 17-18) with window M
      - MP-GBS multi-plane bias suppression (Eq 13-16) with closures for both
        streams
      - Cross-stream fusion every T epochs (Eq 9-11)
      - Optional FP16 mixed precision

    Returns:
        history dict with keys: train_loss, test_acc
    """
    model = model.to(device)

    # --- Optimizers (Line 1: Initialize) ---
    opt_pos = optim.AdamW(model.positive_stream.parameters(),
                          lr=lr, weight_decay=weight_decay)
    opt_neg = optim.AdamW(model.negative_stream.parameters(),
                          lr=lr, weight_decay=weight_decay)
    opt_cls = optim.AdamW(model.classifier.parameters(),
                          lr=lr, weight_decay=weight_decay)

    # Cosine annealing schedule
    sched_pos = optim.lr_scheduler.CosineAnnealingLR(opt_pos, T_max=epochs)
    sched_neg = optim.lr_scheduler.CosineAnnealingLR(opt_neg, T_max=epochs)
    sched_cls = optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=epochs)

    # FP16 mixed precision
    use_amp = use_fp16 and device != "cpu" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Wrap with MP-GBS (Line 9-10) — pass scaler for FP16 compatibility
    if use_mp_gbs:
        mp_gbs_pos = MPGBS(model.positive_stream, opt_pos, rho=mp_gbs_rho,
                           scaler=scaler if use_amp else None)
        mp_gbs_neg = MPGBS(model.negative_stream, opt_neg, rho=mp_gbs_rho,
                           scaler=scaler if use_amp else None)

    # TF-GVS handlers (Line 7-8)
    tf_gvs_pos = BatchGroupTFGVS(window_size=tf_gvs_window) if use_tf_gvs else None
    tf_gvs_neg = BatchGroupTFGVS(window_size=tf_gvs_window) if use_tf_gvs else None

    history = {"train_loss": [], "test_acc": []}

    # --- Main loop (Line 2) ---
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Label embedding (Hinton 2022): embed correct/wrong labels
            pos_images = create_positive_samples(images, labels, num_classes)
            neg_images, neg_labels = create_negative_samples(
                images, labels, num_classes
            )

            # ============================================================
            # Line 4-6: Parallel stream training with dyadic states
            # ============================================================
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Positive stream forward (Eq 8: dyadic states)
                pos_feat, pos_states = model.forward_positive(pos_images)
                # Negative stream forward (Eq 8: dyadic states)
                neg_feat, neg_states = model.forward_negative(neg_images)

                # --- Per-layer energy loss (Eq 6-7) ---
                # Positive stream: push per-layer goodness ABOVE threshold
                loss_pos_layer = model.positive_stream.compute_layer_energy_loss(
                    pos_states, threshold=goodness_threshold,
                    stream_type='positive'
                )
                # Negative stream: push per-layer goodness BELOW threshold
                loss_neg_layer = model.negative_stream.compute_layer_energy_loss(
                    neg_states, threshold=goodness_threshold,
                    stream_type='negative'
                )

                # --- Final feature energy (Eq 6) ---
                # Split into separate terms so each stream's loss has its own
                # computation graph (avoids retain_graph issues).
                g_pos = model.positive_stream.compute_goodness(pos_feat)
                g_neg = model.negative_stream.compute_goodness(neg_feat)
                loss_feat_pos = F.relu(goodness_threshold - g_pos).mean()
                loss_feat_neg = F.relu(g_neg - goodness_threshold).mean()

                # --- Classification loss (CE on positive stream) ---
                logits = model.classify(pos_feat)
                cls_loss = F.cross_entropy(logits, labels)

                # Total loss for positive stream (energy + classification)
                loss_pos = loss_pos_layer + loss_feat_pos + cls_loss
                # Total loss for negative stream (energy only)
                loss_neg = loss_neg_layer + loss_feat_neg

            # ============================================================
            # Line 7-8: TF-GVS temporal averaging (Eq 17-18)
            # ============================================================
            if use_tf_gvs:
                # --- Positive stream: backward + accumulate ---
                opt_pos.zero_grad()
                opt_cls.zero_grad()
                scaler.scale(loss_pos).backward(retain_graph=False)
                tf_gvs_pos.accumulate(model.positive_stream)

                # --- Negative stream: backward + accumulate ---
                opt_neg.zero_grad()
                scaler.scale(loss_neg).backward()
                tf_gvs_neg.accumulate(model.negative_stream)

                if tf_gvs_pos.ready:
                    # Apply variance-suppressed gradient (Eq 18)
                    tf_gvs_pos.apply_averaged_grad(model.positive_stream)
                    tf_gvs_neg.apply_averaged_grad(model.negative_stream)

                    # Line 9-11: MP-GBS update
                    if use_mp_gbs:
                        # Positive stream closure for look-ahead (Eq 16)
                        def _closure_pos():
                            opt_pos.zero_grad()
                            opt_cls.zero_grad()
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                pf, ps = model.forward_positive(pos_images)
                                lp = model.positive_stream.compute_layer_energy_loss(
                                    ps, threshold=goodness_threshold,
                                    stream_type='positive'
                                )
                                gp = model.positive_stream.compute_goodness(pf)
                                fl = F.relu(goodness_threshold - gp).mean()
                                lg = model.classify(pf)
                                cl = F.cross_entropy(lg, labels)
                                loss = lp + fl + cl
                            scaler.scale(loss).backward()
                            return loss

                        # Negative stream closure for look-ahead (Eq 16)
                        def _closure_neg():
                            opt_neg.zero_grad()
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                nf, ns = model.forward_negative(neg_images)
                                ln = model.negative_stream.compute_layer_energy_loss(
                                    ns, threshold=goodness_threshold,
                                    stream_type='negative'
                                )
                                gn = model.negative_stream.compute_goodness(nf)
                                fn = F.relu(gn - goodness_threshold).mean()
                                loss = ln + fn
                            scaler.scale(loss).backward()
                            return loss

                        mp_gbs_pos.step(closure=_closure_pos)
                        mp_gbs_neg.step(closure=_closure_neg)
                    else:
                        scaler.step(opt_pos)
                        scaler.step(opt_neg)

                    scaler.step(opt_cls)
                    scaler.update()

                    tf_gvs_pos.reset()
                    tf_gvs_neg.reset()
                # else: accumulate without stepping (TF-GVS window not full)
            else:
                # Standard update (no TF-GVS)
                # --- Positive stream ---
                opt_pos.zero_grad()
                opt_cls.zero_grad()
                scaler.scale(loss_pos).backward(retain_graph=False)

                # --- Negative stream ---
                opt_neg.zero_grad()
                scaler.scale(loss_neg).backward()

                # Line 9-11: MP-GBS update
                if use_mp_gbs:
                    def _closure_pos():
                        opt_pos.zero_grad()
                        opt_cls.zero_grad()
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            pf, ps = model.forward_positive(pos_images)
                            lp = model.positive_stream.compute_layer_energy_loss(
                                ps, threshold=goodness_threshold,
                                stream_type='positive'
                            )
                            gp = model.positive_stream.compute_goodness(pf)
                            fl = F.relu(goodness_threshold - gp).mean()
                            lg = model.classify(pf)
                            cl = F.cross_entropy(lg, labels)
                            loss = lp + fl + cl
                        scaler.scale(loss).backward()
                        return loss

                    def _closure_neg():
                        opt_neg.zero_grad()
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            nf, ns = model.forward_negative(neg_images)
                            ln = model.negative_stream.compute_layer_energy_loss(
                                ns, threshold=goodness_threshold,
                                stream_type='negative'
                            )
                            gn = model.negative_stream.compute_goodness(nf)
                            fn = F.relu(gn - goodness_threshold).mean()
                            loss = ln + fn
                        scaler.scale(loss).backward()
                        return loss

                    mp_gbs_pos.step(closure=_closure_pos)
                    mp_gbs_neg.step(closure=_closure_neg)
                else:
                    scaler.step(opt_pos)
                    scaler.step(opt_neg)

                scaler.step(opt_cls)
                scaler.update()

            total_loss = loss_pos.item() + loss_neg.item()
            epoch_loss += total_loss
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss:.4f}")

        # ============================================================
        # Line 13-18: Cross-stream fusion (every T epochs)
        # ============================================================
        if fusion_interval > 0 and epoch % fusion_interval == 0:
            print(f"\n  [Epoch {epoch}] Cross-fusion activated")
            model.cross_fusion_step(
                train_loader=train_loader,
                num_classes=num_classes,
                device=device,
                fine_tune_steps=fusion_steps,
                lr_cross=lr,
                threshold=goodness_threshold,
            )
            print("  Cross-fusion completed")

        # LR schedule step
        sched_pos.step()
        sched_neg.step()
        sched_cls.step()

        # ============================================================
        # Logging & evaluation
        # ============================================================
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if epoch % log_interval == 0 or epoch == epochs:
            acc = evaluate(model, test_loader, num_classes, device)
            history["test_acc"].append(acc)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}  test_acc={acc:.2f}%")

        # Save checkpoint
        if save_path and (epoch % 100 == 0 or epoch == epochs):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "history": history,
            }, save_path)

    return history


@torch.no_grad()
def evaluate(model: TSCDFramework, dataloader, num_classes: int,
             device: str = "cuda"):
    """
    Compute Top-1 accuracy on a dataset.

    Uses goodness-based FF inference (try all labels, pick highest
    goodness) for moderate class counts (<=100), and classifier-based
    inference for larger class counts (faster but slightly less aligned
    with the FF paradigm).
    """
    model.eval()
    correct = total = 0
    use_goodness = num_classes <= 100
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        if use_goodness:
            preds = model.predict_goodness(images, num_classes)
        else:
            preds = model.predict(images)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0
