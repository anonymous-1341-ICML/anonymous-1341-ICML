from setuptools import setup, find_packages

setup(
    name="tscd",
    version="1.0.0",
    description=(
        "Tri-Stream Coupled Dynamics for Forward Learning — "
        "Official ICML 2026 implementation"
    ),
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    ],
)
