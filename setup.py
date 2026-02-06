"""
Setup file for the Two-Stage Latent ODE package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="two-stage-latent-ode",
    version="0.1.0",
    author="Zhi Ye",
    author_email="fly9811@163.com",
    description="A Biologically Informed Deep Dynamical Model for Personalized Postprandial Metabolic Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhiye9/Two-stage-latent-ODE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-latent-ode=scripts.train_model:main",
            "demo-latent-ode=scripts.demo:main",
        ],
    },
)
