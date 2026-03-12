from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dl-cdi",
    version="0.13b0",
    description="Deep Learning Phase Retrieval for Coherent Diffractive Imaging (CDI)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marcus Newton, Ahmed H. Mokhtar",
    license="GPL-3.0",
    py_modules=["model", "train", "predict", "gendata"],
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
