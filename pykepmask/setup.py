###FILE: setup.py
###PURPOSE: Python script for setting up the kepmask package.
###NOTE: This file was generated following the (very helpful!) tutorial at https://packaging.python.org/tutorials/packaging-projects/.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kepmask-jpegues",
    version="2.0.0",
    author="jpegues",
    description="A package for generating Keplerian masks, which can be used to extract emission for disks undergoing Keplerian rotation (e.g., protoplanetary disks).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jpegues/kepmask",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)