#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Knexus Research Corporation and DrivenData, Inc",
    author_email="isaac@drivendata.org",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="Heuristics and model for org matching",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords="deid2",
    name="deid2",
    packages=find_packages(include=["deid2", "deid2.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/drivendataorg/deid2-collaboration",
    version="0.1.0",
    zip_safe=False,
)
