from setuptools import setup
import glob
import os

setup(
    name="chaotic_neutral",
    version="0.0.1",
    author="Kartheik Iyer",
    author_email="kartheik.iyer@dunlap.utoronto.ca",
    url = "https://github.com/kartheikiyer/chaotic_neutral",
    packages=["chaotic_neutral"],
    description="Associative clustering and ananlysis of papers on the ArXiv",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "feedparser", "urllib", "tqdm", "collections", "sklearn", "summa", "pickle", "gensim"]
)
