from setuptools import setup
import glob
import os

setup(
    name="chaotic_neural",
    version="0.0.3",
    author="Kartheik Iyer",
    author_email="kartheik.iyer@dunlap.utoronto.ca",
    url = "https://github.com/kartheikiyer/chaotic_neural",
    packages=["chaotic_neural"],
    description="Associative clustering and ananlysis of papers on the ArXiv",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE","data/model_galaxies_all_trained.arxivmodel"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "feedparser", "tqdm", "sklearn", "summa", "gensim"]
)
