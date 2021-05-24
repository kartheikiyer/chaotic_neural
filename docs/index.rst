.. chaotic_neural documentation master file, created by
   sphinx-quickstart on Sat May 22 22:22:18 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

chaotic_neural: Associative clustering and analysis of papers on the ArXiv
===========================================================================

This package aims at providing a model to find related papers on ArXiv given another paper (or a set of keywords).

It aims to be different from existing resources like the default ArXiv search, the new ADS, or ArXivsorter in that it uses Doc2Vec, an unsupervised algorithm that trains a shallow neural network to transform every document (in this case ArXiv abstracts) into a vector in a high-dimensional vector space. Similar papers are then found by finding the closest vectors to one of interest in this space. This also allows for performing vector arithmetic operations on keywords (i.e. adding and subtracting keywords) as well as vectors corresponding to entire documents to structure specific queries.

Users can either build their own model (by searching ArXiv with specific queries) or use the pre-trained model that has been trained on all the astro-ph.GA papers up to Thursday, Oct 24, 2019.


.. toctree::
   :maxdepth: 2
   :caption: General Usage:
   
   usage/installation
   usage/dependencies
   usage/features
   
   
.. toctree::
   :maxdepth: 2
   :caption: Tutorials:
   
   tutorials/basic_usage
   tutorials/visualize_trained_models
   tutorials/arxiv_scraper_location_plots
   tutorials/build_your_own_model
   
   
The code is designed to be intuitive to use, and and consists of three steps to get you started:

- loading a pre-trained model
- performing searches 
- training a new model

More detailed descriptions of these modules can be found in the tutorials. If you are interested in going off the beaten track and trying different things, please let me know so that I can help you run the code as you'd like!


Contribute
----------

- Issue Tracker: https://github.com/kartheikiyer/chaotic_neural/issues
- Source Code: https://github.com/kartheikiyer/chaotic_neural

Support
-------

If you are having issues, please let me know at: kartheik.iyer@dunlap.utoronto.ca

License & Attribution
---------------------

Copyright 2019 Kartheik Iyer and contributors.

`chaotic_neural` is being developed by `Kartheik Iyer <http://kartheikiyer.github.io>`_ in a
`public GitHub repository <https://github.com/kartheikiyer/chaotic_neural>`_.
The source code is made available under the terms of the MIT license.

If you make use of this code, please cite the repository or the upcoming paper (Iyer et al. in prep.).

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`






