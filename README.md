# chaotic_neutral
Associative clustering and ananlysis of papers on the ArXiv

***

This package aims at providing a trained model to find related papers on ArXiv given another paper or a set of keywords. 

It aims to be different from existing resources like the default ArXiv search, the new [ADS](https://ui.adsabs.harvard.edu/), or [ArXivsorter](https://www.arxivsorter.org) in that it uses [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py), an unsupervised algorithm that trains a shallow neural network to transform every document (in this case ArXiv abstracts) into a vector in a high-dimensional vector. Similar papers are then found by finding the closest vectors to one of interest in this space. This also allows for performing vector operations on keywords and documents to structure specific queries.

Users can either build their own model (by searching ArXiv with specific queries) or use the pre-trained model that has been trained on all the astro-ph.GA papers up to Thursday, Oct 24, 2019.

***

This project was started during the hack session for .Astronomy 11 held at Toronto. See the full collection of present and past hacks at [the hacks collector](https://github.com/dotastro/hacks-collector).


