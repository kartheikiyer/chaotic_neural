# chaotic_neutral
Associative clustering and ananlysis of papers on the ArXiv

***

This package aims at providing a trained model to find related papers on ArXiv given another paper or a set of keywords. 

It aims to be different from existing resources like the default ArXiv search, the new [ADS](https://ui.adsabs.harvard.edu/), or [ArXivsorter](https://www.arxivsorter.org) in that it uses [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py), an unsupervised algorithm that trains a shallow neural network to transform every document (in this case ArXiv abstracts) into a vector in a high-dimensional vector. Similar papers are then found by finding the closest vectors to one of interest in this space. This also allows for performing vector operations on keywords and documents to structure specific queries.

Users can either build their own model (by searching ArXiv with specific queries) or use the pre-trained model that has been trained on all the astro-ph.GA papers up to Thursday, Oct 24, 2019.

***

You can load the pre-trained `galaxies_all` model using:

```python
import chaotic_neutral as cn
model_data = cn.load_trained_doc2vec_model('galaxies_all')
model, all_titles, all_abstracts, all_authors, train_corpus, test_corpus = model_data
```

Having loaded the model, queries can now be run using ArXiv ids for papers as:

```python
sims = cn.list_similar_papers(doc_id = 1903.10457, input_type='arxiv_id')
```

which returns

```
ArXiv id:  1903.10457
Title: Learning the Relationship between Galaxies Spectra and their Star
  Formation Histories using Convolutional Neural Networks and Cosmological
  Simulations
-----------------------------
Most similar/relevant papers: 
-----------------------------
0 Learning the Relationship between Galaxies Spectra and their Star
  Formation Histories using Convolutional Neural Networks and Cosmological
  Simulations  (Corrcoef: 0.98 )
1 A Critical Assessment of Photometric Redshift Methods: A CANDELS
  Investigation  (Corrcoef: 0.71 )
2 SHARDS: an optical spectro-photometric survey of distant galaxies  (Corrcoef: 0.66 )
3 Analysis of galaxy SEDs from far-UV to far-IR with CIGALE: Studying a
  SINGS test sample  (Corrcoef: 0.65 )
4 Probability density estimation of photometric redshifts based on machine
  learning  (Corrcoef: 0.65 )
5 Non-parametric Star Formation History Reconstruction with Gaussian
  Processes I: Counting Major Episodes of Star Formation  (Corrcoef: 0.63 )
6 On the recovery of galaxy properties from SED fitting solutions  (Corrcoef: 0.62 )
7 Photometric Redshifts and Systematic Variations in the SEDs of Luminous
  Red Galaxies from the SDSS DR7  (Corrcoef: 0.61 )
8 Recovering galaxy stellar population properties from broad-band spectral
  energy distribution fitting  (Corrcoef: 0.58 )
9 Estimating Spectra from Photometry  (Corrcoef: 0.57 )
```

Or using keywords:

```python
sims = list_similar_papers(doc_id = ['simulation','sed','fitting'], 
                           input_type='keywords', 
                           return_n=10, show_authors = True, show_summary=True)
```

which returns

```
Keyword(s):  ['simulation', 'sed', 'fitting']
multi-keyword
-----------------------------
Most similar/relevant papers: 
-----------------------------
0 Should we believe the results of UV-mm galaxy SED modelling?  (Corrcoef: 0.53 )
Authors:------
[{'name': 'Christopher C. Hayward'}, {'name': 'Daniel J. B. Smith'}]
Summary:------
We compare the properties inferred from the SED modelling with the true values and find that MAGPHYS recovers most physical parameters of the simulated galaxies well.
 
1 Morphology-assisted galaxy mass-to-light predictions using deep learning  (Corrcoef: 0.47 )
Authors:------
[{'name': 'Wouter Dobbels'}, {'name': 'Serge Krier'}, {'name': 'Stephan Pirson'}, {'name': 'Sébastien Viaene'}, {'name': 'Gert De Geyter'}, {'name': 'Samir Salim'}, {'name': 'Maarten Baes'}]
Summary:------
Spectral energy distribution (SED) fitting can make use of all available fluxes and their errors to make a Bayesian estimate of the M/L.
When we combine the morphology features with global g- and i-band luminosities, we find an improved estimate compared to a model which does not make use of morphology.
While our method was trained to reproduce global SED fitted M/L, galaxy morphology gives us an important additional constraint when using one or two bands.
 
2 Geometric and Dynamical Models of Reverberation Mapping Data  (Corrcoef: 0.46 )
Authors:------
[{'name': 'Anna Pancoast'}, {'name': 'Brendon J. Brewer'}, {'name': 'Tommaso Treu'}]
Summary:------
We present a general method to analyze reverberation mapping data that provides both estimates for the black hole mass and for the geometry and dynamics of the broad line region (BLR) in active galactic nuclei (AGN).
```

***

If you have any ideas for further applications for this, or ways it which we can extend the project, you can comment on the following Google [Doc].(https://docs.google.com/document/d/1wDwFwKyPIz0thDSdMWzKbnvDvQQBXIO6x2-CBcMeKpM/edit?usp=sharing). If you are interested in contributing, please email me at kartheik.iyer@dunlap.utoronto.ca

***

This project was started during the hack session for .Astronomy 11 held at Toronto. See the full collection of present and past hacks at [the hacks collector](https://github.com/dotastro/hacks-collector).



