# meta analysis using the pre-trained `galaxies_all` model.

We define the relevance index of a paper as the sum of the correlation reported by Doc2Vec for the top 100 closest related papers. 

Histogram over entire dataset: 
![relevance_index_histogram](meta/images/connectedness_histogram.png)

The most and least connected papers are:

```
Most related paper(s): 
----------------------
ID: http://arxiv.org/abs/1705.03957v1 Title: ALMA Images of the Orion Hot Core at 349 GHz
 
Least related paper(s): 
-----------------------
ID: http://arxiv.org/abs/1608.06756v1  Title: The intrinsic beauty of polytropic spheres in reduced variables
```

Correlations with other quantities:

1. Number of authors:
![relevance_index_numauthors](meta/images/ri_vs_numauthors.png)

2. Number of citations:
![relevance_index_citations](meta/images/ri_vs_numcites.png)

3. Publication year:
![relevance_index_pubyear](meta/images/ri_vs_pubyear.png)

Number of authors as a function of time:
![numauthors_vs_pubyear](meta/images/numauthors_vs_pubyear.png)
