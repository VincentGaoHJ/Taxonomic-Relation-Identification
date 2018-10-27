# Taxonomic-Relation-Identification
This is a note for constructing taxonomy. 

- [总结研究](#总结研究)  
- [TaxoGen：Constructing Topical Concept Taxonomy by Adaptive](#taxogen-constructing-topical-concept-taxonomy-by-adaptive)  
- [Derivative Study of Taxonomy](#derivative-study-of-taxonomy)  

## 总结研究
### Pattern-Based Methods
Based on hypernym-hyponym pairs from the corpus.
### Clustering-Based Methods
### Supervised Methods

## [TaxoGen：Constructing Topical Concept Taxonomy by Adaptive](https://pdfs.semanticscholar.org/c420/af96a6725414b7c631757503ed6ac61020e6.pdf)
### Contributions
1. An adaptive spherical clustering module for allocating terms to proper levels when splitting a coarse topic into fine-grained ones.
2. A local embedding module for learning term embeddings that maintain strong discriminative power at different levels of the taxonomy. 

### Methods
**1. Adaptive Spherical Clustering**
* Identify general terms and refine the sub-topics and push general terms back to the parent.
* Using TF-IDF to generate representativeness terms, because  representativeness term should appear frequently in topic S but not in the sibling of topic S.

**2. Local Embedding**
* Using Skip-Gram to learn word embedding。
* Retrieve sub-corpus for topic.

### Datasets
1. DBLP(Contains 600 thousand computer science paper titles)
2. SP(Contains 91 thousand paper abstracts)

### Compared Methods
1. HLDA(Hierarchical Latent Dirichlet Allocation Model)
> [Hierarchical Topic Models and the Nested Chinese Restaurant Process](https://papers.nips.cc/paper/2466-hierarchical-topic-models-and-the-nested-chinese-restaurant-process.pdf)
3. HPAM(Hierarchical Pachinko Allocation Model)
> [Mixtures of Hierarchical Topics with Pachinko Allocation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?referer=https://www.google.co.jp/&httpsredir=1&article=1074&context=cs_faculty_pubs)
5. HCLUS(Hierarchical Clustering)
6. NoAC
7. NoLe

### Quantitative Analysis
1. Relation Accuracy
2. Term Coherence
3. Cluster Quality

## Derivative Study of Taxonomy


