# Taxonomic-Relation-Identification
This is a note for constructing taxonomy.
## 总结研究
### Pattern-Based Methods
Based on hypernym-hyponym pairs from the corpus.
### Clustering-Based Methods
### Supervised Methods

## TaxoGen：Constructing Topical Concept Taxonomy by Adaptive
### Contributions
1. An adaptive spherical clustering module for allocating terms to proper levels when splitting a coarse topic into fine-grained ones.
2. A local embedding module for learning term embeddings that maintain strong discriminative power at different levels of the taxonomy. 
### Methods
**1. Adaptive Spherical Clustering**
* Identify general terms and refine the sub-topics and push general terms back to the parent.
* Using TF-IDF to generate representativeness terms, because  representativeness term should appear frequently in topic S but not in the sibling of topic S.

**2. Local Embedding**
* Skip-Gram
### Datasets

## Derivative Study of Taxonomy


