# Taxonomic-Relation-Identification
This is a note for constructing taxonomy. 

---

## 总结研究
### Pattern-Based Methods
Based on hypernym-hyponym pairs from the corpus.
### Clustering-Based Methods
### Supervised Methods

---

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
2. HPAM(Hierarchical Pachinko Allocation Model)
> [Mixtures of Hierarchical Topics with Pachinko Allocation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?referer=https://www.google.co.jp/&httpsredir=1&article=1074&context=cs_faculty_pubs)
3. HCLUS(Hierarchical Clustering)
4. NoAC
5. NoLe

### Quantitative Analysis
1. Relation Accuracy
2. Term Coherence
3. Cluster Quality

---

## [Learning Semantic Hierarchies via Word Embeddings](http://ir.hit.edu.cn/~jguo/papers/acl2014-hypernym.pdf)
### Contributions
* This paper proposes a novel method for semantic hierarchy construction based on word embeddings, which are trained using a large-scale corpus.
* Generally speaking, the proposed method greatly improves the recall and F-score but damages the precision.

### Methods
**1. Word Embedding Training**

**2. A Uniform Linear Projection**
Intuitively, we assume that all words can be projected to their hypernyms based on a uniform transition matrix.

**3. Piecewise Linear Projections**
Specifically, the input space is first segmented into several regions. That is, all word pairs (x, y) in the training data are first clustered into several groups, where word pairs in each group are expected to exhibit similar hypernym–hyponym relations.

**4. Piecewise Linear Projections**
 If a circle has only two nodes, we remove the weakest path. If a circle has
more than two nodes, we reverse the weakest path to form an indirect hypernym–hyponym relation.


### Datasets
* Learning word embeddings: Baidubaike, which contains about 30 million sentences (about 780 million words). 
* The Chinese segmentation is provided by the open-source Chinese language processing platform LTP5.
* The training data for projection learning is collected from CilinE.
* For evaluation, we collect the hypernyms for 418 entities, which are selected randomly from Baidubaike.
* The final data set contains 655 unique hypernyms and 1,391 hypernym–hyponym relations among them. 
* Randomly spliting the labeled data into 1/5 for development and 4/5 for testing. 


### Compared Methods
* MWiki+CilinE refers to the manually-built hierarchy extension method of Suchanek et al. (2008).
* MPattern refers to the pattern-based method of Hearst (1992). 
* MSnow originally proposed by Snow et al. (2005)

---

## [Translating Representations of Knowledge Graphs with Neighbors](http://delivery.acm.org/10.1145/3220000/3210085/p917-wang.pdf?ip=121.50.45.233&id=3210085&acc=OPENTOC&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E9F04A3A78F7D3B8D&__acm__=1540699470_287a3f90001d001c2072c8892070aa80)

### Contributions


### Methods
**1. Word Embedding Training**



### Datasets


### Compared Methods


## Derivative Study of Taxonomy

