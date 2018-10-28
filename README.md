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

## [Translating Representations of Knowledge Graphs with Neighbors](http://sigir.org/sigir2018/toc.html)

### Contributions
**A approach to capture more precise context information and to incorporate neighbor information dynamically.**
1. Firstly, we apply effective neighbor selection to reduce the number of neighbors.
2. Second, we try to encode neighborhood information with context embeddings. 
3. Third, we further utilize attention mechanism to focus on most influential nodes since different neighbors provide different level of information.


### Methods
**1. Neighbor Selection**
For each epoch t, we derive θte, which means the number of neighbors to be considered for an entity e.

**2. Neighbor-based Representation**
each object, entity or relation, is represented by two vectors, one is called object embedding while the other is called context embedding. 


---

## [Enriching Taxonomies With Functional Domain Knowledge](http://sigir.org/sigir2018/toc.html)

### Contributions
**A novel framework, ETF, to enrich large-scale, generic taxonomies with new concepts from resources such as news and research publications.**
1. We develop a novel, fully automated framework, ETF, that generates semantic text-vector embeddings for each new concept. 
2. We propose the use of a learning algorithm that combines a carefully selected set of graph-theoretic and semantic similarity based features to rank candidate parent relations. 
3. We test ETF on large, real-world, publicly available knowledge bases such as Wikipedia and Wordnet, and outperform baselines at the task of inserting new concepts. 



### Methods
**1. Finding Concepts and Taxonomic Relations**
Acquire the entities and categories from the given taxonomy structure. And then obtain the novel concepts to be integrated into T.

**2. Learning Concept Representations**
 * To get the representation of an entity, we add a tf-weighted sum of hte word2vec embeddings of its context terms to the doc2vec representation of its associated document. 
 * After creating embeddingd for the existing concepts in T, we next learn representations for the new concepts tp be inserted into T.

**3. Filtering and Ranking Potential Parents**



---

## Derivative Study of Taxonomy

