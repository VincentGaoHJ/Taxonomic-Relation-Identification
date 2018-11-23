# Taxonomic-Relation-Identification
This is a note for constructing taxonomy. 

---

## 总结研究

### Statistical Approaches（统计方法）
* 频繁共同出现的词更可能有taxonomic relationship
* 但这种方法非常依赖于特征类型的选取，正确率低

### Pattern-Based Methods（语言学方法）
* Based on hypernym-hyponym pairs from the corpus（基于词汇语义结构）
* 因为语言结构的多样性和组成的不确定性，通过特别的pattern进行寻找，使得coverage低，正确率低

### Word Embedding（词嵌入）
* 主要集中在通过词共生（word co-occurrence），来学习word embedding
* 因此，相似的词往往有相似的embedding
* 然而，此方法对于identify taxonomic relations效果差


### Clustering-Based Methods（无监督学习）

### Supervised Methods（监督学习）
* 训练集不可能包含所有的taxonomic relations，所以一定存在缺点

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

## [Customized organization of social media contents using focused topic hierarchy](https://dl.acm.org/citation.cfm?id=2661829.2661896)

### Abstract

* A novel function to measure the likelihood of a topic hierarchy
* A probability based model to identify the representative contents for topics 


### Introduction

* Information overload and noise —— organizing the social media contents into a general topic hierarchy.
* Find useful contents is time-consuming —— identify representative contents for each node on the hierarchy.
* Step:
    * Step1: use propagation algorithm to collect the potentially useful topics.
    * Step2: devise a function to estimate likelihood of a topic hierarchy and use it to integrate topic hierarchy, which fit social media corpus and user need.
    * Step3: propose a probability based ranking model



### Related work

### Methodology

### Evaluation

---

## [Learning Term Embeddings for Hypernymy Identification](http://pdfs.semanticscholar.org/0fc0/33f32f420ed3ff4330f60ccd0686db3deaea.pdf)

### Contributions

* We introduce a dynamic distance margin model to learn term embeddings that capture hypernymy properties, and we train an SVM classifier for hypernymy identification using the embeddings as features.
* 之前Hypernymy Identification的工作主要基于词汇模式（lexical pattern）和分布假设（distributional inclusion hypothesis），准确率不高。
* 现在设计了一种distance-margin neural network ，通过提前获得的上位关系数据，来学习term embedding。然后把获得的term embedding作为特征，通过有监督的方法来识别上下位关系。
* 原先人们非常关注从 term co-occurrence data 学习，因此相似的词语往往有相似的 embedding，但是我们需要判别的是两个词语是否有特定的关系，而不是经常一起出现。
* 更高的准确率，not domain dependent



### Methods

**1. Dynamic distance-margin model**
  * **embedding for the hypernymy relationship**
    * 每个term有两个embedding，一个hyponym embedding O，一个hypernym embedding E
    * 所有的embedding 满足三个性质：
      1. hyponym-hypernym similarity
      2. co-hyponym similarity
      3. co-hypernym similarity

  * **learning embedding**
    * hypernymy relationship：x=（v,u,q），v hypernyms，u hyponyms
    * 目标：O(u)接近E(v)



  * **neural network architecture**

**2. Supervised hypernymy identification**
  * f(x)<Δ，确定阈值Δ比较困难
  * 因此使用SVM，输入特征为O(u)、E(v)、O(u)-E(v)




---

## [Learning Term Embeddings for Taxonomic Relation Identification Using Dynamic Weighting Neural Network](http://www.aclweb.org/anthology/D16-1039)

### Contributions
1. For this purpose, we first design a dynamic weighting neural network to learn term embeddings based on not only the hypernym and hyponym terms, but also the contextual information between them. （提出用dynamic weighting neural network学习word embedding）
2. We then apply such embeddings as features to identify taxonomic relations using a supervised method.（用得到的embedding为特征，用SVM进行分类）

### Methods
* **和Yu et al.（2015）的工作很像，distance-margin neural network（这个可以看一下）**

**1. Learning Term Embedding**

* Extracting taxonomic relations
  * 用WordNet hierarchies获得所有的taxonomic relations，去掉其中的top-level terms
* Extracting training triples
  * 维基百科上获得含有taxonomic relations词语的句子，除了taxonomic relations词语，句子中其他词语都是context
  * we use the Stanford parser (Manning et al., 2014) to parse it, and check whether there is any pair of terms which are nouns or noun phrases in the sentence having a taxonomic relationship. 

* Training neural network
  * Specifically, the target of the neural network is to predict the hypernym term from the given hyponym term and contextual words.
  * 所有词用one-hot表示成向量（这个或许能改进）
  * 根据context数量动态调整
  * Softmax
  


**2. Supervised Taxonomic Relation Identification**
* embedding作为特征
* 输入SVM的特征（x, y, x-y）三个维度


### Datasets
**1. BLESS**
It covers 200 distinct, unambiguous concepts (terms); each of which is involved with terms, called relata, in some relations.
**2. ENTAILMENT**
It consists of 2,770 pairs of terms, with equal number of positive and negative examples of taxonomic relations.
**3. Animal, Plant and Vehicle datasets**
They are taxonomies constructed based on the dictionaries and data crawled from the Web for the corresponding domains.

### Compared Methods
1. SVM + Our
2. SVM + Word2Vec
3. SVM + Yu


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

# Taxonomic-Relation-Identification
This is a note for constructing taxonomy. 

---

## 总结研究

### Statistical Approaches（统计方法）
* 频繁共同出现的词更可能有taxonomic relationship
* 但这种方法非常依赖于特征类型的选取，正确率低

### Pattern-Based Methods（语言学方法）
* Based on hypernym-hyponym pairs from the corpus（基于词汇语义结构）
* 因为语言结构的多样性和组成的不确定性，通过特别的pattern进行寻找，使得coverage低，正确率低

### Word Embedding（词嵌入）
* 主要集中在通过词共生（word co-occurrence），来学习word embedding
* 因此，相似的词往往有相似的embedding
* 然而，此方法对于identify taxonomic relations效果差


### Clustering-Based Methods（无监督学习）

### Supervised Methods（监督学习）
* 训练集不可能包含所有的taxonomic relations，所以一定存在缺点

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

## [Customized organization of social media contents using focused topic hierarchy](https://dl.acm.org/citation.cfm?id=2661829.2661896)

### Abstract


### Introduction

### Related work

### Methodology

### Evaluation

---

## [Learning Term Embeddings for Hypernymy Identification](http://pdfs.semanticscholar.org/0fc0/33f32f420ed3ff4330f60ccd0686db3deaea.pdf)

### Contributions

* We introduce a dynamic distance margin model to learn term embeddings that capture hypernymy properties, and we train an SVM classifier for hypernymy identification using the embeddings as features.
* 之前Hypernymy Identification的工作主要基于词汇模式（lexical pattern）和分布假设（distributional inclusion hypothesis），准确率不高。
* 现在设计了一种distance-margin neural network ，通过提前获得的上位关系数据，来学习term embedding。然后把获得的term embedding作为特征，通过有监督的方法来识别上下位关系。
* 原先人们非常关注从 term co-occurrence data 学习，因此相似的词语往往有相似的 embedding，但是我们需要判别的是两个词语是否有特定的关系，而不是经常一起出现。
* 更高的准确率，not domain dependent



### Methods

**1. Dynamic distance-margin model**
  * **embedding for the hypernymy relationship**
    * 每个term有两个embedding，一个hyponym embedding O，一个hypernym embedding E
    * 所有的embedding 满足三个性质：
      1. hyponym-hypernym similarity
      2. co-hyponym similarity
      3. co-hypernym similarity

  * **learning embedding**
    * hypernymy relationship：x=（v,u,q），v hypernyms，u hyponyms
    * 目标：O(u)接近E(v)



  * **neural network architecture**

**2. Supervised hypernymy identification**
  * f(x)<Δ，确定阈值Δ比较困难
  * 因此使用SVM，输入特征为O(u)、E(v)、O(u)-E(v)




---

## [Learning Term Embeddings for Taxonomic Relation Identification Using Dynamic Weighting Neural Network](http://www.aclweb.org/anthology/D16-1039)

### Contributions
1. For this purpose, we first design a dynamic weighting neural network to learn term embeddings based on not only the hypernym and hyponym terms, but also the contextual information between them. （提出用dynamic weighting neural network学习word embedding）
2. We then apply such embeddings as features to identify taxonomic relations using a supervised method.（用得到的embedding为特征，用SVM进行分类）

### Methods
* **和Yu et al.（2015）的工作很像，distance-margin neural network（这个可以看一下）**

**1. Learning Term Embedding**

* Extracting taxonomic relations
  * 用WordNet hierarchies获得所有的taxonomic relations，去掉其中的top-level terms
* Extracting training triples
  * 维基百科上获得含有taxonomic relations词语的句子，除了taxonomic relations词语，句子中其他词语都是context
  * we use the Stanford parser (Manning et al., 2014) to parse it, and check whether there is any pair of terms which are nouns or noun phrases in the sentence having a taxonomic relationship. 

* Training neural network
  * Specifically, the target of the neural network is to predict the hypernym term from the given hyponym term and contextual words.
  * 所有词用one-hot表示成向量（这个或许能改进）
  * 根据context数量动态调整
  * Softmax
  


**2. Supervised Taxonomic Relation Identification**
* embedding作为特征
* 输入SVM的特征（x, y, x-y）三个维度


### Datasets
**1. BLESS**
It covers 200 distinct, unambiguous concepts (terms); each of which is involved with terms, called relata, in some relations.
**2. ENTAILMENT**
It consists of 2,770 pairs of terms, with equal number of positive and negative examples of taxonomic relations.
**3. Animal, Plant and Vehicle datasets**
They are taxonomies constructed based on the dictionaries and data crawled from the Web for the corresponding domains.

### Compared Methods
1. SVM + Our
2. SVM + Word2Vec
3. SVM + Yu


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

