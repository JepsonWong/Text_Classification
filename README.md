# 文本分类

## 1. 收集数据

数据集：

任务：

## 2. 特征工程

### 2.1 预处理

* 删除不想关的字符，例如换行符(中文)、任何非字母数字字符(英文)等。
* 将文本进行分词。
* 删除不想关的单词，如停用词、一些网址(中文、英文)、提取词干(英文)。
* 将所有字符转换为小写，以处理诸如“hello”、“Hello”和“HELLO”等单词(英文)。
* 考虑将拼错的单词或拼写单词组合成一类(如：“cool”/“kewl”/“cooool”)(英文)。
* 考虑词性还原(将「am」「are」「is」等词语统一为常见形式「be」)(英文)。

### 2.2 特征生成

#### 2.2.1 unigram

#### 2.2.2 bigram

#### 2.2.3 trigram

### 2.3 特征选择

#### 2.3.1 互信息

#### 2.3.2 gbdt

#### 2.3.3 tf\_idf

### 2.4 降维

#### 2.4.1 svm

#### 2.4.2 lda 线性判别分析

#### 2.4.3 lda 文档主题模型

## 3. 数据表示：生成特征矩阵

### 3.1 one\_hot encoding(词袋模型)(符号表示)

离散、高维、稀疏

### 3.2 分布表示

连续、低维、稠密；**针对词、短语、句子、篇章的分布表示**；便于计算语言单元之间的距离和关系。

#### 3.2.1 词向量(word2vec等)

用一种形式的词向量；或者多种词向量进行结合，例如300维词向量，用3种词向量各训练100维，然后结合成300维。**多种词向量的结合**：Glove和Word2Vec各自产生的embedding可以同时作为输入层给supervised neural network；仿照CV里的术语，它们被称为不同的channel。

但无法解决一词多义、一义多词的问题，解决方式是用文档主题模型提取特征，或者为多义词的每一个词义学习一个词向量。

[一种基于词语多原型向量表示的句子相似度计算方法](https://www.xzbu.com/8/view-10943333.htm)

[learning sense-specific word embeddings by exploiting bilingual resources](http://anthology.aclweb.org/C/C14/C14-1048.pdf)

#### 3.2.2 doc2vec

## 4. 模型

### 4.1 gbdt

### 4.2 random\_forest

### 4.3 cnn

textcnn

### 4.4 rnn

textrnn

### 4.3 多种模型结合

## 5. 评估

### 5.1 混淆矩阵(Confusion Matrix)

### 5.2 AUC

## 6. 论文

### Multi-Task Label Embedding for Text Classification

17年 

以往的文本分类任务中，标签信息是作为无实际意义、独立存在的one-hot编码形式存在。这种做法会造成部分潜在语义信息丢失。本文将文本分类任务中的标签信息转换成含有语义信息的向量，将文本分类任务转换城向量匹配任务，并且建立了有监督、无监督和半监督三种模型。解决了以往模型无法迁移、无法扩展缩放和部分信息缺失这些问题。

[阅读笔记：Multi-Task Label Embedding for Text Classification](https://blog.csdn.net/tangpoza/article/details/85002646#_12)

### Learning Structured Text Representations

https://github.com/nlpyang/structured

https://github.com/vidhishanair/structured-text-representations

https://arxiv.org/pdf/1705.09207.pdf

**让AI当法官比赛第一名使用了论文Learning Structured Text Representations中的模型**

### 2018-文本分类文献阅读总结

https://www.cnblogs.com/demo-deng/p/9609767.html

## 7. 代码

[文本建模、文本分类相关开源项目推荐（Pytorch实现）](https://www.cnblogs.com/d0main/p/9462954.html)

## 8. 参考

[cnn\_rnn文本分类](https://github.com/gaussic/text-classification-cnn-rnn)
