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

[《从零开始学习自然语言处理(NLP)》-基础准备(0)](https://zhuanlan.zhihu.com/p/54731715)

[深度学习文本分类在支付宝投诉文本模型上的应用](https://zhuanlan.zhihu.com/p/42236712)

[达观数据：如何用深度学习做好长文本分类与法律文书智能化处理](https://zhuanlan.zhihu.com/p/46331902)

[中文文本分类：你需要了解的10项关键内容](https://zhuanlan.zhihu.com/p/47761862)

[NLP概述和文本自动分类算法详解](https://zhuanlan.zhihu.com/p/40651970)

[达观数据：文本大数据的机器学习自动分类方法](https://zhuanlan.zhihu.com/p/24256814)

[达观数据情感分析架构演进](https://zhuanlan.zhihu.com/p/27068121)

[文本关键词提取算法解析](https://zhuanlan.zhihu.com/p/33605700)

[cnn\_rnn文本分类](https://github.com/gaussic/text-classification-cnn-rnn)

https://blog.csdn.net/Koala_Tree/article/details/77765436

https://blog.csdn.net/selinda001/article/details/80446423

https://blog.csdn.net/luoyexuge/article/details/78398782?yyue=a21bo.50862.201879
https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=0&rsv_idx=1&tn=baidu&wd=rnn%2Bcnn%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB&rsv_pq=ed4bcd4400004897&rsv_t=07cbOvkjSEjVUvhMbX%2Bcj3ff%2FjxXdtX5Kl3yRH%2B%2F3jUpAzEjEQ7Gw1OZ83w&rqlang=cn&rsv_enter=1&rsv_sug3=8&rsv_sug1=6&rsv_sug7=101&rsv_sug2=0&inputT=9712&rsv_sug4=9711

http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/

https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=0&rsv_idx=1&tn=baidu&wd=tensorflow%20shuffle%E6%80%8E%E4%B9%88%E5%81%9A%E7%9A%84&rsv_pq=95900a460002e423&rsv_t=3bd86cnLRpqfS5MegiMMdc7rVc5eivDFxfNc5LbWsN9I6uq1c43CX9iDXEo&rqlang=cn&rsv_enter=1&rsv_sug3=11&rsv_sug1=4&rsv_sug7=100&rsv_sug2=0&inputT=10116&rsv_sug4=10116
https://blog.csdn.net/cyningsun/article/details/7545679

https://www.zhihu.com/question/50888062

https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=0&rsv_idx=1&tn=baidu&wd=%E7%94%A8svm%E8%AE%AD%E7%BB%83%E5%87%BA%E6%9D%A5%E7%9A%84%E6%A8%A1%E5%9E%8B%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E4%B8%AA%E6%95%B0%E5%92%8C&rsv_pq=b318cad00000c658&rsv_t=21604ya8ds5KJZveW1B54C8sbBnqCzQ81cNfE4bXZ68wXRKIuhcdkytTXh4&rqlang=cn&rsv_enter=1&rsv_sug3=7&rsv_sug1=2&rsv_sug7=100&rsv_sug2=0&inputT=16968&rsv_sug4=16968

https://zhuanlan.zhihu.com/p/34212945

https://zhuanlan.zhihu.com/p/39774203

https://blog.csdn.net/xiaodongxiexie/article/details/76229042

https://blog.csdn.net/babybirdtofly/article/details/72886879

https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=1&tn=baidu&wd=svm.SVC%20%E5%A4%9A%E5%88%86%E7%B1%BB&oq=svm%2520%25E5%25A4%259A%25E5%2588%2586%25E7%25B1%25BB&rsv_pq=bc649ab20006b7e2&rsv_t=d372i9pNO1lzAtYYuXxgxkEbXIMTVD0kbD6ruCgR3YQfO32U15c4TSXofgQ&rqlang=cn&rsv_enter=1&inputT=1167&rsv_n=2&rsv_sug3=6&rsv_sug2=0&rsv_sug4=1167

https://blog.csdn.net/T7SFOKzorD1JAYMSFk4/article/details/80269129


https://github.com/gaussic/text-classification-cnn-rnn

TextRCNN: recurrent convolutional nerual networks for text classification

2017知乎看山杯总结(多标签文本分类): https://blog.csdn.net/Jerr__y/article/details/77751885

```
Convolutional Methods for Text: https://weibo.com/1402400261/F4nWcmOMi?sudaref=www.google.com&display=0&retcode=6102&type=comment#_rnd1548677150694
```

THUCTC: 一个高效的中文文本分类工具包: http://thuctc.thunlp.org/

入门 | 自然语言处理是如何工作的？一步步教你构建 NLP 流水线: http://dy.163.com/v2/article/detail/DP0RI1MU0511AQHO.html

融合多种embedding：
Improving AI language understanding by combining multiple word representations: https://code.fb.com/ai-research/dynamic-meta-embeddings/
Dynamic Meta-Embeddings for Improved Sentence Representations: https://blog.csdn.net/qq_32782771/article/details/85067849
https://www.google.com/search?q=Dynamic+Meta-Embeddings+for+Improved+Sentence+Representations&oq=Dynamic+Meta-Embeddings+for+Improved+Sentence+Representations&aqs=chrome..69i57j69i60j0.255j0j4&sourceid=chrome&ie=UTF-8

A Benchmark of Text Classification in PyTorch: https://github.com/pengshuang/TextClassificationBenchmark
FastText
BasicCNN (KimCNN,MultiLayerCNN, Multi-perspective CNN)
InceptionCNN
LSTM (BILSTM, StackLSTM)
LSTM with Attention (Self Attention / Quantum Attention)
Hybrids between CNN and RNN (RCNN, C-LSTM)
Transformer - Attention is all you need
ConS2S
Capsule
Quantum-inspired NN

DPCNN做文本分类《Deep Pyramid Convolutional Neural Networks for Text Categorization》: https://blog.csdn.net/u014475479/article/details/82081578
文本分类问题不需要ResNet？小夕解析DPCNN设计原理（上）: https://cloud.tencent.com/developer/news/169649
从DPCNN出发，撩一下深层word-level文本分类模型: https://zhuanlan.zhihu.com/p/35457093

HAN文本分类: https://www.google.com/search?q=HAN+%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB&oq=HAN+%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB&aqs=chrome..69i57j0.5265j0j7&sourceid=chrome&ie=UTF-8
文献阅读笔记：Hierarchical Attention Networks for Document Classification: https://www.jianshu.com/p/37422ce8b2d7

深度学习与文本分类总结第一篇--常用模型总结: https://blog.csdn.net/liuchonge/article/details/77140719
文本分类实战--从TFIDF到深度学习（附代码）: https://blog.csdn.net/liuchonge/article/details/72614524

深度学习与文本分类总结第二篇--大规模多标签文本分类: https://blog.csdn.net/liuchonge/article/details/77585222 博客里有DPCNN、HAN等的复现

```
Investigating Capsule Networks with Dynamic Routing for Text Classification: 胶囊网络在文本分类中的应用: https://zhuanlan.zhihu.com/p/51008729
教程 | 可视化CapsNet，详解Hinton等人提出的胶囊概念与原理: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650740517&idx=2&sn=1cf855299c42bcc930265f2f93696a12&chksm=871ad35bb06d5a4dd7cf445172332a4625806d28a4fefe4eefcf9e1b9a81e9106e9dd00a3fa6&scene=21#wechat_redirect
胶囊网络（Capsule Network）在文本分类中的探索: https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/79825597
Capsule官方代码开源之后，机器之心做了份核心代码解读: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650737203&idx=1&sn=43c2b6f0e62f8c4aa3f913aa8b9c9620&chksm=871ace4db06d475be8366969d74c4b2250602f5e262a3f97a5faf2183e53474d3f9fd6763308&scene=21#wechat_redirect
浅析Geoffrey Hinton最近提出的Capsule计划: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650731207&idx=1&sn=db9b376df658d096f3d1ee71179d9c8a&chksm=871b36b9b06cbfafb152abaa587f6730716c5069e8d9be4ee9def055bdef089d98424d7fb51b&scene=21#wechat_redirect
终于，Geoffrey Hinton那篇备受关注的Capsule论文公开了: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650732472&idx=1&sn=259e5aa77b62078ffa40be9655da0802&chksm=871b33c6b06cbad0748571c9cb30d15e9658c7509c3a6e795930eb86a082c270d0a7af1e3aa2&scene=21#wechat_redirect
```

Graph Convolutional Networks for Text Classification: https://github.com/yao8839836/text_gcn
用于文本分类的图形卷积网络(Graph Convolutional Networks for Text Classification): http://www.tuan18.org/thread-13271-1-1.html
SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS: https://zhuanlan.zhihu.com/p/49541317

Recurrent-Convolutional-Neural-Network-Text-Classifier: https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier

Learning Structured Representation for Text Classification via Reinforcement Learning: https://github.com/keavil/AAAI18-code

Generative Adversarial Network for Abstractive Text Summarization: https://github.com/iwangjian/textsum-gan

Learning Deep Latent Spaces for Multi-Label Classifications: https://github.com/chihkuanyeh/C2AE

Explicit Interaction Model towards Text Classification: https://github.com/NonvolatileMemory/AAAI_2019_EXAM

Hierarchical Attention Transfer Network for Cross-domain Sentiment Classification: https://github.com/hsqmlzno1/HATN

HARP: Hierarchical Representation Learning for Networks: https://github.com/GTmac/HARP

AI Challenger 2018 文本挖掘类竞赛相关解决方案及代码汇总: https://zhuanlan.zhihu.com/p/51462820

AI-Challenger Baseline 细粒度用户评论情感分析:https://github.com/pengshuang/AI-Comp

AI Challenger 2018：细粒度用户评论情感分类冠军思路总结: https://zhuanlan.zhihu.com/p/55887135

如何到top5%？NLP文本分类和情感分析竞赛总结: https://zhuanlan.zhihu.com/p/54397748

"中国法研杯"司法人工智能挑战赛: https://github.com/thunlp/CAIL
https://github.com/thunlp/CAIL2018
https://arxiv.org/pdf/1810.05851.pdf

A Hierarchical Neural Attention-based Text Classifier: https://www.google.com/search?q=A+Hierarchical+Neural+Attention-based+Text+Classifier&oq=A+Hierarchical+Neural+Attention-based+Text+Classifier&aqs=chrome..69i57j69i60l2j69i64l2.224j0j4&sourceid=chrome&ie=UTF-8
http://www.aclweb.org/anthology/D18-1094

BDCI_Car_2018: https://github.com/yilirin/BDCI_Car_2018

Dimensional Sentiment Analysis Using a Regional CNN-LSTM Model
deep convolutional neural networks for sentiment analysis of short texts
两个基于神经网络的情感分析模型: https://blog.csdn.net/youngair/article/details/78013352

CNN用于文本分类综述: https://zhuanlan.zhihu.com/p/55946246

香侬科技提出中文字型的深度学习模型Glyce，横扫13项中文NLP记录: https://zhuanlan.zhihu.com/p/56012870

深度学习第48讲：自然语言处理之情感分析: https://zhuanlan.zhihu.com/p/54029827

深度学习在文本分类中的应用: https://zhuanlan.zhihu.com/p/34383508

python | sklearn ，做一个调包侠来解决新闻文本分类问题: https://zhuanlan.zhihu.com/p/30455047

A Structured Self-Attentive Sentence Embedding
https://github.com/facebookresearch/pytext?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more

https://github.com/IsaacChanghau/DL-NLP-Readings/blob/master/readme/nlp/datasets.md

https://github.com/susht3/Text_Mutil_Classification_keras


