# Pytorch-NLP-
参考[zhendong/finch](https://github.com/zhedongzheng/finch/blob/master/README.md#question-answering%E9%97%AE%E9%A2%98%E5%9B%9E%E7%AD%94) 在此基础上重新用pytorch实现nlp多种任务，并进行大量扩展和解释

一、word embedding
  1. word2vec（上）：分别学会用gensim的word2vec和pytorch的nn.Embedding来训练词向量
  2. word2cec（下）：手动实现word2vec， 包括CBOW和Skip-gram， 以及Huffman树的优化解法，其中Huffman树解法涉及Huffman树的建立、Huffman编码的生成以及Huffman树的中间节点的编码优化
  
