# coding: utf-8

# # 二、学会自己编写实现word2vec

# 参考博文：https://blog.csdn.net/u014595019/article/details/51884529
# 
# 感觉这篇说的最清楚
# 
# 需要说明的是，当语料较少时使用CBOW方法比较好，当语料较多时采用skip-gram表示比较好。

# ![image.png](attachment:image.png)

# # 1. 构建数据
# 步骤：
# 1. 将分词后的文本处理成词典、word2ix、ix2word，并因此将原文中的所有词映射成对应的词向量，从而将原文变成一个词向量。其中需要处理掉低频词
# 2. 然后遍历原文词向量，开始制造输入和label的格式：输入为每个词的词向量，label为其窗口大小内的上下文词向量矩阵


class Config(object):
    data_path = './data/corpus/in_the_name_of_people.txt'
    jieba_path = './data/corpus/JieBa2_in_the_name_of_people.txt'
    min_freq = 5
    window = 5
    embed_dim = 100
    batch_size = 100
    epochs = 10
    word_dict = None
    learn_rate = 0.025


opt = Config()

import jieba
import re
from collections import Counter


def precess_raw(opt):
    with open(opt.data_path, encoding='utf8') as f:
        txt = f.read()
        txt_cut = jieba.cut(txt)
        txt_cut = ' '.join(txt_cut)
        txt_re = re.sub('\n', ' ', txt_cut)
        txt_re = re.sub('\s+', ' ', txt_re)
        with open(opt.jieba_path, 'w', encoding='utf8') as f2:
            f2.write(txt_re)
        f.close()
        f2.close()


def process_data():
    if not opt.jieba_path:
        process_raw()
    with open(opt.jieba_path, encoding='utf8') as f:
        txt = f.read()
        txt_list = txt.split(' ')
        words_count = Counter(txt_list)

        words = [w for w, c in words_count.items() if c > opt.min_freq]
        word_dict = {
        w: {'word': w, 'count': words_count[w], 'ix': i, 'vec': np.random.random([1, opt.embed_dim]), 'Huffman': None}
        for i, w in enumerate(words)}

        opt.word_dict = word_dict

        txt_list = [w for w in txt_list if w in words]
        #         txt=[word_dict[w]['ix'] for w in txt_list]
        return txt_list


def get_y(txt, i):
    # win=np.random.randint(1, opt.skip_window+1)
    win = opt.window
    left = i - win if i - win > 0 else 0
    right = i + win
    return txt[left:i] + txt[i + 1:right]


def make_data():
    inputs = []
    targets = []
    txt_list = process_data()
    for i in range(len(txt_list)):
        input_w = txt_list[i]
        target_w = get_y(txt_list, i)
        inputs.append(input_w)
        targets.append(target_w)
    return inputs, targets


# # 2. 输入层到隐藏层

# 上步process_data函数中直接使用np.random.random初始化了每个词的词向量

# # 3. 隐藏层到输出层

# ## 3.1 两种模型

# ## 3.1.1. Skip
# 输入是一个特定的一个词对应的上下文词向量，输出是该词的词向量

# 映射层到输出层：值沿着Huffman树不断的进行logistic分类，并且不断的修正各中间向量和词向量。
# ![image.png](attachment:image.png)
# 此时中间的单词为w(t)，而映射层输入为 
# pro(t)=v(w(t-2))+v(w(t-1))+v(w(t+1))+v(w(t+2))
# 
# 假设此时的单词为“足球”，即w(t)=“足球”，则其Huffman码可知为d(t)=”1001”(具体可见上一节),那么根据Huffman码可知，从根节点到叶节点的路径为“左右右左”，即从根节点开始，先往左拐，再往右拐2次，最后再左拐。
# 
# 既然知道了路径，那么就按照路径从上往下依次修正路径上各节点的中间向量。在第一个节点，根据节点的中间向量Θ(t,1)和pro(t)进行Logistic分类。如果分类结果显示为0，则表示分类错误(应该向左拐，即分类到1)，则要对Θ(t,1)进行修正，并记录误差量。
# 
# 接下来，处理完第一个节点之后，开始处理第二个节点。方法类似，修正Θ(t,2)，并累加误差量。接下来的节点都以此类推。
# 
# 在处理完所有节点，达到叶节点之后，根据之前累计的误差来修正词向量v(w(t))。
# 
# 这样，一个词w(t)的处理流程就结束了。如果一个文本中有N个词，则需要将上述过程在重复N遍，从w(0)~w(N-1)。

# 基本流程：
# ![image.png](attachment:image.png)

# 其中：
# ![image.png](attachment:image.png)

# 首先将上下文词的词向量累加到一个向量上，作为输入，然后沿着Huffman树向下根据词的Huffman编码修正树中的节点向量，并累加误差，最后修正叶子节点的词的词向量，最最后修正输入的上下文中每个词的词向量

# ## 3.1.2. CBOW
# 输入是一个特定的一个词的词向量，输出是特定此对应的上下文词向量

# 由于输出为多个词，所以需要在Huffman树上循环多遍。每次找一个词。
# 
# 伪代码如下：
# ![image.png](attachment:image.png)


class ToOutLayer():
    def __init__(self, huffman, one_words, win_contexts, method='CBOW'):
        super(ToOutLayer, self).__init__()
        self.huffman = huffman
        self.one_words = one_words
        self.win_contexts = win_contexts
        self.method = method
        self.word_dict=None
        self.__forward()

    def __Sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def __Deal_Gram_Skip(self, one_word, win_context,count):
        if one_word not in opt.word_dict:
            return 'Error'
        word_huffman = opt.word_dict[one_word]['Huffman']
        win_context_vec = np.zeros([1, opt.embed_dim])
        for i in range(len(win_context))[::-1]:
            w = win_context[i]
            if w not in opt.word_dict:
                win_context.pop()
            else:
                win_context_vec += opt.word_dict[w]['vec']
        if not win_context:
            return
        e = self.__GoAlong_Huffman(word_huffman, win_context_vec)

        for w in win_context:
            opt.word_dict[w]['vec'] += e
            opt.word_dict[w]['vec'] = preprocessing.normalize(opt.word_dict[w]['vec'])  # 修正词向量
        #if count/10 == 0:
            #print('e:', count, np.average(e))

    def __Deal_Gram_CBOW(self, one_word, win_context,count):
        if one_word not in opt.word_dict:
            return 'Error'
        one_vec = opt.word_dict[one_word]['vec']
        for i in range(len(win_context))[::-1]:
            w = win_context[i]
            if w not in opt.word_dict:
                win_context.pop()

        if not win_context:
            return
        for w in win_context:
            w_huffman = opt.word_dict[w]['Huffman']
            e = self.__GoAlong_Huffman(w_huffman, one_vec)
            opt.word_dict[w]['vec'] += e
            opt.word_dict[w]['vec'] = preprocessing.normalize(opt.word_dict[w]['vec'])  # 修正词向量
        #if count/10 == 0:
            #print('e:', count, np.average(e))

    def __GoAlong_Huffman(self, word_huffman, input_vec):
        node = self.huffman.root
        e = np.zeros([1, opt.embed_dim])
        for j in range(len(word_huffman)):
            a_code = word_huffman[j]
            q = self.__Sigmoid(input_vec.dot(node.vec.T))  # 判断误差
            grad = opt.learn_rate * (1 - int(a_code) - q)  # 更新梯度
            e += grad * node.vec  # 累加误差
            node.vec += grad * input_vec  # 修正中间节点向量
            if a_code == '0':
                node = node.left
            else:
                node = node.right
        return e

    def __forward(self):
        if self.method == 'CBOW':
            func = self.__Deal_Gram_CBOW
        else:
            func = self.__Deal_Gram_Skip
        count = 0
        for one_word, win_context in zip(self.one_words, self.win_contexts):
            count += 1
            func(one_word, win_context, count)
        self.word_dict=opt.word_dict


# ## 3.2 两种优化方法

# ### 3.2.1 Huffman树的构建

# 伪代码：
# while (单词列表长度>1) {
#     从单词列表中挑选出出现频率最小的两个单词 ;
#     创建一个新的中间节点，其左右节点分别是之前的两个单词节点 ;
#     从单词列表中删除那两个单词节点并插入新的中间节点 ;
# }


class HuffmanTreeNode():
    def __init__(self, vec, count, word=None):
        self.count = count  # 该词词频
        self.left = None
        self.right = None
        self.vec = vec  # 该节点的词向量
        self.code = ''  # 该节点的Huffman编码
        self.word = word  # 叶子节点上的词本身, 非叶子节点上没有词所以默认为None


import numpy as np


class HuffmanTree():
    def __init__(self):
        #         self.embed_dim=embed_dim#词向量长度
        self.root = None
        # word_count_list=list(word_count.values())#所有词的词频
        word_list = list(opt.word_dict.values())
        node_list = [HuffmanTreeNode(w['vec'], w['count'], w['word']) for w in word_list]
        self.builf_tree(node_list)  # 建立Huffman树
        self.generate_code(self.root)  # 根据已经生成的树结构来为词典中的每个词产生对应的Huffman码

    def merge(self, node1, node2):  # 合并两个节点并产生他们的父节点
        count = node1.count + node2.count
        # vec=np.zeros(1,self.vec_len)#所有中间节点的向量初始为0
        vec = np.zeros([1, opt.embed_dim])
        node = HuffmanTreeNode(vec, count)
        if node1.count <= node2.count:
            node.left = node1
            node.right = node2
        else:
            node.left = node2
            node.right = node1
        return node

    def builf_tree(self, node_list):  # 构建Huffman树
        while len(node_list) > 1:
            i1 = 0  # i1表示概率最小的节点
            i2 = 1  # i2表示概率第二小的节点
            if node_list[i1].count > node_list[i2].count:
                i1, i2 = i2, i1
            for i in range(2, len(node_list)):
                if node_list[i].count < node_list[i2].count:
                    i2 = i
                    if node_list[i2].count < node_list[i1].count:
                        i1, i2 = i2, i1
            node = self.merge(node_list[i1], node_list[i2])
            # print(i1,i2,node_list[i1].word, node_list[i2].word,node_list[i1].count, node_list[i2].count)
            if i1 < i2:  # 必须检查， 不然前一个pop了，后一个就不是原来的位置了
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2!')
            node_list.insert(0, node)
        self.root = node_list[0]

    def generate_code(self, node):
        # 左节点为1， 右节点为0
        stack = [node]
        while len(stack) > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.code
                node.left.code = code + '0'
                node.right.code = code + '1'
                stack.append(node.right)
                node = node.left

            word = node.word
            code = node.code
            opt.word_dict[word]['Huffman'] = code


# # 4. 主函数，即word2vec类

import torch as t
from torch.utils.data import DataLoader
from sklearn import preprocessing


# class word2vec():
#     def __init__(self):

def Train_word2vec():
    # 构建数据
    one_words, win_contexts = make_data()
    # 构建dataloader

    # print(len(one_words), len(win_contexts))
    # print(opt.word_dict['人民'])
    # 根据每个词的初始向量构建初始的Huffman树
    Tree = HuffmanTree()
    print('before',opt.word_dict['人民'])
    # 映射层到输出层
    model = ToOutLayer(Tree, one_words, win_contexts)  # 默认的方法为CBOW
    Out = model.word_dict
    print('after',opt.word_dict['人民'])
    print(type(Out))
    return Out

word_dict=Train_word2vec()

import csv

#python2可以用file替代open
with open("result/20181216.csv","w", encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["word","vec"])
    #写入多行用writerows
    writer.writerows([i, opt.word_dict[i]['vec']] for i in opt.word_dict)