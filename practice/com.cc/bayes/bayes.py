#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *
import feedparser


# # 创建数据集
def load_data_set():
    # 创建特征集：5篇评论
    posting_list = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 创建对应的类别集：0为非侮辱评论，1为侮辱评论
    class_vec = [0, 1,0, 1, 0, 1]
    # 返回这两个数据集
    return posting_list, class_vec


# # 创建词库：涵盖了出现过的所有词，词不重复，参数为文本的特征数据集
def create_vocab_list(data_set):
    # 创建一个空的set
    vocab_set = set([])
    # 对每条评论做遍历
    for document in data_set:
        # 将每条评论中的词去重，然后递归地合并到vocab_set中
        vocab_set = vocab_set | set(document)
    # 返回拥有所有词的列表
    return list(vocab_set)


# # 将词转换成one-hot数字向量形式。输入的参数时：词库列表，一篇文档（一个词向量）
def set_word2vec(vocab_list, input_set):
    # 创建一个与词库相同维度的0列表
    return_vec = [0] * len(vocab_list)
    # 对文档中的每个词遍历
    for word in input_set:
        # 如果这个词是在词库中有的
        if word in vocab_list:
            # 就将return_vec中该词所在词库的索引处，设置值为1
            return_vec[vocab_list.index(word)] = 1
        # 否则
        else:
            # 打印警告：改词不在词库中
            print "the word: %s is not in  my vocabulary!" % word
    # 返回这个文档或这篇评论的one-hot词向量
    return return_vec


# # 词袋向量
def bag_of_word2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            # 这是与上个方法中唯一不同的地方：即为出现的次数计数
            return_vec[vocab_list.index(word)] += 1
    return return_vec


# # 训练朴素贝叶斯
def train_nb(train_matrix, train_category):
    # 训练数据集中实例的总数
    num_train = len(train_matrix)
    # 特征的总数
    num_words = len(train_matrix[0])
    # 类别为1 先验概率
    class1_pro = sum(train_category)/float(num_train)
    # 创建与特征相同长度的0向量
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_demo = 2.0
    p1_demo = 2.0
    # 遍历训练数据集的实例
    for i in range(num_train):
        # 如果实例对应的类别为1
        if train_category[i] == 1:
            # 就把该实例的特征向量加上之前相同长度的的0向量（为了把同意类别的特征值全部相加）
            p1_num += train_matrix[i]
            # 并且，将该实例下的特征向量内部求和并累加
            p1_demo += sum(train_matrix[i])
        # 如果实例的类别为0
        else:
            # 做同样的操作
            p0_num += train_matrix[i]
            p0_demo += sum(train_matrix[i])
    # 给定类别为1，所有特征的出现概率的向量
    p1_vect = log(p1_num/p1_demo)
    # 给定类别为2，所有特征的出现概率的向量
    p0_vect = log(p0_num/p0_demo)
    return p0_vect, p1_vect, class1_pro


# # 做分类预测，输入参数：测试集的词向量，类别0的条件概率，类别1的条件概率，类别的先验概率
def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    #将词向量乘以对应的概率并相加（因为转换成了log，相加就是连乘），在加上先验概率
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


# # 将以上方法汇总，并做出测试
def testing_nb():
    list_posting, list_class = load_data_set()
    vocab_list = create_vocab_list(list_posting)
    train_mat = []
    for doc in list_posting:
        train_mat.append(set_word2vec(vocab_list, doc))
    p0_v, p1_v, p_class1 = train_nb(array(train_mat), array(list_class))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_word2vec(vocab_list, test_entry))
    print test_entry, 'classify as:', classify_nb(this_doc, p0_v, p1_v, p_class1)
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_word2vec(vocab_list, test_entry))
    print test_entry, 'classify as:', classify_nb(this_doc, p0_v, p1_v, p_class1)


# # 对文本进行分词与预处理
def text_parse(big_string):
    import re
    list_token = re.split(r'\w*', big_string)
    return [tok.lower() for tok in list_token if len(tok) < 2]


# # 对垃圾邮件进行训练贝叶斯，并做测试
def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        # 读取；某目录下的垃圾邮件
        word_list = text_parse(open('').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)
        # 读取某目录下的非垃圾邮件
        word_list = text_parse(open('').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)
    # 创建词库
    vocab_list = create_vocab_list(doc_list)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform((0, len(training_set))))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    training_class = []
    for doc_index in training_set:
        train_mat.append(set_word2vec(vocab_list, doc_list[doc_index]))
        training_class.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb(array(train_mat), array(training_class))
    error_count = 0
    for doc_index in test_set:
        word_vec = set_word2vec(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vec), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print 'the error rate is:', float(error_count)/len(test_set)






