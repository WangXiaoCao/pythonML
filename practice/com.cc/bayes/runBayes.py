#!/usr/bin/python
# -*- coding:utf-8 -*-

import bayes
from numpy import *
import re


# list_post, list_class = bayes.load_data_set()
# vocab_list = bayes.create_vocab_list(list_post)
# word2vec_0 = bayes.set_word2vec(vocab_list, list_post[0])
# print vocab_list
# print word2vec_0

# train_mat = []
# for line in list_post:
#     train_mat.append(bayes.set_word2vec(vocab_list, line))
#
#
# p0_v, p1_v,p_ab = bayes.train_nb(train_mat, list_class)
# print p_ab
# print p0_v
# print p1_v
#
# a = [1,2,3]
# print sum(a)

# a = bayes.testing_nb()
# print a
a = 'my name is hehehaha!'
# print a.split()

reg = re.compile('\w*')
token = reg.split(a)
print token

