#!/usr/bin/python
# -*- coding:utf-8 -*-


from math import log
import operator


# # 计算熵
def calc_shannon_ent(data_set):
    # 计算数据集中实例的总数
    num_entry = len(data_set)
    # 创建一个字典
    label_counts = {}
    for featVec in data_set:
        # 取出每个实例的最后一列的数据
        current_label = featVec[-1]
        # 为所有类别创建字典：label, count
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 初始化熵
    shannon_ent = 0.0
    # 计算熵
    for key in label_counts:
        # 计算每个类别的概率P
        prob = float(label_counts[key])/num_entry
        # 累减每个类别的（概率*log概率）
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# # 创建数据集与特征名称
def create_data_set():
    # 前面两列是两个特征值，后面一列是类别
    data_set = [[1,1,'y'],
               [1,1,'y'],
               [1,0,'n'],
               [0,1,'n'],
               [0,1,'n']]
    # 特征的名称
    labels = ['no surfacing','flippers']
    return data_set, labels


# # 分割数据集，第一个参数时样本数据，第二个参数是特征的索引，第三个参数时需要返回的特征的值
# # 也就是说将每个样本遍历一次，如果目标特征正好等于需求的value,那么就返回除了该特征值之外的所有数据
def split_data_set(data_set, axis, value):
    # 创建一个空的set
    ret_data_set = []
    # 遍历每一个样本
    for featVec in data_set:
        # 如果该样本中的目标特征 = 需要的value
        if featVec[axis] == value:
            # 就提取出这个特征之前的数据
            reduced_feat_vec = featVec[:axis]
            # 将这个特征之后的数据也追加进去
            reduced_feat_vec.extend(featVec[axis+1:])
            # 将得到的除特征外的数据集追加到刚刚创建的set中
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


# # 选择最优修的特征，对于C4.5决策数来说，就是求出信息增益最大的特征
def choose_best_feature(data_set):
    # 特征的个数
    num_feature = len(data_set[0]) - 1
    # 计算H(D)--样本集的熵
    base_entropy = calc_shannon_ent(data_set)
    # 初始化最大的信息增益，以及对应的特征的索引
    best_info_gian = 0.0; best_feature = -1
    # 对每个特征进行遍历
    for i in range(num_feature):
        # 取出一列特征
        feat_list = [example[i] for example in data_set]
        # 将特征值放进set里去重
        unique_vals = set(feat_list)
        # 初始化熵
        new_entropy = 0.0
        # 对每一类特征值进行遍历
        for value in unique_vals:
            # 对该特征的该value进行数据拆分
            sub_data_set = split_data_set(data_set, i, value)
            # 计算该特征下该value的样本数/总的样本数
            prob = len(sub_data_set)/float(len(data_set))
            # 以上概率* H（Di)
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # 信息增益 = 该特征下所有value下的newEntropy的累减
        info_gain = base_entropy - new_entropy
        # 如果该特征的信息增益大于之前的最高信息增益
        if info_gain > best_info_gian:
            # 就把新下信息增益赋值给最高的信息增益变量
            best_info_gian = info_gain
            # 并且对应的对佳的特征索引也替换为这个新的特征的索引
            best_feature = i
    # 返回具有最高信息增益的那个特征的索引
    return best_feature


# # 当特征全部用完的时候叶子节点上仍然有不同类别的样本，那么就取频次最多的类别作为最后的预测类别
# # 计算每个类别及其出现频次，最后取出频次最大的。
def majority_cont(class_list):
    # 创建一个存放类别与其计数的空字典
    class_count = {}
    # 对类别列表里的每个元素进行遍历
    for vote in class_list:
        # 如果在计数集合中没有该类别
        if vote not in class_count.keys():
            # 就添加字典[label,0]
            class_count[vote] = 0
        # 将对应的类别频次加1
        class_count[vote] += 1
    # 根据频次对类别进行降序排序
    sorted_class_cnt = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回频次最大的那个类别
    return sorted_class_cnt[0][0]


# # 重头戏：创建决策树
def create_tree(data_set, labels):
    # 取出数据集中的所有类别作为一个list
    class_list = [example[-1] for example in data_set]
    # 如果类别中的第一个类别相同的数量等于所有类别的数量（如果所有的类别标签完全相等）
    if class_list.count(class_list[0]) == len(class_list):
        # 就返回该类别，说明都分类一致，可以停止再分支了
        return class_list[0]
    # 如果数据集中的特征数量为0，只剩下一列类别标签了，说明所有的特征都用完了，但还有不同的类别
    if len(data_set[0]) == 1:
        # 就返回最大频次的类别
        return majority_cont(class_list)
    # 除了以上两种可能，就是特征还没有用完，类别也还有不同，需要继续分支
    # 找到信息增益最大的特征的索引
    best_feat = choose_best_feature(data_set)
    # 这个最优特征的名称
    best_feat_label = labels[best_feat]
    # 创建一个决策树的字典，key是最优的那个特征，value是它下面的所有树枝
    my_tree = {best_feat_label:{}}
    # 既然已经使用了这个特征，那么就把它从特征名称的列表中删掉，以后都不会再用了
    del(labels[best_feat])
    # 将这列特征从样本集中取出
    feat_values = [example[best_feat] for example in data_set]
    # 将特征的值去重，变成一个set
    unique_value = set(feat_values)
    # 对每一种特征值做遍历
    for value in unique_value:
        # 把删掉了已经被使用的特征之后的新的特征名称列表全部赋值给sub_label
        sub_labels = labels[:]
        # 将这一轮的得到的各值放到决策树中
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    # 返回决策树
    return my_tree


# # 使用训练好的决策树进行分类预测
# # 参数输入：决策树，特征的名称，测试样本
def classify(input_tree, feat_label, test_vec):
    # 取出决策树中的第一个特征名称
    first_str = input_tree.keys()[0]
    # 该特征之下的决策树，就是取决策树的value
    second_dict = input_tree[first_str]
    # 把这个特征的索引拿出来
    feat_index = feat_label.index(first_str)
    # 对该特征的所有取值进行遍历
    for key in second_dict.keys():
        # 如果测试样本中的该特征值等于某取值
        if test_vec[feat_index] == key:
            # 如果这个取值作为key,他的value类型是字典的话（说明该结点下面还有子结点）
            if type(second_dict[key]).__name__ == 'dict':
                # 继续循环做classify这个类的工作
                class_label = classify(second_dict[key], feat_label, test_vec)
            # 否则，如果value不是字典的话，说明是一个string类型的类别标记,将这个类别赋值给测试集
            else: class_label = second_dict[key]
    # 返回测试样本的类别
    return class_label


# # 存储决策树
def store_tree(input_tree, file_name):
    import pickle
    fw = open(file_name, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


# # 调用决策树
def grab_tree(file_name):
    import pickle
    fr= open(file_name)
    return pickle.load(fr)
