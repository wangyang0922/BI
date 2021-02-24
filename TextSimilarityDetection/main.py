# -*- coding:utf-8 -*-
import jieba
import pandas as pd
import nltk
from TextSimilarityDetection.utils.config import news_data_path, stop_word_path,corpus_path,source
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
# 引入日志配置
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    sentence = sentence.replace(' ','')
    sentence = sentence.replace("/n", '')
    sentence = jieba.cut(sentence.strip())
    return ' '.join([w for w in sentence if w not in stop_words])


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_word_path)


def load_corpus_file(file_path,data):
    '''
    加载文件
    :param file_path:停用词路径
    :return: 停用词表 list
    '''
    if not os.path.exists(file_path):
        # 可优化为线程处理
        res = list(map(split_sentence_proc, [i for i in data.content]))
        with open(file_path, 'rb') as file:
            pickle.dump(res, file)
    else:
        with open(file_path, 'rb') as file:
            res = pickle.load(file)
    return res


def find_similar_text(cpindex, top=10):
    # Step8，使用编辑距离 edit distance，计算两篇文章的距离
    dist_dict = {i: cosine_similarity(data[cpindex], data[i]) for i in class_id_table[id_class_table[cpindex]]}
    return sorted(dist_dict.items(), key=lambda x: [1][0], reverse=True)[:top]


if __name__ == '__main__':
    # Step1，数据加载
    news = pd.read_csv(news_data_path, encoding="gb18030")

    # Step2，数据预处理
    news = news.dropna(subset=['content'])
    print('news data size {}'.format(len(news)))
    corpus = load_corpus_file(corpus_path, news)
    labels = list(map(lambda v: 1 if source in str(v) else 0, news.source))  # 标记是否是自己的新闻
    print('label data size {}'.format(len(labels)))

    # Step3，提取文本特征TF-IDF
    count_vectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
    count_vectorizer = count_vectorizer.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer()
    data = tfidf_transformer.fit_transform(count_vectorizer)
    normalizer = Normalizer()
    scaled_arr = normalizer.fit_transform(data.toarray())    # 对tfidf特征进行规范化
    X_train, X_test, y_train, y_test = train_test_split(scaled_arr, labels, test_size=0.3)  # 切分数据集
    print('X_train data size {}, X_test data size {}, y_train data size {}, y_test data size {}'.format(len(X_train),len(X_test),len(y_train),len(y_test)))

    # Step4，预测文章风格是否和自己一致
    model = MultinomialNB()  # 朴素贝叶斯--多项式贝叶斯训练模型
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)  # 预测
    compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': y_test})
    print(compare_news_index.head())

    # Step5，找到可能Copy的文章，即预测label=1，但实际label=0
    copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index

    # Step6，根据模型预测的结果来对全量文本进行比对，如果数量很大，我们可以先用k - means进行聚类降维，比如k = 25种聚类
    kmeans = KMeans(n_clusters=25)  # 使用K-means对文章进行聚类(聚为25类)
    k_labels = kmeans.fit_transform(data.toarray())
    id_class_table = {i: tuple(v) for i, v in enumerate(k_labels)}  # 创建id_class
    source_news_index = compare_news_index[(compare_news_index['labels'] == 1)].index  # 实际为新华社的新闻
    class_id_table = defaultdict(set)
    for i, v in id_class_table.items():
        if i in source_news_index.tolist():  # 只统计source（新华社）发布class—id
            class_id_table[v].add(i)

    cpi = 3352
    # Step7，找到一篇可能的Copy文章，从相同label中，找到对应新华社的文章，并按照TF - IDF相似度矩阵，从大到小排序，取Top10
    similar_list = find_similar_text(cpi)
    print(similar_list)
    print("可能抄袭：\n",news.iloc[cpi].content)
    # 找一相似
    similar2 = similar_list[0][0]
    print('相似原文：\n',news.iloc[similar2].content)
    # Step8，使用编辑距离editdistance，计算两篇文章的距离

    # Step9，精细比对，对于疑似文章与原文进行逐句比对，即计算每个句子的编辑距离editdistance


