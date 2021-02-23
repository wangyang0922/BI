# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib


# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 新闻数据路径
news_data_path = os.path.join(root, 'data', 'news.csv')

# 停用词路径
stop_word_path = os.path.join(root, 'data', 'chinese_stopwords.txt')

# 停用词路径
corpus_path = os.path.join(root, 'data', 'corpus.pkl')

source = "新华社"
# 词向量维度
embedding_dim = 300

sample_total = 82871

batch_size = 64

epochs = 2

vocab_size = 50000
