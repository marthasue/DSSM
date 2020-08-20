# coding:utf-8
#############################################
# FileName: predict.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: get sentence vector
#############################################
import numpy as np
import codecs
import json
import os
from scipy.spatial import distance
from tensorflow.contrib import predictor
from tqdm import tqdm
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 读取字典文件
vocab = json.load(codecs.open("./char.json", "r", "utf-8"))
# 这里只用query做测试
model_q = "./host_model"

model_d = "./guest_model"
# 句子最大长度
max_size = 50

def sent2id(sent):
    # 将句子转换为id序列
    #print(sent)
    if isinstance(sent,float)==True:
        sent = '其'
    sent = [vocab.get(c, 1) for c in sent]
    sent = sent[:max_size] + [0] * (max_size - len(sent))
    return sent

def load_model(model_name):
    # 读取模型
    model_time = str(max([int(i) for i in os.listdir(model_name) if len(i)==10]))
    model = predictor.from_saved_model(os.path.join(model_name,model_time))
    return model

def get_vector(sentence, model,x,y):
    # 输入句子并转换为向量
    feed_dict = {x: [sent2id(sentence)]}
    vector = model(feed_dict)
    return vector[y][0]

def similar_index(sentence, file, max_sentence_num=10000, topn=10):
    # 输入一个句子和包含一系列句子的文件，找到文件中跟该句子最相似的N条
    modelq = load_model(model_q)
    modeld = load_model(model_d)
    source_vec = get_vector(sentence, modelq,"host_char","host_vector")

    df_excel = pd.read_excel(file,'Sheet1')
    df_guest = df_excel['guest']

    target_vec = dict()
    for i in df_guest.index:
        vec = get_vector(df_guest[i],modeld,"guest_char","guest_vector")
        target_vec[df_guest[i]] = 1.0-distance.cosine(source_vec,vec)
    rank = sorted(target_vec.items(),key=lambda e:e[1],reverse=True)
    # with codecs.open(file_path, "r", "utf-8") as fr:
    #     for line in tqdm(fr):
    #         line = line.strip().split("\t")
    #         vec = get_vector(line[0], model)
    #         target_vec[line[0]] = 1.0 - distance.cosine(source_vec, vec)
    # rank = sorted(target_vec.items(), key=lambda e:e[1], reverse=True)
    print("source: %s"%sentence)
    print("target: \n")
    for i in rank[:topn]:
        print("%s\t%s"%(round(i[1],6), i[0]))

similar_index("食品、饮料和烟草批发食品、饮料和烟草批发其他批发业其他批发业",
              "./data/pipei_data.xlsx",
              10000,
              10)
