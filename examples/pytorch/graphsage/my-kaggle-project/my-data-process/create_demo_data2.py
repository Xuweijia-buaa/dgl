#!/usr/bin/env python
# coding: utf-8

# ## 先以整套valid数据集。做训练/测试。最后整体换成test的
import polars as pl
import pandas as pd
import os
from tqdm import tqdm
import dgl
import torch
import gc
import pickle
from gensim.models import Word2Vec
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from sklearn.preprocessing import StandardScaler
from undirect_g import build_undirect_g

data_path = '/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/'

# 建图：只使用测试环境的item:
eage_file = os.path.join(data_path,"all_edges_without_duplicates_90m.pt")
edges_tensor = torch.load(eage_file)
df_pair=pd.DataFrame({
    'aid_x':edges_tensor[0],
    'aid_y':edges_tensor[1]
})

g,item2nodeId,revers_map=build_undirect_g(df_pair,dataset=1)  # 无向图
#g = dgl.graph((edges_tensor[0], edges_tensor[1]))  # nodeid就对应itemid,不需要映射了

# 设置path:
valid=False
if valid:
    test_path=os.path.join(data_path,'otto-train-and-test-data-for-local-validation/test.parquet')  # 截断的test session.基于此产生submission.含所有测试用户
    label_path = os.path.join(data_path,'otto-train-and-test-data-for-local-validation/test_labels.parquet') # 要预测的用户+行为 -> item
else:
    test_path=os.path.join(data_path,'otto-full-optimized-memory-footprint/test.parquet')
    label_path=None


del df_pair
gc.collect()

## Save
out_directory='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data_1/'

dgl.save_graphs(os.path.join(out_directory, "g.bin"), g)  # 把图单独存一下.   # 有向图。每个edge,对应1条边.

dataset = {  # 原始数据metadata,单独存一下。都是只用于eval
    "item2nodeId": item2nodeId,         # dict: g中item到nodeid的映射. 按nodeid的顺序 {aid:nodeid}
    'test_path': test_path,             # test session,用来预测已有的
    "labels_path": label_path,          # df:每个用户的ground_truth。 用来评估。 所有item都在g中。用户同test
    'revers_map':revers_map             # 无向图，id到reverse_e_id的映射
}

with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
    pickle.dump(dataset, f)


