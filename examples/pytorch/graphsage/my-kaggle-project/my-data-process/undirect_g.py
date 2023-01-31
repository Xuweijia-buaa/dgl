#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

out_directory='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data/'
# 给自己做的数据集，构建反向边。以及对应的图
def build_undirect_g(df_pair,dataset=0):
    # aid_x，aid_y,w
    df_pair.aid_x = df_pair.aid_x.astype('int')
    df_pair.aid_y = df_pair.aid_y.astype('int')
    df_pair=df_pair.drop_duplicates(['aid_x','aid_y'])         # 去掉self-loop

    # 可能存在的反向边
    df_left_bigger=df_pair[df_pair['aid_x']>df_pair['aid_y']]  # 左边大的边。reverse以后，加进去，再去重。都是左边小的边了。无反向边
    if dataset==0:
        df_left_bigger.columns=['aid_y','aid_x','w']           # 我们自己的数据集。有w
    else:
        df_left_bigger.columns = ['aid_y', 'aid_x']
    # 删掉已有的
    df_pair=df_pair[df_pair['aid_x']<df_pair['aid_y']]
    # 最终单边,含df_pair中所有共现。且x<y
    if dataset==1:
        single_pair = pd.concat([df_pair, df_left_bigger]).reset_index(drop=True).drop_duplicates(['aid_x', 'aid_y'])
    else:
        single_pair = pd.concat([df_pair, df_left_bigger]).groupby(['aid_x', 'aid_y']).w.sum().reset_index()  # 有权重的，累加权重

    # 建图：
    graph_items=list(set(single_pair.aid_x) | set(single_pair.aid_y))       # 图中全部item
    #item2idx=  dict([ (item, idx) for idx,item in enumerate(graph_items)])  # item2idx  存下来 不映射了
    #u,v=       single_pair.aid_x.map(item2idx).values,single_pair.aid_y.map(item2idx).values
    u, v = single_pair.aid_x.values, single_pair.aid_y.values

    g=dgl.graph(data=(u,v))

    #加反向边（按顺序）
    bg=dgl.add_reverse_edges(g)

    # 正反向边id映射
    E=g.number_of_edges()
    revers_map=torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

    # 加初始特征(可以考虑换成node2vec)
    w2v = Word2Vec.load(os.path.join(out_directory,'w2v_64/w2vModel.model'))
    def getvec(key):
        return w2v.wv[str(key)]
    pool = ThreadPool(6)
    #item_w2v = pool.map(getvec,item2idx.keys())  #根据每个item。得到对应vector.
    item_w2v = pool.map(getvec, g.nodes().numpy())  # 根据每个item。得到对应vector.
    g.nodes().numpy()
    pool.close()
    pool.join()

    #node_features_scaled = StandardScaler().fit_transform(np.concatenate(item_w2v, 0).reshape(-1, 32))
    node_features_scaled = np.concatenate(item_w2v, 0).reshape(-1, 64)
    bg.ndata['w2v'] = torch.Tensor(node_features_scaled)   # 设上节点的w2v权重，作为初始embedding

    # 保证labels中的item，图中都有.不用专门删除了，就用完整的测吧
    # if labels is not None:
    #     labels['ground_truth'] = labels['ground_truth'].apply(
    #         lambda x: [i for i in x if i in graph_items])  # lables去掉entity中没有的
    #     lens=[len(i) for i in labels['ground_truth'] if len(i)!=0]
    #     print("labels中可测试用户比例：",len(lens)/len(labels))  # 可测用户比例。 debug=False时，labels中可测试用户比例： 0.9661394355834432

    return bg,None,revers_map




