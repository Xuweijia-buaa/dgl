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
from undirect_g import build_undirect_g

data_path='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/'
out_directory='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data/'
valid=True
def orig_data():
    if valid:
        train_session = pd.read_parquet(os.path.join(data_path,'otto-train-and-test-data-for-local-validation/train.parquet'))
        test_session = pd.read_parquet(os.path.join(data_path,'otto-train-and-test-data-for-local-validation/test.parquet'))  # 截断的test session.基于此产生submission.含所有测试用户
        test_labels = pd.read_parquet(os.path.join(data_path,'otto-train-and-test-data-for-local-validation/test_labels.parquet')) # 要预测的用户+行为 -> item
        test_path=os.path.join(data_path,'otto-train-and-test-data-for-local-validation/test.parquet')  # 截断的test session.基于此产生submission.含所有测试用户
        label_path = os.path.join(data_path,'otto-train-and-test-data-for-local-validation/test_labels.parquet') # 要预测的用户+行为 -> item
    else:
        train_session = pd.read_parquet(os.path.join(data_path,'otto-full-optimized-memory-footprint/train.parquet'))
        test_session = pd.read_parquet(os.path.join(data_path,'otto-full-optimized-memory-footprint/test.parquet'))
        test_labels=None
        test_path = os.path.join(data_path, 'otto-full-optimized-memory-footprint/test.parquet')
        label_path = None

    debug=False                        #  debug模式下，train和测试都只采样部分用户
    debug_fraction=0.001
    if debug:
        lucky_users_train = train_session.drop_duplicates(['session']).sample(frac=debug_fraction, random_state=42)['session']
        train = train_session[train_session.session.isin(lucky_users_train)]

        lucky_users_test = test_session.drop_duplicates(['session']).sample(frac=debug_fraction, random_state=42)['session']
        test = test_session[test_session.session.isin(lucky_users_test)]

        if valid:
            labels= test_labels[test_labels.session.isin(lucky_users_test)]
        # del test_labels,lucky_users_train,lucky_users_test
    else:
        train=train_session
        test=test_session
        if valid:
            labels=test_labels

    sample_fraction = 0.01  # 完整训练时，train可以只选择部分用户。test用全部
    lucky_users_train = train.sample(frac=sample_fraction, random_state=42)['session']
    train = train[train.session.isin(lucky_users_train)]  # 非debug时，测试集用全部用户。sample_fraction自由调节，减少使用的数目

    # test中每个用户的最后一次(非购买)点击。用来i2i
    test = test.sort_values(by=['session', 'ts'], ascending=[True, False])
    last_item = test.groupby('session').head(1)[['session', 'aid', 'type']]

    test['train'] = 0
    train['train'] = 1

    # del train_session, test_session
    # gc.collect()

    # 得到了最终的train/test/labels
    return train,test,labels,last_item,test_path,label_path

train,test,labels,_,test_path,label_path=orig_data()


# grapg-sage:  只构建i2i的边。
# train,test session合并，用来构建i2i的边。
# 图中只有item. 不含用户。预测时，用test用户的最后一次点击，i2i召回
# 准备好了原始数据，开始构建pair和对应的图
# 内存约占30%。10个G。1min 37s
def build_i2i_pair():
    train.index = pd.MultiIndex.from_frame(train[['session']])  # 用户作为索引
    test.index = pd.MultiIndex.from_frame(test[['session']])
    df=pd.concat([train,test])
    users=df.session.unique()    # 涉及到的用户

    # 每次处理n个用户。 300w个用户。最后concat
    chunk_size = 100000
    n_users=users.shape[0] # train_test中总的用户数目
    recent_number = 30
    hour_threshold = 24

    type_weight = {0: 0.1, 1: 0.3, 2: 0.6}  # 行为权重
    df_adj = pd.DataFrame(columns=["aid_x", "aid_y", 'w'])  # 建图所需的最终的pair和对应权重
    for i in tqdm(range(0, n_users, chunk_size)):
        df_chunk = df.loc[users[i]:users[min(n_users - 1, i + chunk_size - 1)]].reset_index(drop=True)  # 这部分用户对应df
        # 每个用户只截取最近的30个item
        df_chunk = df_chunk.sort_values(['session', 'ts'], ascending=[True, False])  # 按用户,t排好。越近越靠前
        df_chunk['n'] = df_chunk.groupby('session').cumcount()  # 每个用户到此刻的行为数目。


        #df_chunk = df_chunk.loc[df_chunk.n < recent_number].drop('n', axis=1)  # 取每个用户序列长为30以内（较近的）。留下所有test_sesison
        df_chunk = df_chunk.loc[(df_chunk.n < recent_number) | (df_chunk.train == 0)].drop('n',
                                                                                           axis=1)  # 取每个用户序列长为30以内（较近的）

        # CREATE part PAIRS
        df_pair = df_chunk.merge(df_chunk, on='session')  # 同一用户。各行为22pair.只取行为间隔一天内的pair
        df_pair['hour_elapsed'] = (df_pair.ts_y - df_pair.ts_x) / 3600  # 只取行为间隔一天内的pair，且x早于y的的pair

        #df_pair = df_pair.loc[(df_pair['hour_elapsed'] > 0) & (df_pair['hour_elapsed'] < hour_threshold) & (df_pair.aid_x != df_pair.aid_y)]
        df_pair = df_pair.loc[
            ((df_pair.train_x == 0) & (df_pair.train_y == 0) & (df_pair.aid_x != df_pair.aid_y) ) |
            ((df_pair['hour_elapsed'] > 0) & (df_pair['hour_elapsed'] < hour_threshold) & (
                        df_pair.aid_x != df_pair.aid_y))]

        # 去重并赋予权重
        df_pair['act_w'] = df_pair.type_y.map(type_weight)  # 行为权重 按原来评估时的权重
        df_pair['time_w'] = df_pair.hour_elapsed.apply(lambda x: 1 / (1 + abs(x)))  # 时间差权重
        df_pair['w'] = df_pair['time_w'] * df_pair['act_w']  # 上边2种权重分别加起来。作为边特征。
        # 合并所有pair
        df_pair = df_pair[['aid_x', 'aid_y', 'w']]
        df_pair.w = df_pair.w.astype('float32')
        df_pair = df_pair.groupby(['aid_x', 'aid_y']).w.sum()  # 聚合所有用户的i2i，得到i2i的权重
        df_pair = df_pair.reset_index()
        # 汇总
        df_adj = pd.concat([df_adj, df_pair[["aid_x", "aid_y", 'w']]], axis=0, ignore_index=True)
        df_adj.w = df_adj.w.astype('float32')
    return df_adj

df_pair=build_i2i_pair()  # 66387144 rows × 3 columns   # 6千万

del train
gc.collect()

def build_orig_graph():
    # 建图

    graph_items=set(df_pair.aid_x) | set(df_pair.aid_y)   # 图中全部item
    if labels is not None:
        labels['ground_truth'] = labels['ground_truth'].apply(
            lambda x: [i for i in x if i in graph_items])  # lables去掉entity中没有的
        lens=[len(i) for i in labels['ground_truth'] if len(i)!=0]
        print("labels中可测试用户比例：",len(lens)/len(labels))  # 可测用户比例。 debug=False时，labels中可测试用户比例： 0.9661394355834432

    item2idx=  dict([ (item, idx) for idx,item in enumerate(graph_items)])  # item2idx  存下来
    u,v,edge_w=df_pair.aid_x.map(item2idx),df_pair.aid_y.map(item2idx),torch.Tensor(df_pair.w.values)
    g=dgl.graph(data=(u,v))

    g.edata['w']=edge_w                 # 设上i2i权重
    # 可以考虑换成node2vec
    w2v = Word2Vec.load(os.path.join(out_directory,'w2v_64/w2vModel.model'))
    def getvec(key):
        return w2v.wv[str(key)]
    pool = ThreadPool(6)
    item_w2v = pool.map(getvec, item2idx.keys())  #根据每个item。得到对应vector.
    pool.close()
    pool.join()

    g.ndata['w2v']=torch.Tensor(np.concatenate(item_w2v,0).reshape(-1,64)) # 设上节点的w2v权重，作为初始embedding

    return g,item2idx


#g,item2nodeId=build_orig_graph()  # 有向图
g,item2nodeId,revers_map=build_undirect_g(df_pair)  # 无向图

del df_pair
gc.collect()

## Save
#out_directory='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data_debug/'
out_directory='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data_0/'
#out_directory=''
dgl.save_graphs(os.path.join(out_directory, "g.bin"), g)  # 把图单独存一下.   # 有向图。每个edge,对应1条边.

dataset = {  # 原始数据metadata,单独存一下。都是只用于eval
    "item2nodeId": item2nodeId,         # dict: g中item到nodeid的映射. 按nodeid的顺序 {aid:nodeid}
    'test_path': test_path,             # test session,用来预测已有的
    "labels_path": label_path,          # df:每个用户的ground_truth。 用来评估。 所有item都在g中。用户同test
    'revers_map':revers_map             # 无向图，id到reverse_e_id的映射
}

with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
    pickle.dump(dataset, f)


