import dask.dataframe as dd
import numpy as np
import scipy.sparse as ssp
import torch
import tqdm

import dgl


# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.
def train_test_split_by_time(df, timestamp, user):         # df:原始click_df,每个u-i inter一条记录
    df["train_mask"] = np.ones((len(df),), dtype=np.bool)  # 增加3列，用来做trian/valid/test的标记
    df["val_mask"] = np.zeros((len(df),), dtype=np.bool)
    df["test_mask"] = np.zeros((len(df),), dtype=np.bool)
    df = dd.from_pandas(df, npartitions=10)

    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    df = (
        df.groupby(user, group_keys=False)
        .apply(train_test_split)
        .compute(scheduler="processes")
        .sort_index()
    )
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return (
        df["train_mask"].to_numpy().nonzero()[0],      # 返回划分后，train中inter在click_df中的位置
        df["val_mask"].to_numpy().nonzero()[0],
        df["test_mask"].to_numpy().nonzero()[0],
    )


def build_train_graph(g, train_indices, etype, etype_rev):
    train_g = g.edge_subgraph(
        # edges:每种类型的边，指定要留的edge_id. 我们是无向边，正反都留
        # relabel_nodes：去除子图中的孤立节点，重新编号。这里不重新给节点编号
        edges={etype: train_indices, etype_rev: train_indices}, relabel_nodes=False
    )

    # copy features到子图中
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[
                train_g.edges[etype].data[dgl.EID]
            ]

    return train_g

'''
根据val_inter,对应的边节点id,在原始子图中找这些要预测的边，对应UV，即ui。构建稀疏矩阵，每个u,对应一个要预测的i. (非稀疏行)  # test同
'''
def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.num_nodes(utype)    # 总的用户节点
    n_items = g.num_nodes(itype)    # 总的item节点
    val_src, val_dst = g.find_edges(val_indices, etype=etype)         # val_indices，作为边节点，去原始大图中找（u-i类型的边）对应的UV
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()   # val inter对应的U   是val inter边对应的用户
    val_dst = val_dst.numpy()   # val inter对应的V   是val inter边对应的item
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix(
        (np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items)  # （data,(row,col),shape
    )
    test_matrix = ssp.coo_matrix(
        (np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items)
    )

    return val_matrix, test_matrix


def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / (
        values.max(0, keepdims=True) - values.min(0, keepdims=True)
    )
