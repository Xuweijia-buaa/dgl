from collections import defaultdict
import collections
import numpy as np
import pandas as pd
import os
import gc
import polars as pl

def evaluate(path, index, mode="validation", n_neighbors=20,labels_path=""):
    test = pl.read_parquet(path)

    session_types = ['clicks', 'carts', 'orders']
    test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
    test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

    del test
    gc.collect()
    labels = []

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    for AIDs, types in zip(test_session_AIDs, test_session_types):
        if len(AIDs) >= 20:
            # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
            weights = np.logspace(0.1, 1, len(AIDs), base=2, endpoint=True) - 1
            aids_temp = defaultdict(lambda: 0)
            for aid, w, t in zip(AIDs, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]

            sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
            labels.append(sorted_aids[:20])
        else:
            # here we don't have 20 aids to output -- we will use word2vec embeddings to generate candidates!
            AIDs = list(dict.fromkeys(AIDs[::-1]))

            # let's grab the most recent aid
            most_recent_aid = AIDs[0]

            # and look for some neighbors!
            nns = [i for i in index.get_nns_by_item(most_recent_aid, n_neighbors)[1:]]

            labels.append((AIDs + nns)[:n_neighbors])

    labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

    predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

    prediction_dfs = []

    for st in session_types:
        modified_predictions = predictions.copy()
        modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
        prediction_dfs.append(modified_predictions)

    sub = pd.concat(prediction_dfs).reset_index(drop=True)

    del prediction_dfs, predictions, labels_as_strings, labels, test_session_types, test_session_AIDs
    gc.collect()
    if mode == "test":
        sub.to_csv("submission.csv", index=False)
        return sub
    else:

        sub['labels_2'] = sub['labels'].apply(lambda x: [int(s) for s in x.split(' ')])
        submission = pd.DataFrame()
        submission['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
        submission['type'] = sub.session_type.apply(lambda x: x.split('_')[1])
        submission['labels'] = sub.labels_2.apply(
            lambda x: [item for item in x[:]])  # .apply(lambda x: [int(i) for i in x.split(',')[:20]])
        test_labels = pd.read_parquet(labels_path)
        test_labels = test_labels.merge(submission, how='left', on=['session', 'type'])
        del sub, submission
        gc.collect()
        gc.collect()
        test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))),
                                                axis=1)
        test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0, 20)
        recall_per_type = test_labels.groupby(['type'])['hits'].sum() / test_labels.groupby(['type'])['gt_count'].sum()
        score = (recall_per_type * pd.Series({'clicks': 0.1, 'carts': 0.30, 'orders': 0.60})).sum()
        print("Recall clicks: {:f} carts: {:f} orders {:f}".format(recall_per_type[1], recall_per_type[0],
                                                                   recall_per_type[2]))
        print("Score on validation data: {:f}".format(score))  # valid score没有问题
        return score

def create_sub(args,index,test_path,mode="validation",epoch=0):
    test_df = pl.read_parquet(test_path).to_pandas().reset_index(drop=True)

    top_clicks = test_df.loc[test_df['type'] == 0, 'aid'].value_counts().index.values[:20]
    top_charts = test_df.loc[test_df['type'] == 1, 'aid'].value_counts().index.values[:20]
    top_orders = test_df.loc[test_df['type'] == 2, 'aid'].value_counts().index.values[:20]

    type_weight_multipliers = {0: 0.5, 1: 9, 2: 0.5}

    # item2nodeId=args.item2nodeId
    # node2Item = dict([(v, k) for k, v in item2nodeId.items()])

    def suggest_item(df, act_type):
        # USE USER HISTORY AIDS AND TYPES
        aids = df.aid.tolist()
        types = df.type.tolist()
        unique_aids = list(dict.fromkeys(aids[::-1]))
        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            # 每个序列，长已经超过20，按时长/类型对应的权重重排， 没有用学到的i2i
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = defaultdict(lambda: 0)
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]
            sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
            return sorted_aids
        else:
            # 否则用test中的aids,w2v寻找。
            AIDs = list(dict.fromkeys(aids[::-1]))  # 历史recent

            recent_len = max(min(3, len(AIDs)), 1)  # 最多3个，最少一个recent item （即使是0，也是1）
            AIDs_num = round((20 - len(AIDs)) / recent_len) + 2  # 每个item平均找几个
            nns_it = []

            for it in range(0, recent_len):  # 每个item，来i2i。（AIDs[it]）
                nns_it += [i for i in index.get_nns_by_item(AIDs[it], AIDs_num)[1:]]
                #nns_it += [node2Item[i] for i in index.get_nns_by_item(item2nodeId[AIDs[it]], AIDs_num)[1:]]
                # try:
                #     nns_it += [node2Item[i] for i in index.get_nns_by_item(item2nodeId[AIDs[it]], AIDs_num)[1:]]
                # except:
                #     continue

            # if args.dataset==0:                  # 我们自己的数据集。可能有部分没有。（只选了train,test中部分pair.）
            #     for it in range(0, recent_len):  # 每个item，来i2i。（AIDs[it]）
            #         nns_it += [node2Item[i] for i in index.get_nns_by_item(item2nodeId[AIDs[it]], AIDs_num)[1:]]
            #         # try:
            #         #     nns_it += [node2Item[i] for i in index.get_nns_by_item(item2nodeId[AIDs[it]], AIDs_num)[1:]]
            #         # except:
            #         #     continue
            # else:
            #     for it in range(0, recent_len):  # 每个item，来i2i。（AIDs[it]）
            #         nns_it += [i for i in index.get_nns_by_item(AIDs[it], AIDs_num)[1:]]

            # select repeating and unique neighbors
            nns_repeated = [item for item, count in collections.Counter(nns_it).items() if count > 1]  # 优先放重复的
            nns_once = [item for item, count in collections.Counter(nns_it).items() if count == 1]
            nns = (nns_repeated + nns_once)[:20]

            result = (AIDs + nns)[:20]

        if act_type == 'click':
            # USE TOP20 TEST CLICKS
            return result + list(top_clicks)[:20 - len(result)]  # 补充热门
        elif act_type == 'chart':
            return result + list(top_charts)[:20 - len(result)]  # 补充热门
        else:
            return result + list(top_orders)[:20 - len(result)]

    pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
        lambda x: suggest_item(x, 'click')
    )
    pred_df_charts = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
        lambda x: suggest_item(x, 'chart')
    )
    pred_df_buys = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
        lambda x: suggest_item(x, 'buy')
    )

    clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df_charts.add_suffix("_carts"), columns=["labels"]).reset_index()  # 用的一个

    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))

    if mode == "test":
        save_path = os.path.join(args.data_path, "submission_epoch{}.csv".format(epoch))
        pred_df.to_csv(save_path, index=False)

    return pred_df

# COMPUTE METRIC
# def eval_metric(dataset,sub):
#     test_labels=dataset['labels']
#     submission = pd.DataFrame()
#     submission['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
#     submission['type'] = sub.session_type.apply(lambda x: x.split('_')[1])
#     #submission['labels'] = sub.labels.apply(lambda x : [item for item in x[:20] ]) # 如果val时创建的labels是list
#     submission['labels'] = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')[:20]])
#     test_labels = test_labels.merge(submission, how='left', on=['session', 'type'])
#     test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
#     test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
#     recall_per_type = test_labels.groupby(['type'])['hits'].sum() / test_labels.groupby(['type'])['gt_count'].sum()
#
#     score = (recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()
#     print("Recall clicks: {:f} carts: {:f} orders {:f}".format(recall_per_type[1],recall_per_type[0],recall_per_type[2]))
#     print("Score on validation data: {:f}".format(score)) # valid score没有问题

def eval_metric(sub,labels_path):
    #labels_path = '/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/otto-train-and-test-data-for-local-validation/test_labels.parquet'
    test_labels = pd.read_parquet(labels_path)
    submission = pd.DataFrame()
    submission['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
    submission['type'] = sub.session_type.apply(lambda x: x.split('_')[1])
    # submission['labels'] = sub.labels.apply(lambda x : [item for item in x[:20] ]) # 如果val时创建的labels是list
    submission['labels'] = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')[:20]])
    test_labels = test_labels.merge(submission, how='left', on=['session', 'type'])
    test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))),
                                            axis=1)
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0, 20)
    recall_per_type = test_labels.groupby(['type'])['hits'].sum() / test_labels.groupby(['type'])['gt_count'].sum()

    score = (recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()
    print("Recall clicks: {:f} carts: {:f} orders {:f}".format(recall_per_type[1], recall_per_type[0],
                                                               recall_per_type[2]))
    print("Score on validation data: {:f}".format(score))  # valid score没有问题

import torch
import dgl
def to_bidirected_with_reverse_mapping(g):
    """
    把g变成一个没有重复边的无向图（同质图）。每个uv含正反2条边。
    返回图中每条边，到其反向边的映射。
    ------
    return：
       g: 新生成的无向图,没有重复边。每条uv都对应正反2条边。
       reverse_mapping：g中每条边到反向边的映射:  reverse_mapping[eid]: reverse_eid
    """
    g_simple, mapping = dgl.to_simple(    # 去除原图中的重复边 。变成一个没有重复边的无向图 （重复边：同质图看uv,异质图看uev）
        dgl.add_reverse_edges(g),         # 给原图添加了反向边。可能会有重复边
        return_counts='count',            # 原图中每条边的数目，作为新图该边的特征
        writeback_mapping=True)           # 返回 原图的边id:新图的边id

    c = g_simple.edata['count']           # 每条边的重复数目（在原图中）。
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype) # 有正反边的简单图，每个边的反向边
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()                       # 正向边对应的uv
    src2, dst2 = g_simple.find_edges(reverse_mapping)   # 反向边对应的uv
    assert torch.equal(src1, dst2)                      # 验证
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping