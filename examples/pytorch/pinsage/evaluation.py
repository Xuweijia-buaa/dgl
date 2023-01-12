import argparse
import pickle

import numpy as np
import torch

import dgl


def prec(recommendations, ground_truth):
    '''
    recommendations: 每个用户预测出的topK item(在h_items里的位置)   (|U|,topK)
    ground_truth：测试集对应的稀疏矩阵，shape同原图。但边只包含测试集中u-i interation。是用户train序列之后的next item
    -------
    单个样本(用户)，预测的所有结果，命中了next_item,该样本是1.否则是0. 返回所有样本预测正确的比例
    '''
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)  # 每个用户id，重复topK次。 users: 0 0 0 | 1 1 1 ...
    item_idx = recommendations.flatten()         # 和对应的预测item:   pre_items: 2 1 1 | 3 1 1 ...
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K)) # 原始矩阵中，预测结果所在位置。对每个用户，如果该位置是1，则命中了next_item
    hit = relevance.any(axis=1).mean()   # 每个用户预测的topK个item,至少有一个命中next_item，该样本就是1。否则是0
    return hit                           # 最终的hitRate是所有样本，预测正确的比例。


class LatestNNRecommender(object):
    def __init__(
        self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    ):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        h_item：所有item对应的最终embedding   （|I|,h）
        return：每个用户用最后一次点击，i2i预测出的topK个item_id （|U|,topK）
                预测时，根据item和全体item之间的向量内积，计算相似度。
                每个item_id，对应全体item中的index
        """
        # 1:从train_g中抽取每个用户的最后一次点击，构成新的子图
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])  # 只抽取(u->i)的子图
        n_users = full_graph.num_nodes(self.user_ntype)
        # 采样得到的新子图：每个用户，只保留最后一次点击的item，对应的边
        latest_interactions = dgl.sampling.select_topk(
            graph_slice, k=1, weight=self.timestamp, edge_dir="out"    # 采样得到的新子图：每个节点，只保留边特征weight最大的topK个出边
        )
        user, latest_items = latest_interactions.all_edges(            # u_list，对应的每个用户最后点击的i_list。 按uid排好
            form="uv", order="srcdst"
        )
        assert torch.equal(user, torch.arange(n_users))                 # each user should have at least one "latest" interaction

        # 2：每个用户用train中最后一次点击的item， i2i, 找对应的topK item  (根据item,和全部item向量h_item的内积)
        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(            # 每个用户在train_g中的最后一次点击的item （B,1）。
                device=h_item.device
            )
            dist = h_item[latest_item_batch] @ h_item.t()               # train中item向量和所有item的向量内积。i2i,找topK
            for index, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(               # 排除每个用户已经点过的item （TODO：重复购买场景下，不需要去除）
                    u, etype=self.user_to_item_etype
                )
                dist[index, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])              # 分数最高的topK item，在h_item(全体item)中的index

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations  # 所有用户用最后一次购买，i2i出的topK item


def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    )
    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()  # 每个用户预测出的topK item (根据train中的最后一次点击,i2i,计算内积)
    return prec(recommendations, val_matrix) #返回所有样本预测正确的比例


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("item_embedding_path", type=str)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, "rb") as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
