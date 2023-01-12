import argparse
import os
import pickle

import evaluation
import layers
import numpy as np
import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import dgl


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        '''
        Parameters
        ----------
        full_graph: train对应训练子图
        ntype：      item节点类型名。
        textsets：   item类型对应的序列特征
        '''
        super().__init__()

        # 负责将节点上的各种特征都映射成向量，并聚合在一起，形成这个节点的初始特征向量
        # 每个离散特征初始化一个embedding。 每个连续特征初始化一个Linear映射层
        # 每个节点的各种特征，映射得到的多个向量，直接相加。得到一个初始向量。(B,|F|,h) ->  (B,h)
        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims  # 只考虑item类型节点的特征 (各层邻居生成的blocks,只包含item节点)
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)         # 核心模型。最后一阶邻居，经过多层conv，得到的表示。(还需加本身embed)(邻居信息存在blocks中)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)  # 根据最终向量表示，计算head->tail 节点相似度。

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)                 # 是blocks中最原始的3种输入节点，经过net后最终的表示 （B,h）
                                                       # 最后一层邻居特征 -> 映射得到初始embed -> 多层conv -> 目标节点的最终表示
        pos_score = self.scorer(pos_graph, h_item)     # B个正样本对的相似度 （根据向量内积得到,还加了一个额外bias，考虑去掉）
        neg_score = self.scorer(neg_graph, h_item)     # B个负样本对的相似度  (和上边B对对应)
        return (neg_score - pos_score + 1).clamp(min=0) # loss:max(0, n-p+1).   margin hinge loss (B,1)

    def get_repr(self, blocks):
        '''
        1 本批次B个目标节点，以及对应的最后一层邻居的节点。
          每个节点的多个原始特征，根据self.proj映射成的多个特征向量(长均为h)，直接相加，得到每个节点的初始表示：(F,h)->（1,h）
        2 经过多层graphconv,得到目标节点的最终表示
        -----
        blocks:存图中的B个item（含item原始特征）,多阶邻居采样后，每阶的子图。 每个子图的src含dst，是前|dst|个
        '''
        h_item = self.proj(blocks[0].srcdata)           # 目标节点的最后一层邻居，映射得到的初始embedding （B*n1*n2,h）。每个节点一个向量
        h_item_dst = self.proj(blocks[-1].dstdata)      # 目标节点映射得到的初始embedding. (B,h)。每个节点一个向量
        return h_item_dst + self.sage(blocks, h_item)   # 最后的表示(B,h)：是目标节点本身的表示(B,h) + 多阶邻居聚合后的表示(B,h)


def train(dataset, args):
    g = dataset["train-graph"]                 # 准备好的训练子图
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable embedding for each entity)
    # user_id.item_id,本身作为特征：
    g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    for i in range(g.num_nodes(item_ntype)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)
    for key, field in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"]
        )
        textset[key] = (
            textlist,
            vocab2,
            vocab2.get_stoi()["<pad>"],
            batch_first,
        )

    # Sampler

    # 每个batch,采样B个item节点。和对应的B个正样本+B个负样本（都是item类型节点），用来无监督训练
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size
    )

    # 封装每层对应的PinSAGESampler。每层节点用PinSAGESampler采样重要邻居。再用邻居采样下一阶邻居
    # 每层节点和下一阶节点，生成只包含这2层节点的block。2层layer，共生成2个block,分别用来聚合。
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )

    # 用来把采样好的节点，在送入前，做处理。
    # train时. 每次采样得到B个head,tail,neg_tail3类节点。都采样n_layer层邻居，作为n个子图
    # test，每次传入B个item节点（train中已有的），同样采样n层邻居，作为n个子图。
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset
    )
    dataloader = DataLoader(
        batch_sampler,                         # 迭代器。训练时，每次从图中，随机采样B个item,和该item的B个正负样本
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)), # 测试，用train_g中所有item。分batch送入  （TODO:可以用这里的，代替原来的测试）
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # blocks 和 子图， copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()  # margin hinge loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        # TODO：可以用我们自己的测试集。（user完全不同，item完全相同. 给定test中的user，根据test截断session中的最后一次item,i2i.用labels eval）
        #       train和test中的截断session,拼成大图，训练。（user不区分是否是test中的。都能用。item节点相同）
        #       预测：根据test中的user，在截断session中的最后一次item,i2i，取top20
        #       1：大图中要不要ground truth对应的inter？ground truth中的item,在大图中都有,不担心预测不出来。
        #          但如果要用小数据做demo:
        #                    train中去掉低频用户无影响。
        #                    train中去掉低频item(主要去click),可能test就没法预测出来了。但可能next click/next chart中也不多。主要看总的命中数目，没关系
        #                    test的截断session中，也可以去低频用户。不影响最后测试i2i。
        #                                        也可以去低频item。（labels中不去）
        #                    最终只留部分ui在大图中。但测试用全体用户的最后一次点击（事先算好），i2i. 得topK.
        # TODO:看下去掉大部分节点后，图大小，边大小。跑通demo
        #
        model.eval()
        with torch.no_grad():
            # train中所有item_id。用来测试
            h_item_batches = []
            for blocks in dataloader_test:           # 每B个item_id，采样n阶邻居。每层得到一个block,含本层和下一层采样出的邻居节点
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks))  # model.get_repr(blocks)： B个item的最终表示。
            h_item = torch.cat(h_item_batches, 0)              # 测试item的最终表示.（n,h）

            # TODO:这些item对应的最终表示，我们可以先保存下来
            #      或者直接在这里，写我们自己的eval
            print(
                # 根据每个item的topK预测结果，结合抽取出来的test inter边，evaluate
                evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size) # 返回所有测试样本(用户)预测正确的比例
            )

# python model.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 64
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--random-walk-length", type=int, default=2)  # 采样邻居时，单次游走只走2步
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)   # 每个节点采样10次
    parser.add_argument("--num-neighbors", type=int, default=3)  # 默认每个节点采样3个邻居
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cpu"
    )  # can also be "cuda:0"
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, "data.pkl")    # 其他metadata.包含节点，边类型名，原始df等。
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")   # 原始train子图
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset["train-graph"] = g_list[0]
    train(dataset, args)
