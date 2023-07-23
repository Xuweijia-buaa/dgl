import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, negative_sampler
import tqdm
import argparse
import os
import pickle
from layer import ItemToItemScorer,Project


class SAGE(nn.Module):
    def __init__(self, g,in_size,hid_size,w2v,mlp=True):
        '''
        核心模型：3层layer的GraphSage。 含一个mlp，对节点对应向量进行交互，得到节点相似度score
        Source: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/sageconv.html
        ------
        in_size: 每个节点的初始embedding长度
        hid_size:每个节点最终的输出维度
        '''
        super().__init__()

        self.project=Project(g,hid_size=in_size,w2v=w2v)             # 用来把item的各种特征，映射成一个输入向量。
        # if w2v:
        #     in_size=2*in_size

        # 3层 GraphSAGE-mean
        # 每层聚合邻居节点: h_l -> h_l+1. 对应公式：https://docs.dgl.ai/en/0.4.x/api/python/nn.pytorch.html#sageconv。
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, 64, 'mean'))  # 每层graphSage。 指定输入，输出特征的维度。以及聚合方式
        self.layers.append(dglnn.SAGEConv(64, hid_size, 'mean')) # 可选聚合方式：mean, gcn, pool, lstm
        self.hid_size = hid_size

        #  一个mlp。 用来对每个节点对(head,tail)中uv节点对应的向量进行交互。得到每个节点对最终的相似度score:  交互向量(B,h) -> (B,1)
        #  预测时，如果已知节点最终向量。可以直接输入节点pair对应向量的element-wise乘 （B,h）,输出对应score（B,1）
        if mlp:
            self.predictor = nn.Sequential(        # 输入pair对应的点乘 （B,h）  输出pair的score （B,1）
                nn.Linear(hid_size, hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, 1))
        else:
            self.predictor=ItemToItemScorer()    # 输入block seeds对应的h. 以及对应子图（其中节点和seeds顺序一一对应）
        self.mlp=mlp

    def forward(self, pair_graph, neg_pair_graph, blocks):
        '''blocks:mixed模式下，放在主机pin_mem上，和model交互. device可访问'''
        # 原始特征映射
        feature_dict=blocks[0].srcdata   # 最后一阶邻居的特征。作为GNN的初始输入.
        x=self.project(feature_dict['id'])

       # graphSage本身：
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):  # 3层SAGEConv
            # block:每阶子图
            h = layer(block, h)           # 消息传递。 h_l -> h_l+1 聚合邻居信息。（如没有邻居，聚合结果只含该dst节点本身的信息）
            # 每层之间加Relu+Dropout
            if l != len(self.layers) - 1:
                h = F.relu(h)            # 得到每个seeds节点的最终表示。
                                         # 对应本批次B个head,tail,neg_tail节点。按顺序对应pos/neg graph中的节点id

        # 计算pair本身的score:
        if self.mlp:                                 # head和tail对应的向量，分别element-wise乘,再mlp。计算相似度
            pos_src, pos_dst = pair_graph.edges()                # B个正节点对的u，v ： (head->tail)。节点id可用来索引seeds的最终表示
            neg_src, neg_dst = neg_pair_graph.edges()            # B个负节点对的u，v ： (head->neg_tail). 节点id也可用来索引seeds的最终表示
            pos_score = self.predictor(h[pos_src] * h[pos_dst])  # 按heads,tail等节点的节点id，索引h,得到这些节点的最终表示
                                                                 # (B,h) * (B,h) -> (B,h) ->mlp -> (B,1)
            neg_score = self.predictor(h[neg_src] * h[neg_dst])  # 负pair的score同:     （B,1）
        else:                                       # 直接按pair的向量dot
            pos_score=self.predictor(pair_graph,h)
            neg_score = self.predictor(neg_pair_graph, h)

        # 计算loss
        loss='max_margin'     # 'point-wise' 'max_margin'
        #loss = 'point-wise'

        if loss=='max_margin':
            loss = (neg_score - pos_score + 1).clamp(min=0).mean()  # max(0, n-p+1) (B,1)
        else:                 # 'point-wise loss'
            score = torch.cat([pos_score, neg_score])  # （2B,1）  2B个节点对，每对得出的相似度
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])  # 每个节点对看做一个样本， (h,t) ->预测为1. （h->-t)，label是0
            loss = F.binary_cross_entropy_with_logits(score, labels) # 按二分类计算loss. point-wise,不是希望拉大差距，而是给负样本学一个分数0
        return loss,h

    def infer(self, g, device, batch_size):
        """
        Offline inference: 线下推断g中每个节点的最终表示。每个节点聚合了全部n阶邻居。（N,h）
                           device:    mxied时是cuda。model放gpu,上一阶表示作为输入放gpu。迭代的每B个节点和对应子图在gpu上做计算，结果缓存在cpu上
                           unseen节点：加入g后，也可以根据邻居信息聚合，infer对应表示。
        -------
        同node_classification.py:
            训练时，为了能把计算放GPU，只采样了部分邻居。但线下infer时，每个节点最好聚合全部邻居，得到信息最丰富的表示。
            为了聚合自己的全部邻居，每层的计算需要得到整个图的上一层表示，因此按层进行计算。Layer-wise inference
            参考：https://docs.dgl.ai/en/latest/guide/minibatch-inference.html
        """
        # 初始特征 (N，d)
        feat = self.project(g.ndata['id'].to(device))                     # g中所有节点的id,先放cuda上，并映射成（N,h）个embed

        # 采样器
        sampler = MultiLayerFullNeighborSampler(1,                          # 邻居采样器。每个节点聚合自己的全部邻居。只采一层，单纯聚合邻居。对应一个block
                                                prefetch_node_feats=['id']) # 采样后的block,带id,来索引embedding

        dataloader = DataLoader(
            g,  # cpu上
            torch.arange(g.num_nodes()).int().to(g.device),                    # 迭代g中所有nodes。 所有nodes_id放gpu中，计算发生在gpu上
            sampler,
            device=device, # cuda.  cpu上采样到的子图，放哪里
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)  # True

        # 分层，计算所有节点的最终表示
        for l, layer in enumerate(self.layers):          # 第i层, 得到所有节点的该阶表示. 下一阶节点的计算，依赖所有节点的上一层表示
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,   # 用来缓存该阶，g上所有节点的表示： （N,h）
                            pin_memory=pin_memory)                                # 整图表示，放在cpu锁页内存上。传输快些
            feat = feat.to(device)                                                # 每层的全部节点上一阶得到的表示，用作输入

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                # input_nodes: 本批节点的所有邻居(原始nodeid)。
                # output_nodes:本批节点
                x = feat[input_nodes.long()]       # 邻居的初始表示。是上一阶的表示（基于上一阶计算结果得到/初始是feat_embedding）
                h = layer(blocks[0], x)     # 根据节点的邻居(block)和对应的上一阶表示(x)，聚合得到该节点高一阶表示 （B,h）
                                            #     blocks只有一阶邻居。直接把邻居子图和特征，输入中间某层: h_l -> h_l+1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes.long()] = h.to(buffer_device)  # 把本层每个batch得到的结果（gpu上计算的），放cpu
            feat = y                         # 作为下一阶计算时，所有节点的初始表示（N,h）
        return y                             # 返回所有节点的最终表示