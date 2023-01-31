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
import dgl.function as fn
class Project(nn.Module):
    def __init__(self, g, hid_size=32,w2v=True):
        '''
        把id映射成对应embedding。
        '''
        super().__init__()
        self.w2v=w2v
        # 负责将节点上的各种特征都映射成向量，并聚合在一起，形成这个节点的初始特征向量
        #self.idembed = nn.Embedding(g.num_nodes(), hid_size)   # 只需要复制id。每个block,按特征id映射
        #nn.init.xavier_uniform_(self.idembed.weight)

        if w2v:
            self.w2v_embed = nn.Embedding(g.num_nodes(), hid_size)
            self.w2v_embed.from_pretrained(g.ndata['w2v'])

    def forward(self,ndata_id):
        '''
        ndata: block中，对应节点的原始特征 ,是一个dict: 如block[0].srcdata
        ----------
        直接concat
        '''

        return self.w2v_embed(ndata_id)   # 直接nid映射得到的w2v向量。index同node_id

        # list=[self.idembed(ndata_id)]               # 该block的所有src节点，对应的id   ndata['id']在cuda上(block).  model在cuda上
        # if self.w2v:
        #     list.append(self.w2v_embed(ndata_id))   # 按照id映射。node_i,对应w2v[i]
        # return torch.concat(list,-1)


class ItemToItemScorer(nn.Module):
    def __init__(self):
        '计算pair的点积'
        super().__init__()

    def forward(self, pair_graph, h):
        """
        pair_graph : pos_graph/neg_graph. 只包含B个head->tail边的子图。但compact后，每个子图都含另一个子图的节点。共3B个节点
        h :               heads, tails, neg_tails，3B个节点，每个节点经过网络后的最终表示。（3B,h）
        -----
        Returns:
            pair_score ：该子图中的B个pair(head->tail),根据最终表示h间的向量内积,得到B对节点的节点间相似度（B,1）
        """
        with pair_graph.local_scope():
            pair_graph.ndata["h"] = h                          # 把h设成子图节点的特征。子图中节点顺序，同block中seeds的顺序
            pair_graph.apply_edges(fn.u_dot_v("h", "h", "score"))  # 用消息传递api,计算边head和tail的特征's'（是head、tail的向量内积）
            pair_score = pair_graph.edata[ "score"]
        return pair_score   # 该子图中，B个pair根据向量内积得到的相似度

class NegativeSampler(object):
    def __init__(self, g, k,device):
        # caches the probability distribution
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.device=device

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        src = src.repeat_interleave(self.k)
        dst = self.weights.multinomial(len(src), replacement=True).int().to(self.device)
        return src, dst