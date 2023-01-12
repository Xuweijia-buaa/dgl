import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchtext.data.functional import numericalize_tokens_from_iterator

import dgl


def padding(array, yy, val):
    """
    :param array: torch tensor array
    :param yy: desired width
    :param val: padded value
    :return: padded array
    """
    w = array.shape[0]
    b = 0
    bb = yy - b - w

    return torch.nn.functional.pad(
        array, pad=(b, bb), mode="constant", value=val
    )

# 根据目标节点，对采样得到的子图进行裁剪
#v该二分图只包含目标节点，和目标节点需要的源节点. 以及其中的边。去掉了g中其他无关的节点和边
def compact_and_copy(frontier, seeds):
    '''
    根据目标节点，对采样得到的子图进行裁剪. 只包含seeds和邻居节点。去掉了g中其他无关的节点和边
    ----------
    frontier:原图。由sampler根据源节点seeds,每个节点在原图中采样n个邻居节点，得到的子图。仍包含原图中其他节点
    seeds:   源节点
    -------
    返回裁剪后的子图。是一个block,二分图。只包含seeds,和要聚合seeds的邻居节点，以及对应的边。B个dst节点，Bn个邻居，共Bn条边。：
        neibor1 -> src
        neibor2 -> src
    '''
    block = dgl.to_block(frontier, seeds)               # seeds:指定该图中的目标节点id。裁剪后，只剩这些节点，和对应的入边（+用来聚合的邻居节点）。

    for col, data in frontier.edata.items():            # 子图中，边原有的特征。
        if col == dgl.EID:                              #  col: 'weight'  data:是每个邻居节点到源节点的权重，代表该邻居的重要性。 B个节点，BN个邻居
            continue
        block.edata[col] = data[block.edata[dgl.EID]]   # block.edata[dgl.EID]: block中，对应的原始边的id。存在边特征dgl.EID中
                                                        # 把原始的每个邻居到seeds的权重特征，复制成block的边特征
    return block


class ItemToItemBatchSampler(IterableDataset):
    '''
    随机采样B个item节点，作为主节点
       采样B个B个正样本，是主节点的iui，是每个item，被同一用户购买的item
       采样B个B个正样本，是任意B个其他item.
    作为无监督训练的正负样本pair (h,t) (h,-t). max-margin-loss
    '''
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]   # etype原始名称   'click' : u->i
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint( 0, self.g.num_nodes(self.item_type), (self.batch_size,)) # 随机B个item节点 （对应节点id）

            tails = dgl.sampling.random_walk(  # 采样B个正样本。每个正样本是src按路径i->u->i)随机游走后. 被同一用户购买过的item
                self.g,
                heads,                               # 每个起始节点的node_id[array]。可含重复节点，相当于一个节点多轮random_walk,起点都是该节点
                metapath=[self.item_to_user_etype, self.user_to_item_etype],  # 指定startnode开始以后的edge types（list）
                                                                              # 从起点开始，找符合每个边类型的下一个节点（多个，随机找一个).直到找不到metapath[t],停止
            )[0][:, 2]                               # 每个起始节点，最多采样len(metapath)个node。提前停止的，节点id用-1填充。（B,L）
                                                     # 只取randomwalk中的最后一个节点 （item）,作为正向item (nodeid)


            neg_tails = torch.randint(         # 采样B个负样本。
                0, self.g.num_nodes(self.item_type), (self.batch_size,)  # 这里随机采样B个item节点,作为负样本
            )                      # TODO:可以用heads通过PinSAGESampler采样邻居。补充rank靠后的一些item做强负样本

            mask = tails != -1                 # B个正样本中，未有效采到的。扔掉
            yield heads[mask], tails[mask], neg_tails[mask]  # 返回本批次的B个item,和对应的B个正样本。B个负样本（item对应的node_id）


class NeighborSampler(object):
    def __init__(
        self,
        g,
        user_type,
        item_type,
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers,
    ):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            # 一个PinSAGESampler：
            # 每个给定节点，返回的邻居是num_random_walks次随机游走后，访问次数最多的num_neighbors个同类型节点。
            #            每个node可以随机游走num_random_walks次
            #            每次随机游走，可以包含多个metapath，以[i->u],[i->u]的循环。直到以random_walk_restart_prob概率终止。
            # 给定起始节点后，返回一个子图。
            #           包含每个起始节点，和各自采样出的最重要的num_neighbors个邻居。作为边.
            #           每条边从邻居指向源节点。
            #           每个节点和他们的邻居之间的边，含特征['weights']。是随机游走时该邻居被访问的次数（重要性）
            # neibor1 -> src
            # neibor2 -> src
            dgl.sampling.PinSAGESampler(      # 每个节点多次游走，落脚于某节点的次数越多，则这个邻居越重要
                g,
                item_type,                    # 主类型. 循环[type->other_type]。随机游走，只采样该类型作为邻居。返回的子图只含该类型
                user_type,                    # other_type
                random_walk_length,          # 每个node单次随机游走的最大长度（每个节点可能有多次）
                random_walk_restart_prob,    # 每个node单次随机游走的终止概率（每次metapath过后）
                num_random_walks,            # 每个node可以随机游走的次数
                num_neighbors,               # 每个node总共采样的邻居数目
            )
            for _ in range(num_layers)       # 每层一个独立的PinSAGESampler
        ]


    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        '''
        seeds: 是B个item对应的node_id。作为采样邻居的源节点
          从seeds开始，每次采样n个邻居，得到一个只含seeds和一阶邻居节点的二分图:block[0]；
          之后一阶邻居再采样，得到只含一二阶邻居的二分图:block[1]；
          ...
        返回：seeds对应的各阶子图倒序排列。blocks[0]是最后一阶邻居对应的子图。blocks[0]是只包含seeds和一阶邻居的子图
             可以从blocks[0]开始聚合。
        '''
        blocks = []
        for sampler in self.samplers:            # 每层的PinSAGESampler
            # 根据seeds,从原图中采样邻居. 返回这些节点和邻居对应的子图。仍含有原图中的其他节点
            frontier = sampler(seeds)  # 子图的每条边，从邻居节点，指向源节点。每条边的权重'weight'，是该邻居节点的重要性。
            # neibor1 -> src
            # neibor2 -> src

            if heads is not None:      # 看该子图是否包含待预测的边h->t/h->-t。如果包含，从train中去掉
                eids = frontier.edge_ids(torch.cat([heads, heads]),torch.cat([tails, neg_tails]),return_uv=True,)[2] # 根据uv,得边id
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)  # 去掉指定的边

            # 对采样得到的子图进行裁剪. 得到一个二分图，只含seeds和一阶邻居。
            block = compact_and_copy(frontier, seeds)  # 去掉了其他无关的节点和边。邻居的随机游走权重，也复制到block上。
            blocks.insert(0, block)                    # layer越靠后，对应的采样结果越靠前。
                                                       # 首个二分图只含源节点和一阶邻居。 第二个二分图只含一二阶邻居节点。最后一个二分图是采样结束的子图
                                                       # 倒序排列，从最后一个二分图开始聚合

            # 新的邻居节点，作为源节点，再去采样。得到只含一阶二阶邻居的子图。
            seeds = block.srcdata[dgl.NID]            # seeds的邻居节点（在原图中的node_id）
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        '''
        返回:
         positive graph：只抽取heads，tail节点构建子图，但包含了3部分节点
         negative graph：只抽取heads，neg_tails节点构建，也包含了3部分节点。子图中的3部分节点一起编号，编号同positive graph。
         blocks: 3种节点一起作为seeds,采样邻居,得到n层layer分别对应的block.
        '''
        # Create a graph with positive connections only and another graph with negative
        # connections only.

        # 由heads->tails构建positive graph
        # 由heads->neg_tails构建negative graph
        pos_graph = dgl.graph(
            (heads, tails), num_nodes=self.g.num_nodes(self.item_type)     # 重建只含B个heads和B个正样本的子图。把head作为U，tail作为V
        )                                                                  # num_nodes先设置成原图中所有item节点，之后compact去掉多余节点
        neg_graph = dgl.graph(
            (heads, neg_tails), num_nodes=self.g.num_nodes(self.item_type)  # 重建只含B个heads和B个负样本的子图。把head作为U，neg_tail作为V
        )

        # 去除heads, tails, neg_tails以外的节点。将大图压缩成小图，避免不必要的信息传递，提升计算效率
        # 输入的所有子图的节点一起编号。各子图边不变，但包含了其他所有子图的节点。
        #                          如pos_graph节点包含heads+tails+neg_tails，仍只有heads->tails的边
        # 它们来自于同一幅由heads, tails, neg_tails组成的大图
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])     # 去掉各图中都没有的孤立节点。

        seeds = pos_graph.ndata[dgl.NID] # pos_graph中的节点(在原图中的编号)，包含heads,tail,neg_tail 3部分节点

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)  # 训练时，3种节点都采样邻居，最终聚合得到各自表示
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    原图中的每种特征，按原图中的取值，和block中对应节点（在源节点中的id）,把原图中item的节点特征，赋给block这些节点
    ndata: block中的部分节点(src/dst)，对应的特征dict。 如block.srcdata
    """
    for col in g.nodes[ntype].data.keys():          # trian_g中，item类节点的每种特征
        if not assign_id and col == dgl.NID:        # 默认不含trian_g节点，对应的原图nodeid特征。(但包含"id"特征,是trian_g中节点重新编号后的id特征,只含train中节点)
            continue
        induced_nodes = ndata[dgl.NID]              # block的这部分节点，对应在train_g中的原始id
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]  # 把train_g中该特征对应的取值，按block中节点的原始编号，复制到block的这些节点上。
                                                              # g.nodes[ntype].data[col]：train_g中，所有item类节点，特征col的取值

def assign_textual_node_features(ndata, textset, ntype):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[dgl.NID].numpy()           # block中对应节点，在原图中id

    for field_name, field in textset.items():   # item的各类textual特征
        textlist, vocab, pad_var, batch_first = field

        examples = [textlist[i] for i in node_ids]
        ids_iter = numericalize_tokens_from_iterator(vocab, examples)

        maxsize = max([len(textlist[i]) for i in node_ids])
        ids = next(ids_iter)
        x = torch.asarray([num for num in ids])
        lengths = torch.tensor([len(x)])
        tokens = padding(x, maxsize, pad_var)

        for ids in ids_iter:
            x = torch.asarray([num for num in ids])
            l = torch.tensor([len(x)])
            y = padding(x, maxsize, pad_var)
            tokens = torch.vstack((tokens, y))
            lengths = torch.cat((lengths, l))

        if not batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + "__len"] = lengths


def assign_features_to_blocks(blocks, g, textset, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    '''
    只把train_g中的节点特征，赋给了seeds和seeds的最后一阶邻居
    因为在原图中采样出的frontier,不带节点原始特征。block子图同。因此需要把原图中节点特征，赋给采样后的最后一阶邻居（和自己）
    ----------
    Parameters
    ----------
    blocks: 从B个item开始采样，每层得到一个block,是只含第i和第i+1阶邻居节点的二分图。
             倒序放： block[0]放最后2层邻居节点。block[-1]放seeds和一阶邻居。
    g：      trian_g
    textset: item的text类型的序列特征，处理后的dict.含字典等
    ntype:  item节点类型。
    '''
    assign_simple_node_features(blocks[0].srcdata, g, ntype)        # 第一个block，源节点是最后一阶邻居。把原图中item节点的特征，赋给这些节点
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)       # 最后一个block,dst节点是最初的seeds。把原图中item节点的特征，赋给这些节点
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)

# collator,经过该函数后，才能传给forward
class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, textset):
        self.sampler = sampler         # 用来给B个节点，采样下一阶邻居的采样器 （通过多次随机游走，采样访问次数多的同类型节点。Personal PageRank）
        self.ntype = ntype
        self.g = g                     # 训练时用的图
        self.textset = textset

    # 训练时使用的collator
    def collate_train(self, batches):
        '''
        训练前，每个batch采样出的B个heads, tails, neg_tails。一起作为源节点，采样各自的邻居。
        每层邻居暂存成一个block子图，只含本阶节点和下一阶邻居
        ----------
        batches：每个batch采样出的B个heads, 以及对应的B个正负样本tails, neg_tails
        ----------
         positive graph：只抽取本次的heads，tail节点，构建head->tail的子图，但包含了3部分节点
         negative graph：只抽取本次的heads，neg_tails节点，构建head->neg_tails的子图，包含相同的节点
         blocks: 3种节点一起作为seeds,采样邻居,得到n层layer分别对应的n个block. 源节点和最后一阶邻居节点，从原图中复制了初始特征。
        '''
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        # 采样出的heads, tails, neg_tails，都作为源节点，采样各自的邻居
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            heads, tails, neg_tails
        )
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)

        return pos_graph, neg_graph, blocks

    # test时使用的collate_func
    def collate_test(self, samples):
        '''
        测试前，用samples提前采样好n阶邻居。每层节点和对应的邻居，对应一个block子图。倒序放，且设置上train_g中的原始特征。
        ----------
        samples:每个batch，B个测试item
        ----------
        blocks: 每层邻居暂存成一个block子图，只含本阶节点和下一阶邻居，每个邻居指向对应seeds：
                blocks[-1]: B个测试item作为seeds， 每个节点采样n个邻居，得到一个只含seeds和一阶邻居节点的二分图block。
                blocks[-2]: 用上次采样得到的一阶邻居，作为新的seeds,再采样，得到只含一二阶邻居的二分图block
                ...
                blocks[0]: 放最后一阶邻居对应的子图。
                blocks中，最初的源节点和最后一阶邻居节点，从原图中复制了初始特征，聚合用
        '''
        batch = torch.LongTensor(samples)               # B个测试item_id
        blocks = self.sampler.sample_blocks(batch)      # 用这B个item,采n阶邻居。对应n个block。block只含本阶节点和下一阶邻居。倒序，最后一阶邻居放blocks[0]。

        # 把train_g中的节点原始特征，赋给了seeds节点，和seeds的最后一阶邻居.聚合时会用到
        # 设置的特征，对应通过df直接设置给g的节点特征，是df列labelencode后的
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        return blocks
