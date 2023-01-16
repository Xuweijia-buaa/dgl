import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, negative_sampler
import tqdm
import argparse
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

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
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
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

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size):
        '''
        核心模型：3层layer的GraphSage。 含一个mlp，对节点对应向量进行交互，得到节点相似度score
        Source: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/sageconv.html
        ------
        in_size: 每个节点的初始embedding长度
        hid_size:每个节点最终的输出维度
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        # 3层 GraphSAGE-mean
        # 每层聚合邻居节点: h_l -> h_l+1. 对应公式：https://docs.dgl.ai/en/0.4.x/api/python/nn.pytorch.html#sageconv。
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))  # 每层graphSage。 指定输入，输出特征的维度。以及聚合方式
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean')) # 可选聚合方式：mean, gcn, pool, lstm
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.hid_size = hid_size
        #  一个mlp。 用来对每个节点对(head,tail)中uv节点对应的向量进行交互。得到每个节点对最终的相似度score:  交互向量(B,h) -> (B,1)
        #  预测时，如果已知节点最终向量。可以直接输入节点pair对应向量的element-wise乘 （B,h）,输出对应score（B,1）
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        '''
            无监督训练：
            每个批次抽取B条边，和对应的B条负边，分别作为B个正样本：(head->tail)和B个负样本:(head->neg-tail)
            其中所有节点作为seeds，根据采样得到的邻居子图，得到seeds的最终表示h
            每个节点对uv对应的2个节点向量，通过一个mlp交互，得到节点间相似度。B个正节点对，返回正节点对的相似度score（B,1）。B个负节点对同
        -------
        Args:
            pair_graph： 每个batch中的正节点对组成的子图。只含B个正节点对对应的边:(head->tail)
            neg_pair_graph: 每个batch中的负节点对组成的子图。只含与B个正节点对，对应的负节点对:(head->neg-tail)。
                            如果一个head采样了多个负节点，对应的多个负节点对，都包含在该子图中，作为多条边。
                            同pinsage，pair_graph和neg_pair_graph子图，所包含的边不同，但经过compact后，节点相同。
            blocks：  边采样场景下，把B个heads,tail,neg_tail作为seeds，采样出的N阶子图
                      其中seeds节点的顺序，同pair_graph/neg_pair_graph中的节点顺序。可用pair-graph中的节点id,索引seeds的最终表示h
            x:   blocks中，seeds节点的最后一阶邻居，对应的初始embedding。可作为block[0]对应的源节点特征,输入SageConv
        -------
        Return:
            h_pos：B个正节点对(head->tail)，每对节点的最终表示，经过mlp交互后，得到的节点相似度分数 （B,1）   (类似FM，特征向量点乘+mlp+二分类)
            h_neg：B个负节点对(head->neg_tail)，每对节点的最终表示，经过mlp交互后，得到的节点相似度分数 （B,1）
        '''
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):  # 3层SAGEConv
            # block:每阶子图
            # h:    该block源节点的表示，用来聚合目标节点。
            #            初始是最后一阶邻居的表示。
            #            之后每层，是上个block得到的dst节点的表示，也是本层block src节点的新表示，可用来聚合本阶dst节点
            h = layer(block, h)          # 消息传递。 h_l -> h_l+1 聚合邻居信息。（如没有邻居，聚合结果只含该dst节点本身的信息）

            # 每层之间加Relu+Dropout
            if l != len(self.layers) - 1:
                h = F.relu(h)            # 最终得到每个seeds节点的最终表示。对应本批次B个head,tail,neg_tail节点
                                         # 每个向量，对应pos、neg graph中的一个节点id

        pos_src, pos_dst = pair_graph.edges()          # B个正节点对的u，v ： (head->tail)。节点id可用来索引seeds的最终表示
        neg_src, neg_dst = neg_pair_graph.edges()      # B个负节点对的u，v ： (head->neg_tail). 节点id也可用来索引seeds的最终表示
                                                        #                   这里每个head只采样了一个负样本。对应B个neg_tail

        # 按heads,tail等节点的节点id，索引h,得到这些节点的最终表示。经过一个mlp,对每个节点对中的2节点向量进行交互，得到每个节点对的节点相似度。（二分类logits）
        h_pos = self.predictor(h[pos_src] * h[pos_dst])  # 按heads,tail等节点的节点id，索引h,得到这些节点的最终表示
                                                         # B个head和B个tail对应的向量，分别element-wise乘。计算相似度-> (B,h) * (B,h) -> (B,h)
                                                         # 再经过一个mlp,得到正节点对的预测score:(B,h) ->mlp -> (B,1)
        h_neg = self.predictor(h[neg_src] * h[neg_dst])  # 同上，根据负节点对对应的向量，得到每对负样本的score:     （B,1）

        # TODO:
        #     1  对于每个pair,score也可以直接用向量内积. (上边是用了向量用mlp交互后的结果，类似FM)
        #         edge_subgraph.ndata['h'] = h
        #         edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'))  作为边特征
        #         h_pos=edge_subgraph.edata['score']
        #     2 loss也可以用pair-wise loss, 最大化h_pos与对应h_neg之间的分数差。类似pinsage  （这里用的是point-wise,为每个正负样本本身预测label）
        #         如margin hinge loss:
        #               loss=(neg_score - pos_score + 1).clamp(min=0).mean()  # max(0, n-p+1) (B,1)

        # 参考：https://docs.dgl.ai/en/latest/guide/minibatch-link.html#guide-minibatch-link-classification-sampler
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """
        Offline inference: 线下推断g中每个节点的最终表示。每个节点聚合了全部n阶邻居。（N,h）
                           device:    model放gpu,上一阶表示作为输入放gpu。迭代的每B个节点和对应子图在gpu上做计算，结果缓存在cpu上
                           unseen节点：加入g后，也可以根据邻居信息聚合，infer对应表示。
        -------
        同node_classification.py:
            训练时，为了能把计算放GPU，只采样了部分邻居。但线下infer时，每个节点最好聚合全部邻居，得到信息最丰富的表示。
            为了聚合自己的全部邻居，每层的计算需要得到整个图的上一层表示，因此按层进行计算。Layer-wise inference
            参考：https://docs.dgl.ai/en/latest/guide/minibatch-inference.html
        """
        feat = g.ndata['feat']                                                   # 初始特征 (N，d)
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat']) # 邻居采样器。infer时，每个节点聚合自己的全部邻居。
                                                                                 # 只采用一层。每个batch只返回一个block，对应节点和其所有邻居
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device),                        # 迭代g中所有nodes。 所有nodes_id放gpu中，计算发生在gpu上
            sampler,
            device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        # 得到所有节点的最终表示，分层计算.Compute representations layer by layer
        for l, layer in enumerate(self.layers): # 第i层, 得到所有节点的该阶表示. 下一阶节点的计算，依赖所有节点的上一层表示
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,   # 用来缓存该阶，g上所有节点的表示： （N,h）
                            pin_memory=pin_memory)                                #            整图表示，放在cpu锁页内存上。传输快些
            feat = feat.to(device)  # 每层的全部节点，上一阶得到的表示。放gpu,用作计算输入

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]       # 这些节点的邻居，上一阶的表示（基于上一阶计算结果得到/初始是feat_embedding）
                h = layer(blocks[0], x)     # 根据节点的邻居(block)和对应的上一阶表示(x)，聚合得到该节点高一阶表示 （B,h）
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)  # 把本层每个batch得到的结果（gpu上计算的），放cpu
            feat = y                         # 作为下一阶计算时，所有节点的初始表示（N,h）
        return y                             # 返回所有节点的最终表示

def compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500):
    """
       Compute Mean Reciprocal Rank (MRR) in batches.
       把src节点和待预测节点的最终表示，element-wise乘后，直接输入训练好的mlp交互,预测节点对间的score.
       可以按score对dst-item排序。和真实购买比较，算指标
       ----
       Args:
           node_emb:g中所有节点的最终表示。每个节点包含n阶所有邻居的信息。
           src:    测试边对应的源节点
           dst:    测试边对应的目标节点。待预测边（g中应该没有）
           neg_dst: 测试边对应的neg节点
    """
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):  # 每次输入B个源节点
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)   # 对应的B个目标节点 + B和neg节点
        h_src = node_emb[src[start:end]][:, None, :].to(device)              # 源节点(在g中)对应的最终向量。
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device) # 目标节点和neg节点对应的最终向量
        pred = model.predictor(h_src*h_dst).squeeze(-1)                       # 直接用mlp,计算src与预测节点的节点相似度 （底层是mlp对向量交互）
                                                                              # 得到src对target,neg_target的score.
        input_dict = {'y_pred_pos': pred[:,0], 'y_pred_neg': pred[:,1:]}      # 输入,算MRR:每个正确items,score越大越好。最后算平均
        rr[start:end] = evaluator.eval(input_dict)['mrr_list']
    return rr.mean()

def evaluate(device, graph, edge_split, model, batch_size):
    '''
    首先得到g中所有节点的最终表示。每个节点包含n阶所有邻居的信息。（同node-classification）
    再根据节点表示，为每个src节点，和待预测的dst节点，通过model中的mlp, 计算向量交互相似度。
    '''
    model.eval()
    evaluator = Evaluator(name='ogbl-citation2')
    with torch.no_grad():
        node_emb = model.inference(graph, device, batch_size)   # 先整图infer，得到g中所有节点的最终表示。每个节点包含n阶所有邻居的信息。
        results = []
        for split in ['valid', 'test']:
            src = edge_split[split]['source_node'].to(node_emb.device)           # 测试边。对应的源节点
            dst = edge_split[split]['target_node'].to(node_emb.device)           #        对应的目标节点
            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)   #        对应的neg节点
            results.append(compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device))
    return results

def train(args, device, g, reverse_eids:list, seed_edges, model):
    '''
    无监督训练：
        正样本： 每次迭代得到的B条边，直接作为B个正样本对：(head->tail)
        负样本： 使用negative_sampler，采样B条正边对应的负边，作为负样本对：（head->-neg_tail）
        采样邻居：为正负样本对中的所有节点，采样n阶邻居，生成n个block子图。用来聚合作为seeds的head，tail，neg_tail
    训练时：
       1 正负样本对中的节点，经过Sage网络得到每个节点的最终表示
       2 每条边中，节点uv对应的向量，经过mlp后融合，作为uv的节点相似度。类似FM中特征交叉   u(B，h) * v(B,h) -> (B,h) -> mlp -> score(B,1)
         （可以直接用节点uv的向量内积，作为score （B,h）dot (B,h) -> (B,1)）
       3 进行point-wise的学习：为每个正负样本本身，预测对应label
             正样本对对应的label是1，负样本对对应label是0
             样本得出的score，和对应label进行比较，计算二分类交叉熵loss。使得正样本score接近1，负样本对score接近0
         （可以用pair-wise-loss,最大化正负样本对score之差  loss=max(0, n-p+1) neg_score(B,1)-pos_score(B,1)+1  ）
    没有val

    参考：https://docs.dgl.ai/en/latest/guide/minibatch-link.html#guide-minibatch-link-classification-sampler
    --------
        g:           无向图.每个edge,对应正反2条边.
                     是一个同质图，节点类型相同
                     其中的每条边两端的节点uv,作为fact,构成正样本pair。v是u的预测节点。
        reverse_eids：g中所有边，边id到其反向边的映射： reverse_eids[eid]:reverse_eid
        seed_edges:   g中所有边（edge_id），作为fact,采样对应的负节点,用来迭代。
    '''
    # create sampler & dataloader
    # 邻居采样器. 可以为每个节点，均匀采样n阶邻居，生成对应的n个block子图。每个block只含本阶节点和下一阶邻居。倒序放
    nei_sampler = NeighborSampler([15, 10, 5],                # 每层可以采样固定个邻居。seeds节点对应最高层layer，可采样5个邻居。3层layer，生成3个block
                              prefetch_node_feats=['feat'],   # 节点特征。设置了以后，blocks中节点会复制原图中的该特征。
                                                              #         如最后一阶邻居的特征block[0].srcdata['feat']，可作为GNN初始输入层
                              edge_dir = 'in',                # 默认采样的邻居是相邻(入边)节点，不像pinsage，采样的邻居，是随机游走得到的重要节点。
                              )

    # 边预测采样器，可以为每条输入边，采样对应的负边。并根据邻居采样器，为正负边中涉及到的所有节点，采样n阶邻居，得到对应的blocks
    # 输入是B条边：每条输入边. 作为fact，被看做是源节点的正样本对(head->tail)
    #            采样负样本：如果指定了neg_sampler,会为每个head节点,采样K个节点，作为head节点的K个负样本对:（head->-neg_tail）
    #            邻居采样：正负pair对应的B个heads,tail，neg_tail节点，均作为seeds，用指定的邻居采样器，采样各自的n阶邻居，用来聚合各自的表示。
    #  类似pinsage，为每个batch中的B个heads节点,采样正负样本对：只不过pinsage用i->u->i，采样与head被共同购买过的节点，作为正样本
    #            这里用head直接相连的节点，作为正样本。相当于本图中有边相连的节点，已经是有co-purchase关系的item了，可以互相作为正样本。
    sampler = as_edge_prediction_sampler(
        nei_sampler,                                         # 每个batch对应的B个heads,tail，neg_tail节点,都用该采样器采样邻居，得到对应blocks

        negative_sampler=negative_sampler.Uniform(1),        # 给输入边的每个head，随机采样一个节点，构成负边(head->neg_tail)
                                                             # 这里从全图中随机采样，可能同h,可能同+t。可自定义negative_sampler，采样强负样本

        reverse_eids=reverse_eids,                           # g中所有边，边id与其反向边id之间的映射。mapping。 reverse_eids[eid]:r_eid

        exclude='reverse_id',                                # 从采样得到的邻居子图中，删掉与minibatch相关的边。
                                                             # tail/neg_tail是head要预测的节点
                                                             # h聚合邻居时，如果聚合了t的信息，再去预测t,会导致信息泄露
                                                             # 为防止信息泄露，采样得到的邻居子图，需删掉待预测的边ht+对应反向边th ('reverse_id')
                                                             #                               （反向边的边id,根据上边提供的反向边mapping推断）

        # Notes: 设置exclude后，每个head至少需要2个入边邻居，才能在计算ht时，聚合另外的邻居
        #        否则每个block，无法为该节点采样到邻居。block中这类节点无入边(邻居).
        #        用SageConv聚合邻居时，block仍能经过model得到最终的表示
        #        但对于无邻居的dst节点，聚合结果只包含本节点信息,纯邻居聚合结果是0向量：
        #        但对于无邻居的dst节点，聚合结果只包含本节点信息,纯邻居聚合结果是0向量：
        #                      对graphSage-mean,聚合结果是邻居聚合结果+dst节点本身特征mlp。前半部分是0。只含节点本身特征.
        #                      对graphSage-gcn, 聚合结果是mean(邻居聚合结果+dst节点本身特征)。前半部分是0。只含节点本身特征skip

    )
    use_uva = (args.mode == 'mixed')

    # 边迭代器：分批次，迭代给定的边list（seed_edges）。每次抽取B条边，作为B个正样本对。采样对应负样本。采样正负边对应邻居。
    dataloader = DataLoader(
        g,
        seed_edges,                                               # 每次迭代g中的B条边，作为fact，构建子图，和对应的邻居子图
        sampler,                                                  # 为输入的B条边，采样对应的负边。并为正负边涉及到的原始节点，采样对应的n阶邻居。
        device=device, batch_size=512, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    # train-loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
            # 每次抽取B条边：
            # pair_graph： B条边直接作为fact,构建只含B个(head->tail)的子图            （含neg_pair_graph中节点，同pinsage）
            # neg_pair_graph: 采样B条边对应的负边后，构建只含（head->-neg_tail）的子图。（如果每个pair采样K个负节点。K个pair都包含在neg_pair_graph中）
            # blocks： B条边和对应负边中，所有节点：heads,tail,neg_tail，作为seeds,采样N阶邻居。得到N个block子图。（根据邻居采样器）
            x = blocks[0].srcdata['feat']          # seeds节点，最后一阶邻居的特征。作为GNN的初始输入.

            # 根据本批次的heads,tail,neg_tail和采样的邻居，得到每个节点的最终表示。
            # 并经过一个mlp,计算每个节点对,uv 2节点的相似度。(根据uv节点对应的向量表示)
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)  # B个正样本对(head->tail)的相似度score (B，1)。 负样本对同

            score = torch.cat([pos_score, neg_score])  # （2B,1）  2B个节点对，每对得出的相似度
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])  # 每个节点对看做一个样本， (h,t) ->预测为1. （h->-t)，label是0
            loss = F.binary_cross_entropy_with_logits(score, labels) # 按二分类计算loss. point-wise,不是希望拉大差距，而是给负样本学一个分数0
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if (it+1) == 1000: break
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it+1)))

# python3 link_pred.py
# 可参考：https://docs.dgl.ai/en/latest/guide/minibatch-link.html#guide-minibatch-link-classification-sampler
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')

    # load and preprocess dataset。直接建好图
    print('Loading data')
    dataset = DglLinkPropPredDataset('ogbl-citation2')
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # 把g变成无向图。每个edge,对应正反2条边.
    g, reverse_eids = to_bidirected_with_reverse_mapping(g) # reverse_eids：g中每条边，到其反向边的映射： reverse_eids[eid]:rv_eid
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)    # g中所有边节点
    edge_split = dataset.get_edge_split()                  # train,val,test分别对应原图中，不同的边id

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]                     # 输入gnn的初始特征维度。
    model = SAGE(in_size, 256).to(device)

    # model training
    print('Training...')
    train(args, device, g, reverse_eids, seed_edges, model)

    # validate/test the model
    print('Validation/Testing...')
    valid_mrr, test_mrr = evaluate(device, g, edge_split, model, batch_size=1000)
    print('Validation MRR {:.4f}, Test MRR {:.4f}'.format(valid_mrr.item(),test_mrr.item()))
