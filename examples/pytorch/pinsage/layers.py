import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn


def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def _init_input_modules(g, ntype, textset, hidden_dims):
    '''
    给每个item特征，独立设置embedding。可以根据g中特征的原始值，lookup
    ----------
    ntype：只考虑这类节点的特征。（这里传入item类节点的类型名）
    ----------
    module_dict[f_name]:
        f是int64类型的特征： 默认是labelencoder后的离散特征。对应一个[C,h]的embedding
        f是float类型特征：  对应一个Linear层。输入默认是[B,d], 经过该Linear成为[B,h]的layer
    '''
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()

    for column, data in g.nodes[ntype].data.items():  # item类节点的所有特征. data是所有节点的特征取值（数据处理时，设置在train_g中）
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims) # 作为连续特征，对应一个Linear层
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:               # 作为离散特征，对应一个embedding [C,h]
            assert data.ndim == 1
            m = nn.Embedding(data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    if textset is not None:
        for column, field in textset.items():
            textlist, vocab, pad_var, batch_first = field
            module_dict[column] = BagOfWords(vocab, hidden_dims)

    return module_dict


class BagOfWords(nn.Module):
    def __init__(self, vocab, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(vocab.get_itos()),
            hidden_dims,
            padding_idx=vocab.get_stoi()["<pad>"],
        )
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()


class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """

    def __init__(self, full_graph, ntype, textset, hidden_dims):
        '''
        初始化g中每个特征的映射层，得到module_dict[f_name]
           每个int64类型的特征f： 默认是labelencoder后的离散特征。初始化一个[C,h]的embedding
           每个float类型特征f：   默认原特征是[B,d]的连续特征, 初始化一个(d,h)的Linear层
        -----
        ntype:只考虑该类型节点的特征 （item特征）
        '''
        super().__init__()

        self.ntype = ntype
        self.inputs = _init_input_modules(
            full_graph, ntype, textset, hidden_dims   # module_dict[f_name]. 对应每个特征的映射层
        )

    def forward(self, ndata):
        '''
        ndata: 某个block中，对应节点的原始特征 ,是一个dict: {f_name:tensor[]}
        ----------
        每个特征，
        '''
        projections = []
        for feature, data in ndata.items():              # block中。节点对应的每个特征。data:各节点上的特征取值,tensor.
            if feature == dgl.NID or feature.endswith("__len"):
                # This is an additional feature indicating the length of the ``feature``
                # column; we shouldn't process this.
                continue

            module = self.inputs[feature]                  # 该特征对应的初始映射层
            if isinstance(module, BagOfWords):
                # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + "__len"]
                result = module(data, length)
            else:
                result = module(data)                      # 原始特征，经过embedd/Linear: (B,) -> (B,h)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)          # 简单把每种特征向量相加  stack: (B,|F|,h) ->sum  (B,h)


class WeightedSAGEConv(nn.Module):
    '''
    每层网络：聚合该层邻居
    '''
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g: 一个block。只包含本层节点和下一阶邻居。 邻居节点指向seeds.  构建block时，源节点包含dst节点。
        h: 本层block的每个节点,通过上一层得到的向量
        weights:该block中，邻居节点到目标节点的权重。代表邻居采样时，该邻居节点的重要性. 是取值，tensor(|E|,1)
        """
        h_src, h_dst = h       # h_src:该block中源节点的current embedding。
                               # h_dst:该block中目标节点的current embedding。包含在h_src中
        with g.local_scope():
            # 源节点hv本身做变换：
            # h_v=Relu(Q*h_v+q)  仍是h维。相当于map,作为源节点的特征'n'
            g.srcdata["n"] = self.act(self.Q(self.dropout(h_src)))

            # 为目标节点，加权聚合邻居节点（按邻居权重）. 用dgl中内置的消息传递api,更新节点特征
            # h_u= sum(w*h_v)。    相当于每个dst节点，加权reduce
            g.edata["w"] = weights.float()            # 邻居权重，作为边特征
            g.update_all(fn.u_mul_e("n", "w", "m"),   # 每条边的源节点中特征'n'(含dst节点), 乘上对应边权重'w',产生消息“m”(在该边上，类似边特征).
                         fn.sum("m", "n"))            # 目标节点的特征'n',是加权聚合所有邻居的h_v: h_u= sum(w*h_v)  （reduce:每个dst节点，聚合所有入边消息m）
            n = g.dstdata["n"]                        # 目标节点的临时新表示，对应文中n_u  [B,h]

            g.update_all(fn.copy_e("w", "m"), fn.sum("m", "ws")) # 每个dst节点，聚合所有入边权重,用来归一化n_u （原始w是访问次数，未归一化）
            ws = g.dstdata["ws"].unsqueeze(1).clamp(min=1)       # 存入dst节点的'ws'字段

            # 加自己的初始embed：z_u= Relu(W * (concat[n_u,h_u]) + b)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))  # [B,h]   拼上自己后，又通过W,映射到h
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(
                z_norm == 0, torch.tensor(1.0).to(z_norm), z_norm
            )
            z = z / z_norm                                                      # 归一化。得到本次block的目标节点，对应的新的表示
            return z


class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        核心模型：含每层conv
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims)
            )

    def forward(self, blocks, h):
        '''
        Args:
            blocks: 已经采样好的各阶节点
                    每个block子图，只含本阶节点和下一阶邻居。每个邻居指向对应seeds：
                    blocks[-1]: 初始B个item作为seeds， 每个节点采样n个邻居，得到一个只含seeds和一阶邻居节点的二分图block。
                    blocks[-2]: 用上次采样得到的一阶邻居，作为新的seeds,再采样，得到只含一二阶邻居的二分图block
                    ...
                    blocks[0]: 放最后一阶邻居对应的子图。从最后一阶邻居开始聚合
            h: 最后一阶邻居对应的block。源节点的原始特征聚合后，得到的初始embedding。每个block的源节点包含dst节点（位于前dst个）
        Returns:
           blocks中的输入节点，经过多次conv聚合后的表示:(B,h）
        '''
        for layer, block in zip(self.convs, blocks):               # 每层layer
            h_dst = h[: block.num_nodes("DST/" + block.ntypes[0])] # 本次block中，目标节点的初始向量 （位于block源节点的前|dst|个位置）
            h = layer(block, (h, h_dst), block.edata["weights"])   # 得到本次block的目标节点，对应的新的向量表示
                                                                   # 而本次的目标节点，是下次block的源节点。作为h_src输入，去聚合更上层的节点表示。
        return h     # 最原始的输入节点，聚合n层邻居后的表示:(B,h）


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()

        n_nodes = full_graph.num_nodes(ntype)  # g中item节点总数
        self.bias = nn.Parameter(torch.zeros(n_nodes, 1))  # 每个节点学习一个bias

    def _add_bias(self, edges:dgl.udf.EdgeBatch):  # 自定义的消息传递函数。
        "edges,本质是一批edge。可以通过src, dst，data属性来获取u,v节点特征，边特征"
        bias_src = self.bias[edges.src[dgl.NID]]   # 源节点的bias
        bias_dst = self.bias[edges.dst[dgl.NID]]   # 目标节点的bias
        return {"s": edges.data["s"] + bias_src + bias_dst}  # 最终的head->tail的边权重，加上学习到的节点本身的bias

    def forward(self, item_item_graph, h):
        """
        item_item_graph : pos_graph/neg_graph. 只包含B个head->tail的子图。
        h : 子图中，每个节点经过网络后的最终表示
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata["h"] = h
            item_item_graph.apply_edges(fn.u_dot_v("h", "h", "s"))  # 用消息传递api,计算head和tail的向量内积。存于边上特征's'中
            item_item_graph.apply_edges(self._add_bias)             # 该分数，加上一个可学习的bias,是该样本对的最终score (TODO：可以不用)
            pair_score = item_item_graph.edata["s"]
        return pair_score   # 该子图中，B个pair根据向量内积得到的相似度
