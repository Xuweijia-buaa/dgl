import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

'''
可以通过Neiborsampler+dataloaer，把节点按批次输入+采样对应的blocks, 把整图训练，改成minibatch的方式，得到每个批次的节点表示
可以对比train_full.py（整图版本）和本文件（minibatch版本）。
主要是把原来传入的g ->  换成每B个seeds采样得到的blocks,
     g上对应的特征x -> 换成每个block对应的srcdata的特征。（初始是最后一阶邻居的特征）
整图版本，直接传入的整图，每个dst节点，聚合所有入边邻居。
minibatch版本，用sampler指定了每阶能采样的邻居，构建block。
https://docs.dgl.ai/en/latest/guide_cn/minibatch-node.html#guide-cn-minibatch-node-classification-sampler
'''
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        '''
        核心模型：3层layer的GraphSage
        Source: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/sageconv.html
        ------
        in_size: 每个节点的初始embedding长度
        out_size:每个节点最终的输出维度。  node classification中，也是节点类型数目，对应logits
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        # 3层 GraphSAGE-mean
        # 每层聚合邻居节点: h_l -> h_l+1. 对应公式：https://docs.dgl.ai/en/0.4.x/api/python/nn.pytorch.html#sageconv。
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))   # 每层graphSage。 指定输入，输出特征的维度。以及聚合方式
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))  # 可选聚合方式：mean, gcn, pool, lstm
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        '''
        每个批次的seeds,根据采样得到的n层block子图（和对应的初始特征），经过多次SageConv,得到这B个seeds的最终表示。
        -----
        blocks: mini-batch设置下，该batch采样出的n阶子图。
                原来的整图graph:  用来作为消息传递的中介。每层的输入特征，被设置到g.srcdata['h']上，产生消息，更新到graph.dstdata['neibor']中
                                用了graph.srcdata/dstdata/update_all等API。
                                且需要x是该图上所有节点的全部特征(N，D)，把这些特征设置到g.srcdata上
                替换成minibatch后： 用sampler采样了n阶子图。
                                每阶子图，输入与该子图匹配的源节点的初始特征h。可以适配
        x: blocks中，最后一阶邻居的初始embedding。作为block[0]对应的特征,输入SageConv。（B,D）
        '''
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # block:每阶子图
            # h:    该block的源节点的初始表示。最初是最后一阶邻居的初始表示。 之后每层，是上个block得到的邻居i的新表示，用来聚合上一层邻居
            h = layer(block, h)         # 消息传递。

            # 每层中间插入Relu+Dropout
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h  # 本批次B个节点的最终表示 （B,d）

    def inference(self, g, device, batch_size):
        """
        inputs:
            g:完整的图
        returns:
            y:g中每个节点，最终的表示（N,h）。每个节点的最终表示，聚合了所有邻居,信息最完整
        -------
         infer时，每个节点需要聚合自己的全部邻居，得到最完整的表示
                 为了计算每个节点的下一阶表示，需要得到整个图的上一阶表示，因此按层进行计算：
                 计算第i层节点的表示时，只使用layer[i]和图中所有节点的上一层表示y[i] （N,h）.
                        防止太大，该层计算时，按批次切分节点
                        每B个节点，聚合自己的所有邻居，得到block信息。根据邻居信息和对应的上一阶表示，聚合得到该节点高一阶表示 (B,h)
                 缓存本阶所有节点的表示，作为计算所有节点的下一阶表示时的初始特征
         最终得到所有节点的最终表示：（N,h）
        """
        feat = g.ndata['feat']  # 初始特征 (N，d)

        # 邻居采样器。infer时，每个节点聚合自己的全部邻居。
        sampler = MultiLayerFullNeighborSampler(1,                             # 只采用一层。每个batch只返回一个block，对应节点和其所有邻居
                                                prefetch_node_feats=['feat'])  # 使得block复制节点特征'feat'。但未用，特征都是直接传入的

        dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),                         # 迭代该图中所有nodes
                sampler,
                device=device,batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        # 得到所有节点的最终表示，分层计算

        # 每B个节点，聚合自己的所有邻居。子图block和这些节点邻居的上一层表示，传入layer[i]
        for l, layer in enumerate(self.layers):  # 每层i, 得到所有节点的该阶表示. 下一阶节点的计算，依赖所有节点的上一层表示
            y = torch.empty(                                    # 用来缓存该阶，g上所有节点的表示： （N,h）
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1
                else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader): # 得到B个节点的上一阶表示
                # output_nodes: 本批次B个seeds
                # input_nodes:  这B个节点的全部一阶邻居
                # blocks：只包含一个block,是B个节点，和这B个节点的全部邻居。复制了特征'feat'

                # 所有节点，下一阶表示只基于layers[i]==layer计算。基于这些节点邻居的上一阶表示x
                x = feat[input_nodes]   # 这些节点的邻居，上一阶的表示（基于上一阶计算结果得到/初始是feat_embedding）

                h = layer(blocks[0], x) # 下一阶表示只基于layers[i]==layer计算。
                                        # 根据节点的邻居(block)和对应的上一阶表示(x)，聚合得到该节点高一阶表示 （B,h）
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)  # B个测试节点，经过该层后的高阶表示
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)  # 把每B个节点该阶的表示，缓存到y中
            feat = y   # 作为下一阶计算时，所有节点的初始表示（N,h）
        return y  # 返回所有节点的最终表示

def evaluate(model, graph, dataloader):
    '''
    dataloader:valid用dataloader，迭代valid节点。
    每批节点根据采样后的blocks,得到对应的表示.与labels比较，返回节点预测准确率
    '''
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):  # 迭代测试节点
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])            # 测试节点。blocks中保存的label (B,1)
            y_hats.append(model(blocks, x))                   # 得到每B个测试节点的表示 (B,C)
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))      # （N,C）/ (N,1) 计算节点预测准确率。 每个节点预测正确，是1

def layerwise_infer(device, graph, nid, model, batch_size):
    '''
    根据训练好的model，计算graph每个节点的最终表示(使用全部邻居信息)。抽取nid对应节点，infer对应节点类型
    ----
    nid: test节点的node_id'
    '''
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # g中每个节点，最终的表示（N,h）。  pred in buffer_device.
        pred = pred[nid]                                  # 测试节点的最终表示    （可作为logits）
        label = graph.ndata['label'][nid].to(pred.device) #        对应的节点类型 (可作为label)
        return MF.accuracy(pred, label)

def train(args, device, g, dataset, model):
    '''
    按minibatch训练：
          每个epoch,通过dataloaer把节点切成batch，dataloaer中用sampler为每个batch的B个节点，采样n阶邻居
          返回每个batch采样得到的n个block子图，只含每阶邻居和seeds。（以及必要输入特征）
    训练时：
          迭代的是train_id
          每个batch对应的B个train节点，得到的邻居子图和对应特征，经过model,得到B个节点的最终表示
          与节点label比较，计算分类loss
    val时：
           迭代的是g中val节点。
           使用相同sampler采样邻居，得到每个batch val节点的最终表示
           与节点label比较，eval
    '''
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)

    # 邻居采样器。每个batch的节点，用该sampler从原图中采样n阶邻居（每阶邻居数目不同），生成该batch对应的n_layer个block.
    sampler = NeighborSampler([10, 10, 10],                   # 共3层layer.各节点每阶可以采样10个邻居。3层，每个批次生成3个block
                              prefetch_node_feats=['feat'],   # 节点特征。设置了以后，blocks中节点会复制原图中的该特征。
                                                              #         最后一阶邻居block[0].srcdata，可作为GNN初始层
                              prefetch_labels=['label'])      # 节点特征，设置了以后，blocks会复制原图中的该特征。
                                                              #         block中的seeds的label,节点分类时可用
    use_uva = (args.mode == 'mixed')

    # 节点迭代器：分批迭代节点ID数组（train_idx）
    #           每次抽取B个节点和对应的n阶邻居，返回该批次对应的n个block子图。可用来聚合这B个节点的表示。
    train_dataloader = DataLoader(g,
                                  train_idx,                            # 要迭代的训练节点list。每个batch用B个,作为seeds
                                  sampler,                              # 用来给B个seeds采样邻居。每个节点采样n阶邻居，得到n个block子图
                                  device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)                     # Unified Virtual Addressing。图太大，放不到GPU的话

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,    # 迭代测试节点，每次迭代B个节点，得到测试节点的表示
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            # minibatch设置下，不用全图。每次迭代train_idx中的B个节点，作为seeds，采样n层邻居（用sampler）
            # blocks：该批次的B个train节点,采样3层邻居（sampler中定义并采样邻居），得到的3层block子图
            x = blocks[0].srcdata['feat']          # 最后一阶邻居，复制的原图中的该特征
            y = blocks[-1].dstdata['label']        # seeds复制了原图中的label特征。用来做节点分类. (B,1)

            y_hat = model(blocks, x)               # B个节点的最终表示（B,d） （经过多层SageConv）

            loss = F.cross_entropy(y_hat, y)       # 交叉熵损失
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))

if __name__ == '__main__':
    '''
    节点分类任务（minibatch版本）
    每个epoch, 把节点切分minibatch。
              训练时：采样B个节点，每个节点每层采样部分邻居，得到block子图。最终得到B个节点的表示，和节点label比较，算节点分类交叉熵loss
              val时：只是dataloader使用的节点集合不同，用的是val节点。sampler同train，都只采样部分节点，形成子图，聚合节点表示。和节点label比较，eval
              test时：(infer) 模拟线上
                      每层每个节点聚合全部邻居，得到节点最全的表示。最终得到g上每个节点信息最丰富的N阶表示。抽取test节点。预测节点类型
                      为了聚合全部邻居，线下先按层计算。每层得到g上全部节点该阶表示，作为下一层计算的基础。计算单层表示时，也按mini-batch分批计算。
    命令：python3 train_full.py --dataset cora --gpu 0
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')

    # load and preprocess dataset，并生成图。同样是节点分类
    print('Loading data')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print('Training...')
    train(args, device, g, dataset, model)

    # test the model
    print('Testing...')
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
