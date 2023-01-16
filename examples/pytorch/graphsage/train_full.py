import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        '''
            2层layer的GraphSage网络
            ------
            in_size: 每个节点的初始embedding长度
            out_size:每个节点最终的输出维度。  node classification中，也是节点类型数目，对应logits
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        # 2层 GraphSAGE-mean
        #     每层聚合,聚合邻居节点: h_l -> h_l+1. 对应公式：https://docs.dgl.ai/en/0.4.x/api/python/nn.pytorch.html#sageconv。
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))    # 每层graphSage。 指定输入，输出特征的维度。以及聚合方式
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))   # 可选聚合方式：mean, gcn, pool, lstm
        self.dropout = nn.Dropout(0.5)


    def forward(self, graph, x):
        '''
        每层之间用了一个Relu,DropOut.得到最终每个节点的表示。
        不采样邻居，直接把全部源节点作为邻居，聚合全部邻居产生的消息。
        -------
        graph:  用来update_all。聚合邻居时，作为消息传递的中介:
                每层的输入特征，被设置到g.srcdata['h']上，产生/更新消息。
                最后拿出每个节点上的聚合结果graph.dstdata['neibor'],作为该层的输出：（N,D）。 是该层每个节点的新表示
        x: 每个节点的初始embedding（N,D）
        '''
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)                # h_l (N,in) -> h_l+1 (N,out). 用全图时，每层不采样，直接把全部入边作为邻居，聚合全部邻居的消息。
            if l != len(self.layers) - 1:      # 层之间，加上Relu+dropout
                h = F.relu(h)
                h = self.dropout(h)
        return h  # 最终返回每个节点的最终表示：（N,out_size）   这里out_size是C，用来进行节点分类 （相当于logits）


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)   # 仍是全部节点的logits。 （N,h） 每个节点的最终表示
        logits = logits[mask]
        labels = labels[mask]         # 只用val节点的真实类型。logits. 看准确率
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        # 输出每个节点的最终表示：（N, out_size）   这里out_size是C，用来进行节点分类 （相当于logits (N，C)）
        logits = model(g, features)
        # 计算loss(只用所有训练节点)
        loss = loss_fcn(logits[train_mask], labels[train_mask])  # 交叉熵分类损失: pred（N,C）  labels:(N,)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)    # 预测val节点类型
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    '''
    节点分类任务
    每个epoch,用整个图的节点训练。不分minibatch.
             （minibatch版本可参考node_classification.py）
    命令：python3 train_full.py --dataset cora --gpu 0
    '''
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed')", # 3个数据集，对应task都是节点类型预测。有监督。每个节点有固定类型
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphSage module")

    # load and preprocess dataset。直接把原始数据集建成图了
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)

    features = g.ndata["feat"]                          # 每个节点上的初始特征(embedding)。[N,d]
    labels = g.ndata["label"]                           # 每个节点的类型。 有监督训练，去预测该节点类型 [N,] 取值C种。
    masks = g.ndata["train_mask"], g.ndata["val_mask"]  # 全图节点。哪些用来train(算loss),哪些用来预测节点类型。 都是（N,）,用true/false区分

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = SAGE(in_size, 16, out_size).to(device)     # 初始化model. 输出每个节点的类型。（N,C）

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, g.ndata["test_mask"], model)
    print("Test accuracy {:.4f}".format(acc))
