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
from model import SAGE
from util import create_sub,eval_metric,evaluate,to_bidirected_with_reverse_mapping
from annoy import AnnoyIndex
import logging
from sklearn.preprocessing import StandardScaler
from layer import NegativeSampler

lft = "%(asctime)s-%(message)s"
dft = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.DEBUG,format=lft)


def epoch_evaluate(args,device, graph,  model, epoch,train_h,batch_size,dataset):
    '''
    首先得到g中所有节点的最终表示
    再根据节点表示，预测。
    '''
    model.eval()
    with torch.no_grad():
        # 每个node对应的vector
        if args.usetrainh:
            node_emb= train_h                                         # TODO：直接用的训练出来的。不好
        else:
            node_emb = model.infer(graph, device, batch_size)   # 先整图infer，得到g中所有节点的最终表示。每个节点包含n阶所有邻居的信息。
        torch.save(node_emb,os.path.join(args.data_path,'node_embed_epoch{}'.format(epoch)))          # 保存embedding.对应g中node_id

        # 构建index
        index = AnnoyIndex(64, 'angular')    # 存32位向量的索引。用这个结果最好。 只是学一个聚合器。学完就学好了
        #item2nodeId = dataset['item2nodeId']
        #for aid, idx in item2nodeId.items():
        for idx,aid in enumerate(g.nodes()):
            index.add_item(idx, node_emb[idx])  # 添加每个向量，到索引中。 （按w2v中id）
        index.build(10)
        index.save(os.path.join(args.data_path,'AnnoyIndex_epoch{}'.format(epoch)))

        # 构建sub
        #args.item2nodeId=item2nodeId
        if args.eval==True:
            sub = create_sub(args, index, dataset['test_path'], mode="validation",epoch=epoch)
            eval_metric(sub,dataset['labels_path'])
        else:
            create_sub(args, index, dataset['test_path'], mode="test",epoch=epoch)


def train(args, device, g, reverse_eids,model):
    '''
    无监督训练：
        正样本： 每次迭代得到的B条边，直接作为B个正样本对：(head->tail)
        负样本： 使用negative_sampler，采样B条正边对应的负边，作为负样本对：（head->-neg_tail）
        采样邻居：为正负样本对中的所有节点，采样n阶邻居，生成n个block子图。用来聚合作为seeds的head，tail，neg_tail
    '''
    # create sampler & dataloader
    # 邻居采样器. 可以为每个节点，均匀采样n阶邻居，生成对应的n个block子图。每个block只含本阶节点和下一阶邻居。倒序放
    nei_sampler = NeighborSampler([10, 10],                # 每层可以采样固定个邻居。seeds节点对应最高层layer，可采样5个邻居。3层layer，生成3个block
                              prefetch_node_feats=['id'],   # 节点特征。设置了以后，blocks中节点会复制原图中的该特征id。
                                                              #         如最后一阶邻居的特征block[0].srcdata['id']，可作为GNN初始输入层
                              edge_dir = 'in',                # 默认采样的邻居是相邻(入边)节点，不像pinsage，采样的邻居，是随机游走得到的重要节点。
                              )

    # 边预测采样器，可以为每条输入边，采样对应的负边。并根据邻居采样器，为正负边中涉及到的所有节点，采样n阶邻居，得到对应的blocks
    sampler = as_edge_prediction_sampler(
        nei_sampler,                                         # 每个batch对应的B个heads,tail，neg_tail节点,都用该采样器采样邻居，得到对应blocks

        negative_sampler=
        NegativeSampler(g, 1,device),
        #negative_sampler.Uniform(1),        # 给输入边的每个head，随机采样一个节点，构成负边(head->neg_tail)
                                                             # 这里从全图中随机采样，可能同h,可能同+t。
                                                             # TODO:  可自定义negative_sampler，采样强负样本.
        # 无向图
        exclude='reverse_id',                                # 从采样得到的邻居子图中，删掉与minibatch相关的边。
        reverse_eids=reverse_eids.int().to(device),

        # 有向图
        #exclude='self',

    )
    use_uva = (args.mode == 'mixed')                          # 图很大，本身在cpu上，但采样出的子图，和对应计算在gpu上

    # 边迭代器：分批次，迭代给定的边list（seed_edges）。每次抽取B条边，作为B个正样本对。采样对应负样本。采样正负边对应邻居。
    seed_edges = torch.arange(g.num_edges()).int().to(device)  # g中所有边节点
    dataloader = DataLoader(
        g,
        seed_edges,                                               # 每次迭代g中的B条边，作为fact，构建子图，和对应的邻居子图
        sampler,                                                  # 为输入的B条边，采样对应的负边。并为正负边涉及到的原始节点，采样对应的n阶邻居。
        device=device, # 采样后的子图所在的device.如果使用uva,不变，还在cpu上。
        batch_size=512, shuffle=True,
        drop_last=False, num_workers=0,
        use_uva=use_uva)                                          # pin the graph and feature tensors into pinned memory.
                                                                  # 采样得到的子图，放到主机pin mem上。不往gpu上复制了。
                                                                  # 省去cpu->gpu的复制，但访存时带宽低。和model计算时，可能受限于对子图的访存速度。

                                                                  # 如果图在cpu, device在gpu, 且没用uva的话，  会设置use_alternate_streams是true
                                                                  # 用另一个流，传输子图到gpu中。
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    # 缓存训练出来的结果
    if args.usetrainh:
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)                    # mixed情况，device是cuda. pin_memory是true
        y = torch.empty(g.num_nodes(), 64, device=buffer_device, pin_memory=pin_memory)  # 用来缓存本epoch,g上所有节点的表示： （N,h）
        y[:] = g.ndata['w2v']
    else:
        y=None

    # train-loop
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0

        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
            # 每次抽取B条边：
            # pair_graph： B条边直接作为fact,构建只含B个(head->tail)的子图            （含neg_pair_graph中节点，同pinsage）
            # neg_pair_graph: 采样B条边对应的负边后，构建只含（head->-neg_tail）的子图。（如果每个pair采样K个负节点。K个pair都包含在neg_pair_graph中）
            # blocks： B条边和对应负边中，所有节点：heads,tail,neg_tail，作为seeds,采样N阶邻居。得到N个block子图。（根据邻居采样器）
            loss,h= model(pair_graph, neg_pair_graph, blocks)  # B个正样本对(head->tail)的相似度score (B，1)。 负样本对同
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # 缓存训练完实时的表示
            if args.usetrainh:
                batch_nodes_id=blocks[-1].dstdata[dgl.NID].long().to(buffer_device)  # seeds节点id
                y[batch_nodes_id,:]=h.to(buffer_device)  # 本批b个节点的最终embed. 本来在cuda上，复制到cpu的pin_mem

            #if (it+1) == 1000: break
        logging.info("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it+1)))

        # eval
        logging.info('Validation/Testing...')
        epoch_evaluate(args, device, g, model, epoch,y,batch_size=1000,dataset=dataset)

# python3 link_pred.py
# 可参考：https://docs.dgl.ai/en/latest/guide/minibatch-link.html#guide-minibatch-link-classification-sampler
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument("--w2v", default=True,type=bool,help="use pretained w2v as item embeddding or not")
    parser.add_argument("--mlp", default=False, type=bool,help="use mlp to predict link score. otherwise use dot product")
    parser.add_argument("--eval", default=False, type=bool,help="submit / eval")
    parser.add_argument("--data-path", default='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data/', type=str, help="submit / eval")
    parser.add_argument("--epoch", default=10, type=int, help="submit / eval")
    parser.add_argument("--standardize_w2v", default=False, type=bool, help="是否对w2v标准化")
    parser.add_argument("--dataset", default=1, type=int, help="0:自己的 1：别人的")
    parser.add_argument("--usetrainh", default=False, type=bool, help="infer时，使用训练时的h。不太好，学好了聚合函数，用学好的")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')             # mixed,puregpu,都默认用cuda，放模型

    # Load dataset
    print('Loading data')
    args.data_path='/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data_{}/'.format(args.dataset)
    #args.data_path = '/media/xuweijia/新加卷/Kaggle_multi-Obj-Rec/单个大数据集/data_debug/'
    data_info_path = os.path.join(args.data_path, "data.pkl")    # 其他metadata.包含item2nodeid,
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    g_path = os.path.join(args.data_path, "g.bin")               # 原始train子图
    g_list, _ = dgl.load_graphs(g_path)
    g = g_list[0].int().to('cuda' if args.mode == 'puregpu' else 'cpu')        # 图太大，可以放CPU。采样的子图放GPU   mixed:大图也放cpu

    # 把g变成无向图。每个edge,对应正反2条边.
    g.ndata["id"] = torch.arange(g.num_nodes())                                # 每个id一个向量（之后）  还有一个特征w2v. 和id一一对应
    args.standardize_w2v=True
    if args.standardize_w2v: # 是否标准化w2v
        g.ndata['w2v'] = torch.Tensor(StandardScaler().fit_transform(g.ndata['w2v']))

    # create GraphSAGE model
    model = SAGE(g,in_size=64,hid_size=64,w2v=args.w2v,mlp=args.mlp).to(device)  # 模型本身放gpu

    # model training
    print('Training...')
    train(args, device, g, reverse_eids=dataset['revers_map'],model=model)

# epoch 2
# Recall clicks: 0.326642 carts: 0.317168 orders 0.597555
# Score on validation data: 0.486347

# epoch 10
# Recall clicks: 0.326618 carts: 0.317054 orders 0.597510
# Score on validation data: 0.486284

# 换权重，差不多。好一点
# epoch 2
# Recall clicks: 0.326640 carts: 0.317157 orders 0.598545
# Score on validation data: 0.486938

# epoch 10
# Recall clicks: 0.326616 carts: 0.317043 orders 0.598499
# Score on validation data: 0.486874

# 完全用这个+topK
# epoch 2
# Recall clicks: 0.008870 carts: 0.021554 orders 0.092327
# Score on validation data: 0.062749

# epoch 10
# Recall clicks: 0.008282 carts: 0.021169 orders 0.091896
# Score on validation data: 0.062317

# train时占用8G内存，4G显存
# Kaggle上是16G 内存/显存


# 每个epoch没有break。 2个epoch. 每个epoch大概一个小时
# 2023-01-21 14:14:26,860-Epoch 00000 | Loss 0.1252
# 2023-01-21 14:14:26,860-Validation/Testing...
# Inference: 100%|██████████| 1418/1418 [00:10<00:00, 137.65it/s]
# Inference: 100%|██████████| 1418/1418 [00:10<00:00, 137.11it/s]
# Recall clicks: 0.326601 carts: 0.317049 orders 0.598496
# Score on validation data: 0.486872

# 2023-01-21 15:43:36,105-Epoch 00001 | Loss 0.0617
# 2023-01-21 15:43:36,105-Validation/Testing...
# Inference: 100%|██████████| 1418/1418 [00:10<00:00, 135.35it/s]
# Inference: 100%|██████████| 1418/1418 [00:10<00:00, 139.61it/s]
# Recall clicks: 0.326566 carts: 0.316967 orders 0.598483
# Score on validation data: 0.486837

# epoch2 纯测i2i+topK   一般
# Recall clicks: 0.008077 carts: 0.020897 orders 0.091570
# Score on validation data: 0.062019
# CPU times: user 7min 16s, sys: 2.97 s, total: 7min 19s
# Wall time: 7min 21s

# 改成index=30个树 不好了
# Recall clicks: 0.008057 carts: 0.020851 orders 0.091541
# Score on validation data: 0.061985
# CPU times: user 15min 54s, sys: 4.3 s, total: 15min 58s
# Wall time: 13min 3s

#用last_item i2i 也不好了
# Recall clicks: 0.325983 carts: 0.316409 orders 0.598114
# Score on validation data: 0.486390
# CPU times: user 8min 17s, sys: 3.98 s, total: 8min 21s
# Wall time: 8min 21s

# 3个recent aid，也差不多
# Recall clicks: 0.326568 carts: 0.316978 orders 0.597493
# Score on validation data: 0.486246
# CPU times: user 7min 45s, sys: 3.15 s, total: 7min 48s
# Wall time: 7min 48s


# 换数据集 + 只用w2v embedding + last_item
# epoch 0  (大概epoch4就好些了。5个epoch)
# 2023-01-22 22:51:47,187-Epoch 00000 | Loss 0.5061
# 2023-01-22 22:51:47,187-Validation/Testing...
# Inference: 100%|██████████| 1856/1856 [00:11<00:00, 158.14it/s]
# Inference: 100%|██████████| 1856/1856 [00:11<00:00, 161.86it/s]
# Recall clicks: 0.324590 carts: 0.312180 orders 0.592695
# Score on validation data: 0.481730
# 0.48172970117135555
# epoch 8
# 2023-01-23 09:42:09,733-Epoch 00007 | Loss 0.4032
# 2023-01-23 09:42:09,734-Validation/Testing...
# Inference: 100%|██████████| 1856/1856 [00:12<00:00, 151.91it/s]
# Inference: 100%|██████████| 1856/1856 [00:12<00:00, 153.93it/s]
# Recall clicks: 0.326763 carts: 0.313378 orders 0.593276
# Score on validation data: 0.482655
# 0.48265515326887365

# recent 3
# Recall clicks: 0.326931 carts: 0.313465 orders 0.593301
# Score on validation data: 0.482713

# 改权重：type_weight_multipliers = {0: 0.5, 1: 9, 2: 0.5} 高
# Recall clicks: 0.326930 carts: 0.313456 orders 0.594278
# Score on validation data: 0.483297

# 改index:
# angular
# Recall clicks: 0.359155 carts: 0.329087 orders 0.602484
# Score on validation data: 0.496132

# eluciean好太多
# Recall clicks: 0.359060 carts: 0.328787 orders 0.602327
# Score on validation data: 0.495939

# 直接测试w2v 人家原来不错
# (1855603, 32) "euclidean"
# Recall clicks: 0.427514 carts: 0.364051 orders 0.621035
# Score on validation data: 0.524588
# (1855603, 32) # "angular" 好
# Recall clicks: 0.432446 carts: 0.367090 orders 0.622969
# Score on validation data: 0.527153
# CPU times: user 9min 26s, sys: 3.64 s, total: 9min 29s
# Wall time: 8min 52s

# 最终5个epoch。test集上0.512
