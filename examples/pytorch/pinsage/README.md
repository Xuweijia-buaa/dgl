# PinSAGE example

## Requirements

- dask
- pandas
- torchtext>=0.9.0

## Prepare datasets

### MovieLens 1M

1. Download and extract the MovieLens-1M dataset from http://files.grouplens.org/datasets/movielens/ml-1m.zip
   into the current directory.
2. Run `python process_movielens1m.py ./ml-1m ./data_processed`.
   Replace `ml-1m` with the directory you put the `.dat` files, and replace `data_processed` with
   any path you wish to put the output files.

### Nowplaying-rs

1. Download and extract the Nowplaying-rs dataset from https://zenodo.org/record/3248543/files/nowplayingrs.zip?download=1
   into the current directory.
2. Run `python process_nowplaying_rs.py ./nowplaying_rs_dataset ./data_processed`

## Run model

### Nearest-neighbor recommendation

This model returns items that are K nearest neighbors of the latest item the user has
interacted.  The distance between two items are measured by Euclidean distance of
item embeddings, which are learned as outputs of PinSAGE.

```
python model.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 64
```

The implementation here also assigns a learnable vector to each item.  If your hidden
state size is so large that the learnable vectors cannot fit into GPU, use this script
for sparse embedding update (written with `torch.optim.SparseAdam`) instead:


```
python model_sparse.py data_processed --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 1024
```

Note that since the embedding update is done on CPU, it will be significantly slower than doing
everything on GPU.

The HITS@10 is 0.01241, compared to 0.01220 with SLIM with the same dimensionality.\

## Difference from the paper

The implementation here is different from what being described in the paper:

1. 无监督设置：被同一用户点击的item对为正样本对，采样负样本，用max-margin-loss训练。不像原文中，有item是否相关的显示关系：The paper described a supervised setting where the authors have a ground truth set of which items are
   relevant.  However, in traditional recommender system datasets we don't have such labels other than
   which items are interacted by which users (as well as the user/item's own features).  Therefore, I
   adapted PinSAGE to an unsupervised setting where I predict whether two items are cointeracted by the
   same user.
2. item相似度评价：向量内积. 预测时，拿用户train中最后一次点击，i2i，预测next item。
3. 特征较少。相比原文，添加了item_id特征：PinSAGE paper explicitly stated that the items do not learnable embeddings of nodes, but directly
   express the embeddings as a function of node features.  While this is reasonable for rich datasets like
   Pinterest's where images and texts are rich enough to distinguish the items from each other, it is
   unfortunately not the case for traditional recommender system datasets like MovieLens or Nowplaying-RS
   where we only have a bunch of categorical or numeric variables.  I found adding a learnable embedding
   for each item still helpful for those datasets.
3. 原文最终输出经过了MLP。这里未经过，而是直接加上自己，作为最后输出：The PinSAGE paper directly pass the GNN output to an MLP and make the result the final item
   representation.  Here, I'm adding the GNN output with the node's own learnable embedding as
   the final item representation instead.
