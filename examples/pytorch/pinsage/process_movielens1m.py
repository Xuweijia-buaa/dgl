"""
Script that reads from raw MovieLens-1M data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.

This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *

import dgl

# python process_movielens1m.py ./ml-1m ./data_processed
# python process_movielens1m.py  /media/xuweijia/DATA/代码/GNN/Pinsage/dgl-pinsage/ml-1m  ./data_processed
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("out_directory", type=str)
    args = parser.parse_args()
    directory = args.directory
    out_directory = args.out_directory
    os.makedirs(out_directory, exist_ok=True)

    # directory = './ml-1m'
    # output_path ='./data_processed'

    ## Build heterogeneous graph

    # Load data
    # 用户特征df
    users = []
    with open(os.path.join(directory, "users.dat"), encoding="latin1") as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split("::")  # 每行一个样本
            users.append(
                {
                    "user_id": int(id_),     # 含user_id
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                    "zip": zip_,
                }
            )
    users = pd.DataFrame(users).astype("category")  # 变成cate类型，方便后续code对应id（相当于labelEncode）

    #item df
    movies = []
    with open(os.path.join(directory, "movies.dat"), encoding="latin1") as f:
        for l in f:
            id_, title, genres = l.strip().split("::") # 含item_id
            genres_set = set(genres.split("|"))

            # extract year
            assert re.match(r".*\([0-9]{4}\)$", title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {"movie_id": int(id_), "title": title, "year": year}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = pd.DataFrame(movies).astype({"year": "category"}) #  year特征变cate. genre特征split，变成多个0/1特征

    # 交互df
    ratings = []
    with open(os.path.join(directory, "ratings.dat"), encoding="latin1") as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [
                int(_) for _ in l.split("::")
            ]
            ratings.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,            # rating,相当于边权重
                    "timestamp": timestamp,
                }
            )
    ratings = pd.DataFrame(ratings)

    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = ratings["user_id"].unique()                # 只留rating中出现过的u,i
    distinct_movies_in_ratings = ratings["movie_id"].unique()
    users = users[users["user_id"].isin(distinct_users_in_ratings)]
    movies = movies[movies["movie_id"].isin(distinct_movies_in_ratings)]

    # Group the movie features into genres (a vector), year (a category), title (a string)
    genre_columns = movies.columns.drop(["movie_id", "title", "year"])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype("bool") # genres是bool

    movies_categorical = movies.drop("title", axis=1)         # item本身的序列类特征(text)，单独处理

    # Build graph：构建ui二分图
    graph_builder = PandasGraphBuilder()

    # 用户节点，item节点，ui边, 按照不同类型，加入图中。每个u/i一个node.每个交互一条无向边
    graph_builder.add_entities(users, "user_id", "user")                              # u
    graph_builder.add_entities(movies_categorical, "movie_id", "movie")               # i
    graph_builder.add_binary_relations(ratings, "user_id", "movie_id", "watched")     # u ->i
    graph_builder.add_binary_relations(ratings, "movie_id", "user_id", "watched-by")  # i-> u

    g = graph_builder.build()       # 构建好了异质图，UV分别是原始userid/itemid labelencoder后，作为这2类节点的node_id
                                    # 底层node_id等信息，都转成了Tensor

    # Assign features.
    # ui节点，分布设置对应的节点特征： g.nodes[‘node_type’].data[‘feat_name’]
    # 按照顺序，每类型的N个节点，设置N个特征，一一对应 （按照原始user_df）
    # 可以按照节点id去lookup:  g.nodes[‘node_type’].data[‘feat_name’][node_id1:node_id2]
    g.nodes["user"].data["gender"] = torch.LongTensor(users["gender"].cat.codes.values)   # 该类型U个节点，对应U个特征。每个离散特征，放原始特征label_encoder后的编码值
    g.nodes["user"].data["age"] = torch.LongTensor(users["age"].cat.codes.values)         # 每个离散特征，放原始特征label_encoder后的编码值
    g.nodes["user"].data["occupation"] = torch.LongTensor(users["occupation"].cat.codes.values) # 需要底层是Tensor.
    g.nodes["user"].data["zip"] = torch.LongTensor(users["zip"].cat.codes.values)
    g.nodes["movie"].data["year"] = torch.LongTensor(movies["year"].cat.codes.values)
    g.nodes["movie"].data["genre"] = torch.FloatTensor(movies[genre_columns].values)
    # 也按click_df,给每条inter边,设置特征（可设行为权重）
    g.edges["watched"].data["rating"] = torch.LongTensor(ratings["rating"].values)       # 每个边的权重 （比如inter的行为权重/评分高低）
    g.edges["watched"].data["timestamp"] = torch.LongTensor(ratings["timestamp"].values) # 每个边的时间 （inter的时间）
    g.edges["watched-by"].data["rating"] = torch.LongTensor(ratings["rating"].values)
    g.edges["watched-by"].data["timestamp"] = torch.LongTensor(ratings["timestamp"].values)

    # Train-validation-test split：
    # 根据交互记录。把每个用户的序列按时间切分：每个用户的最后一个inter作为test,倒数第二个inter作为valid。其他inter放在train中
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, "timestamp", "user_id")
    # train_indices：划分后，train中inter在click_df中的位置
    # val_indices：  划分后，val中inter  在click_df中的位置
    # test_indices： 划分后，test中inter  在click_df中的位置

    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, "watched", "watched-by")  # 用train_indices作为边id,抽取子图。节点不少，但边少了。（之后去预测边）
    assert train_g.out_degrees(etype="watched").min() > 0

    # 根据click_df中，val/test inter的位置。
    # 作为边id,去原始大图中，找对应的inter对应的UV （这些test inter对应的node_ids）
    # 根据UV，构建稀疏矩阵： (data=1,(row=U,col=V)),shape=(|U|,|I|). 行是User,列是item
    # 是大小为原图2类型节点U,I的矩阵。行号对应原图中用户node_id,列号对应原图中item的node_id
    # 预测每个非0行代表的用户，对应的item_id  （即一个valid inter）
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, "user", "movie", "watched")

    ## Build title set
    movie_textual_dataset = {"title": movies["title"].values} # item的序列特征，先放一个字典里，顺序对应item节点id。之后单独处理

    ## Dump the graph and the datasets

    dgl.save_graphs(os.path.join(out_directory, "train_g.bin"), train_g)  # 把训练子图单独存一下

    dataset = {                                                           # 原始数据metadata,单独存一下
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "item-texts": movie_textual_dataset,
        "item-images": None,
        "user-type": "user",
        "item-type": "movie",
        "user-to-item-type": "watched",
        "item-to-user-type": "watched-by",
        "timestamp-edge-column": "timestamp",
    }

    with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
        pickle.dump(dataset, f)
