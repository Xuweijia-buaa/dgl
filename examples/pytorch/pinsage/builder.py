"""Graph builder from pandas dataframes"""
from collections import namedtuple

from pandas.api.types import (
    is_categorical,
    is_categorical_dtype,
    is_numeric_dtype,
)

import dgl

__all__ = ["PandasGraphBuilder"]


def _series_to_tensor(series):
    if is_categorical(series):
        return torch.LongTensor(series.cat.codes.values.astype("int64"))
    else:  # numeric
        return torch.FloatTensor(series.values)


class PandasGraphBuilder(object):
    """Creates a heterogeneous graph from multiple pandas dataframes.

    Examples
    --------
    Let's say we have the following three pandas dataframes:

    User table ``users``:

    ===========  ===========  =======
    ``user_id``  ``country``  ``age``
    ===========  ===========  =======
    XYZZY        U.S.         25
    FOO          China        24
    BAR          China        23
    ===========  ===========  =======

    Game table ``games``:

    ===========  =========  ==============  ==================
    ``game_id``  ``title``  ``is_sandbox``  ``is_multiplayer``
    ===========  =========  ==============  ==================
    1            Minecraft  True            True
    2            Tetris 99  False           True
    ===========  =========  ==============  ==================

    Play relationship table ``plays``:

    ===========  ===========  =========
    ``user_id``  ``game_id``  ``hours``
    ===========  ===========  =========
    XYZZY        1            24
    FOO          1            20
    FOO          2            16
    BAR          2            28
    ===========  ===========  =========

    One could then create a bidirectional bipartite graph as follows:
    >>> builder = PandasGraphBuilder()
    >>> builder.add_entities(users, 'user_id', 'user')
    >>> builder.add_entities(games, 'game_id', 'game')
    >>> builder.add_binary_relations(plays, 'user_id', 'game_id', 'plays')
    >>> builder.add_binary_relations(plays, 'game_id', 'user_id', 'played-by')
    >>> g = builder.build()
    >>> g.num_nodes('user')
    3
    >>> g.num_edges('plays')
    4
    """

    def __init__(self):
        self.entity_tables = {}       # {节点类型：对应的原始df(含特征)}          是原始的df
        self.relation_tables = {}     # {边类型名：click_df（每个交互一条记录）}   是原始的df

        self.entity_pk_to_name = ( {}) # {df主键列名:节点类型名}
        self.entity_pk = {}            # 与上一个相反， {节点类型名：df主键列名}
        self.entity_key_map = ({})     # {节点类型名：df主键的value}
        self.num_nodes_per_type = {}   # {节点类型名：该类型节点数目}
        self.edges_per_relation = {}   # {边类型全名：（src_nodes的编码U，dst_nodes的编码V）}. 用来建边
        self.relation_name_to_etype = {}     # {边类型名：边类型全名}
        self.relation_src_key = {}           # {边类型名:源节点df主键}
        self.relation_dst_key = ( {})        # {边类型名:目标节点df主键}

    """
    把user,item node加入图中
    entity_table： user/item df。每个entity一条记录
    primary_key：df中的主键。如'user_id'
    name:  该类型节点的名称。之后用在dgl图里，作为该类节点的名称. 如'user'
    """
    def add_entities(self, entity_table, primary_key, name):
        entities = entity_table[primary_key].astype("category")
        if not (entities.value_counts() == 1).all():
            raise ValueError(
                "Different entity with the same primary key detected."
            )
        # preserve the category order in the original entity table
        entities = entities.cat.reorder_categories(
            entity_table[primary_key].values
        )

        # 每种节点类型，类型名到df主键列名间的映射：
        self.entity_pk_to_name[primary_key] = name              # {df主键列名:节点类型名}
        self.entity_pk[name] = primary_key                      # {节点类型名：df主键列名}
        # 每类节点
        self.num_nodes_per_type[name] = entity_table.shape[0]   # 数目
        self.entity_key_map[name] = entities                    # df主键对应的数据本身。所有user_id
        self.entity_tables[name] = entity_table                 # 对应的原始df


    '''
    根据interaction df，补充节点间的边。 u-i点击,作为一个边。 i-u被点击也构建边，相当于无向图。
    # relation_table：交互关系df                  （u，i,t,w）
    # source_key：边对应的源node的df主键
    # destination_key: 边对应的目标node的df主键
    # name: 该边的类型名称。如'watched'
    '''
    def add_binary_relations( self, relation_table, source_key, destination_key, name):
        src = relation_table[source_key].astype("category")         # 源节点，df中对应列的值
        src = src.cat.set_categories(
            self.entity_key_map[
                self.entity_pk_to_name[source_key]  # 节点类型
            ].cat.categories
        )
        dst = relation_table[destination_key].astype("category")  # 目标节点，df中对应列的值
        dst = dst.cat.set_categories(
            self.entity_key_map[
                self.entity_pk_to_name[destination_key]
            ].cat.categories
        )
        if src.isnull().any():
            raise ValueError(
                "Some source entities in relation %s do not exist in entity %s."
                % (name, source_key)
            )
        if dst.isnull().any():
            raise ValueError(
                "Some destination entities in relation %s do not exist in entity %s."
                % (name, destination_key)
            )

        srctype = self.entity_pk_to_name[source_key]            # 源节点类型名   （在dgl中）
        dsttype = self.entity_pk_to_name[destination_key]       # 目标节点类型名  （在dgl中）
        etype = (srctype, name, dsttype)                        # 边类型：（源节点类型名，边类型名，目标节点类型名） 。作为dgl图中的边类型
        self.relation_name_to_etype[name] = etype               # {边类型名：边类型全名}
        self.edges_per_relation[etype] = (                      # {边类型全名：（src_nodes的编码，dst_nodes的编码）}. 用来建边
            src.cat.codes.values.astype("int64"),
            dst.cat.codes.values.astype("int64"),
        )
        self.relation_tables[name] = relation_table             # {边类型名：click_df（每个交互一条记录）}
        self.relation_src_key[name] = source_key                # {边类型名:源节点df主键}
        self.relation_dst_key[name] = destination_key           # {边类型名:目标节点df主键}

    def build(self):
        # Create heterograph
        '''
            heads=[0,1,2,3]   # 边左边的node id   默认从0开始。相当于该类型节点共4个节点
            tails=[4,5,6,7]   # 边右边的node id。 默认从0开始。相当于该类型共8个节点 （从0-max_id）.但该类型有4个节点(0-3)没有边
            edges[edge_type]=(heads,tails)
            data_dict=edges        # 可以指定n类型的边。每类型边，指定对应UV list：(U,V)
        '''
        graph = dgl.heterograph(
            # 构建异质图。这里指定了2类型边（正反）。
            # 对ui类型的边, U是源node_id，V是目标node_id
            # 这里U是user_df对应的cat.codes。相当于把userid的labelencoder，作为该类节点的node_id。 默认从0开始
            #     V是item_df对应的cat.codes。相当于把itemid的labelencoder，作为该类节点的node_id。 默认从0开始
            data_dict=self.edges_per_relation, num_nodes_dict=self.num_nodes_per_type   # 也指定了每类型节点的数目
        )
        return graph
