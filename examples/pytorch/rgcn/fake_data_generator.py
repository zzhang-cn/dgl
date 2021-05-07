"""
This file generates fake data, and then build a graph based on the data

The data contains 5 tables: df, df_seller, df_asin, df_offer, seller_relation_df
(1) df: seller id(merchant_customer_id), asin id(asin), offer listing's label, offer features(of), seller features(sf),
and asin features(af). An asin is a product. An offer listing is a seller-asin pair, which can be uniquely identified by
a pair of seller and asin ids.
ex:
merchant_customer_id	asin	label	of1	        of2	        of3	        of4	        sf1	        sf2	        sf3	        af1	        af2
                s0	    a0	    1	    0.260922	0.407622	0.118446	0.510906	0.158223	0.497794	0.065068	0.350285	0.392781
                s0	    a1	    0	    0.412472	0.781531	0.37534	    0.759486	0.158223	0.497794	0.065068	0.189026	0.61893
                s1	    a2	    0	    0.419034	0.354146	0.029774	0.975905	0.545203	0.182445	0.540075	0.472621	0.436766
                s1	    a0	    1	    0.902422	0.925809	0.537782	0.7365	    0.545203	0.182445	0.540075	0.350285	0.392781
                s2	    a1	    0	    0.979616	0.720452	0.945828	0.953956	0.524404	0.918383	0.129187	0.189026	0.61893

(2) df_seller: seller id(merchant_customer_id) and seller features.
ex:
    merchant_customer_id       sf1       sf2       sf3
0                    s0  0.158223  0.497794  0.065068
1                    s1  0.545203  0.182445  0.540075
2                    s2  0.524404  0.918383  0.129187

(3) df_asin: asin id and asin features
ex:
   asin       af1       af2
0   a0  0.350285  0.392781
1   a1  0.189026  0.618930
2   a2  0.472621  0.436766

(4) df_offer: offer label and features
ex:
    merchant_customer_id asin  label       of1       of2       of3       of4
0                    s0   a0      1  0.284056  0.901859  0.825885  0.237311
1                    s0   a1      0  0.208286  0.499683  0.416775  0.236482
2                    s1   a2      0  0.698869  0.904695  0.287311  0.104970
3                    s1   a0      1  0.746850  0.960512  0.753721  0.083001

(5) seller_relation_df: sellers relations
ex:
   from_seller to_seller edge_type
0          s0        s1  bank account
1          s2        s3  email
2          s4        s5  bank account
3          s6        s7  email
4          s8        s9  bank account

"""

import pandas as pd
import numpy as np
import torch as th
import dgl

class CreateFakeData:
    def __init__(self):

        self.N = 20
        self.seller_col = "merchant_customer_id"  # seller id
        self.asin_col = "asin"  # asin id
        self.label_col = "label"
        self.label_edge = "seller-asin"

        self.seller_feature_cols = ["sf1", "sf2", "sf3"]
        self.asin_feature_cols = ["af1", "af2"]
        self.offer_feature_cols = ["of1", "of2", "of3", "of4"]

    def fake_data(self):

        # seller asin -> offer
        df = pd.DataFrame()
        df[self.seller_col] = [("s%d" % i) for i in range(self.N) for j in range(2)]
        df[self.asin_col] = [("a%d" % (i % 3)) for i in range(2 * self.N)]
        df[self.label_col] = [(1 if i % 3 == 0 else 0) for i in range(2 * self.N)]

        # seller feature
        df_seller = pd.DataFrame()
        df_seller[self.seller_col] = df[self.seller_col].unique()
        for f in self.seller_feature_cols:
            df_seller[f] = np.random.rand(len(df_seller.index))

        # asin feature
        df_asin = pd.DataFrame()
        df_asin[self.asin_col] = df[self.asin_col].unique()
        for f in self.asin_feature_cols:
            df_asin[f] = np.random.rand(len(df_asin.index))

        # offer feature
        for f in self.offer_feature_cols:
            df[f] = np.random.rand(len(df.index))

        df = df.merge(df_seller, how='left', on=self.seller_col)
        df = df.merge(df_asin, how='left', on=self.asin_col)

        df_offer = df[[self.seller_col, self.asin_col, self.label_col] + self.offer_feature_cols]

        return df, df_seller, df_asin, df_offer

    # seller relation
    def seller_relation_fake_df(self):
        sr = pd.DataFrame()
        sr["from_seller"] = [("s%d" % i) for i in range(self.N) if i % 2 == 0]
        sr["to_seller"] = [("s%d" % i) for i in range(self.N) if i % 2 != 0]
        sr["edge_type"] = ["email" if i % 2 != 0 else "bank account" for i in range(self.N // 2)]
        return sr


class IDs:
    def __init__(self, df, col):
        self.col = col
        self.id_map = pd.DataFrame()
        self.id_map[self.col] = df[self.col].unique()
        self.id_map[self.col + '_id'] = np.arange(len(self.id_map[self.col]))

        self.id_dict = dict(zip(self.id_map[self.col], self.id_map[self.col + '_id']))

    def get(self):
        return self.id_map

    # convert id to seller name
    def inverse_id_map(self, ids):
        df = pd.DataFrame()
        df[self.col + '_id'] = ids
        df = df.merge(self.id_map, on=self.col + '_id', how='left')
        return df[self.col]

    # convert seller name to id
    def forward_id_map(self, sellers):
        df = pd.DataFrame()
        df[self.col] = sellers
        df = df.merge(self.id_map, on=self.col, how='left')
        return df[self.col + '_id']


class BuildGraph:
    def __init__(self, df, df_seller, df_asin, df_offer, seller_relation_df):
        self.seller_cfeature_cols = ["email", "bank account"]
        self.seller_col = "merchant_customer_id"
        self.asin_col = "asin"
        self.label = "label"
        self.label_edge = "seller-asin"

        self.seller_feature_cols = [var for var in df_seller.columns if var != self.seller_col]
        self.asin_feature_cols = [var for var in df_asin.columns if var != self.asin_col]
        self.offer_feature_cols = [var for var in df_offer.columns if
                                   var not in [self.seller_col, self.asin_col, self.label]]

        print("num of seller feats : {}".format(len(self.seller_feature_cols)))
        print("num of asin feats : {}".format(len(self.asin_feature_cols)))
        print("num of offer feats : {}".format(len(self.offer_feature_cols)))

        print("seller feature: {} ".format(self.seller_feature_cols))
        print("asin feature: {} ".format(self.asin_feature_cols))
        print("asin feature: {} ".format(self.offer_feature_cols))

        self.df = df
        self.df_seller = df_seller
        self.df_asin = df_asin
        self.df_offer = df_offer
        self.seller_relation_df = seller_relation_df

        self.seller_id = IDs(df, self.seller_col)
        self.asin_id = IDs(df, self.asin_col)

    def build_edges(self):
        edges = {}

        print("construct edge from seller relation df")
        for i, edge_type in enumerate(self.seller_cfeature_cols):
            seller_relation_df = self.seller_relation_df.loc[self.seller_relation_df["edge_type"] == edge_type]
            src = self.seller_id.forward_id_map(seller_relation_df["from_seller"])
            dst = self.seller_id.forward_id_map(seller_relation_df["to_seller"])
            assert (len(src) == len(dst))
            edges[(self.seller_col, edge_type, self.seller_col)] = (src, dst)

        print("construct seller-asin and its reverse edges from df")
        seller_ids = self.seller_id.forward_id_map(
            self.df[self.seller_col])  # output: series. df contains all seller-asin relations
        asin_ids = self.asin_id.forward_id_map(self.df[self.asin_col])

        edges[(self.seller_col, self.label_edge, self.asin_col)] = (seller_ids, asin_ids)
        edges[(self.asin_col, 'rev_' + self.label_edge, self.seller_col)] = (asin_ids, seller_ids)

        print("{} : {}".format(self.label_edge, len(seller_ids)))
        print("{} : {}".format('rev_' + self.label_edge, len(seller_ids)))

        return edges


    def build_hetero_graph(self, edges):
        hg = dgl.heterograph(edges)

        print(hg)
        return hg


    def add_feats_to_heterograph(self, hg):
        seller_feat = self.seller_id.get().copy()
        seller_feat = seller_feat.merge(self.df_seller, on=self.seller_col, how='left')
        hg.nodes[self.seller_col].data["feat"] = th.tensor(seller_feat[self.seller_feature_cols].values,
                                                           dtype=th.float32)
        asin_feat = self.asin_id.get().copy()
        asin_feat = asin_feat.merge(self.df_asin, on=self.asin_col, how='left')
        hg.nodes[self.asin_col].data["feat"] = th.tensor(asin_feat[self.asin_feature_cols].values, dtype=th.float32)

        # add edge features "label" for each edge type
        for et in hg.etypes:
            hg.edges[et].data["label"] = -1 * th.ones(hg.number_of_edges(et), dtype=th.int32)

        offer_feat = self.df[[self.seller_col, self.asin_col]]
        offer_feat = offer_feat.merge(self.df_offer, on=[self.seller_col, self.asin_col], how='left')
        hg.edges[self.label_edge].data["label"] = th.tensor(offer_feat[self.label].values, dtype=th.int32)

        return hg

    def get_node_features(self, hg):
        node_features = []
        for ntype in hg.ntypes:
            if len(hg.nodes[ntype].data) == 0:
                node_features.append(hg.number_of_nodes(ntype))
            else:
                assert len(hg.nodes[ntype].data) == 1
                feat = hg.nodes[ntype].data.pop('feat')
                node_features.append(feat.share_memory_())

        return node_features

    def get_edge_features(self):
        offer_feat = self.df[
            [self.seller_col, self.asin_col]]  # the order of offer_feat table should be the same as df
        offer_feat = offer_feat.merge(self.df_offer, on=[self.seller_col, self.asin_col], how='left')
        offer_feat = offer_feat[self.offer_feature_cols].values
        offer_feat = th.from_numpy(offer_feat)

        return offer_feat


def load_generated_data():
    data = CreateFakeData()
    df, df_seller, df_asin, df_offer = data.fake_data()
    seller_relation_df = data.seller_relation_fake_df()

    print("start building graph")
    graph_builder = BuildGraph(df, df_seller, df_asin, df_offer, seller_relation_df)
    edges = graph_builder.build_edges()
    hg = graph_builder.build_hetero_graph(edges)
    hg = graph_builder.add_feats_to_heterograph(hg)
    node_features = graph_builder.get_node_features(hg)
    edge_features = graph_builder.get_edge_features()

    return hg, node_features, edge_features, data.label_edge