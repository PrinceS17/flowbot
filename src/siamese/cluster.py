import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
import torch.linalg as LA

from .chinese_whispers import chinese_whispers, aggregate_clusters
# from .preprocess import measure_time


def do_cluster(sample, max_iter, label_key, last_labels_list, th, n_voter, weighting):
    """Do the cluster.
        sample: [n_flow, n_feature].
        last_labels_list: [n_flow] of the last labels.
    """
    G = nx.Graph()
    # sample = sample.detach().numpy()     # this doesn't improve speed
    assert type(sample) == np.ndarray, f"type(sample) = {type(sample)}"
    for i, v in enumerate(sample):
        G.add_node(i, tensor=v)
        G.nodes[i][label_key] = i + 1 if not last_labels_list else last_labels_list[i]

    for u in G:
        for v in G:
            if v <= u:
                continue
            # dist = max(1e-4, np.linalg.norm(G.nodes[u]['tensor'] -
            #                                 G.nodes[v]['tensor']))
            # 0.4 to 0.3s for the whole loop
            dist = max(1e-4,
                np.sqrt(sum((G.nodes[u]['tensor'] - G.nodes[v]['tensor']) ** 2)))
            w = 1 / dist
            if w < 1.0 / th and n_voter <= 1:   # Chinese Whispers or distance-based
                continue
            G.add_edge(u, v, weight=w)
    # the th in chinese whispers is the weight threshold
    chinese_whispers(G, weighting=weighting, iterations=max_iter,
                        n_voter=n_voter, th=1.0 / th, label_key=label_key)
    clusters = aggregate_clusters(G)
    # res = torch.zeros(len(sample), dtype=torch.long)
    res = np.zeros(len(sample))
    for k, v in clusters.items():
        for node in v:
            res[node] = k
    return res, clusters


# @measure_time()
def cluster(vectors, max_iter=100, n_voter=0, weighting='top', th=0.6,
            last_labels=None, label_key='label', flow_labels=None):
    """Cluster the given vectors using Chinese Whispers.
    In the algorthms we used (Chinese Whispers, distance-based or n-based voting),
    only partial edges are needed for each nodes, i.e. those with weight >= th
    for Chinese Whispers and distanced-based voting, and n for n-based voting.
    In the current implementation, we only use Chinese Whispers and distance-based
    voting, so we don't add the edges with weight < th for them.

    flow_labels is a dict used to support clustering w/ non-btnk flows marked before.
    If flow_labels are given, cluster() should
        1, get the btnk flow to cluster, i.e. those with flow_labels[flow] == -1,
        2, return the cluster results, and update both flow_labels & y_hat.

    Args:
        vectors (Tensor): [n_flow, n_feature] or [batch_size, n_flow, n_feature]
                        if flow_labels is given, vectors include all flows.
        max_iter (int, optional): Defaults to 100.
        n_voter (int, optional): Number of voters / neighbors, 1 for
            Chinese whisper, 0 for distance based. Defaults to 0.
        weighting (str, optional): Weighting function, 'top' for 2-norm,
            'inv_sq' for inverse sum of square distances, as in ToN'20.
            Defaults to 'top'.
        th (float, optional): Distance threshold for clustering.
            Defaults to 1.0.
        last_labels (list, optional): [n_flow] of the cluster No. for the flows.
        last_labels (dict, optional): last flow labels for the init this time.
        label_key (str, optional): Defaults to 'label'.
        flow_labels (dict, optional): {flow: label}, -1 for btnk flows to be assigned.

    Returns:
        Tensor: labels, [n_flow] of the cluster No. for each flow, or [batch_size, n_flow]
        list: flow labels, {flow: label} with non-btnk included
        list: clusters, i.e. cluster No.
    """
    if len(vectors.shape) == 2:         # [n_flow, n_feature] branch
        last_labels_list = None
        if flow_labels is not None:
            btnk_flows = sorted([flow for flow, label in flow_labels.items()
                                if label == -1])
            # find the indices of btnk flow in the sorted flows for vector indexing
            i_btnk_flows = [i for i, flow in enumerate(sorted(flow_labels.keys()))
                            if flow_labels[flow] == -1]
            vectors = vectors[i_btnk_flows]
            if last_labels is not None:
                last_labels_list = [last_labels.setdefault(flow, flow)
                                    for flow in btnk_flows]
        elif last_labels is not None:
            last_labels_list = [last_labels[flow] for flow in sorted(last_labels.keys())]
        

        flow_labels1 = flow_labels.copy()
        btnk_labels, clusters = do_cluster(vectors, max_iter, label_key,
                                           last_labels_list, th, n_voter, weighting)
        if flow_labels is not None:
            # merge btnk labels back & return all labels
            for flow, label in zip(btnk_flows, btnk_labels):
                flow_labels1[flow] = label
            labels = [flow_labels1[flow] for flow in sorted(flow_labels1.keys())]
        else:
            labels = btnk_labels
        return np.array(labels), flow_labels1, clusters

    # branch for batched vectors, unlikely to happen, as the caller clusters the
    # flows iteratively in current implementation.
    assert flow_labels is None, "flow_labels is not supported for batched vectors"
    labels, clusters = [], []
    for sample in vectors:
        label, cluster = do_cluster(sample, max_iter, label_key, last_labels,
                                    th, n_voter, weighting)
        labels.append(label)
        clusters.append(cluster)
    return torch.stack(labels), flow_labels, clusters


def cluster_col(df, col, grp_col, p, is_relative=False):
    # is_relative: whether to use relative threshold
    df = df.sort_values(col).reset_index(drop=True)
    if grp_col not in df.columns:
        df[grp_col] = 0
    prev = df.iloc[0][col]
    for i in range(1, len(df)):
        th = p * df.loc[i, col] if is_relative else p
        if df.loc[i, col] - prev <= th:
            df.loc[i, grp_col] = df.loc[i-1, grp_col]
            continue
        df.loc[i, grp_col] = df.loc[i-1, grp_col] + 1
        prev = df.loc[i, col]
    return df

# print(cluster_col(a, 'skew_est', 'g1', 0.1))


def rmcat(df, cols, ths, relatives):
    # df: [flow, skew_est, var_est, freq_est, pkt_loss]
    grp_cols = [f'g{i}' for i in range(len(df.columns)-1)]
    for col, grp_col, th, relative in zip(cols, grp_cols, ths, relatives):
        df = cluster_col(df, col, grp_col, th, relative)
    base = df.flow.nunique()
    df['group'] = df.apply(
        lambda x: sum([x[grp_cols[i]] * base ** i for i in range(len(grp_cols))]),
        axis=1
    )
    l = df.group.unique().tolist()
    df['group'] = df['group'].apply(lambda x: l.index(x))
    return df.sort_values(['flow', 'group']).reset_index(drop=True)
