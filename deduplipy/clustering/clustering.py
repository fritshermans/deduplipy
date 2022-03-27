from typing import List

import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

from deduplipy.config import DEDUPLICATION_ID_NAME, ROW_ID, ROW_ID_CENTRAL
from deduplipy.clustering.fill_missing_edges import fill_missing_links


def find_central_cluster_nodes(clusters, nodes, distances):
    central_node_dict = dict()
    for cluster in set(clusters):
        cluster_nodes = nodes[np.where(clusters == cluster)]
        cluster_distance_matrix = distances[np.where(clusters == cluster)][:, np.where(clusters == cluster)[0]]
        mean_distances = cluster_distance_matrix.mean(axis=1)
        central_cluster_node = cluster_nodes[np.where(mean_distances == mean_distances.min())][0]
        central_node_dict.update({cluster: central_cluster_node})
    central_cluster_nodes = [central_node_dict.get(x) for x in clusters]
    return central_cluster_nodes


def hierarchical_clustering(scored_pairs_table: pd.DataFrame, col_names: List,
                            cluster_threshold: float = 0.5, fill_missing=True) -> pd.DataFrame:
    """
    Apply hierarchical clustering to scored_pairs_table and perform the actual deduplication by adding a cluster id to
    each record

    Args:
        scored_pairs_table: Pandas dataframe containg all pairs and the similarity probability score
        col_names: name to use for deduplication
        cluster_threshold: threshold to apply in hierarchical clustering
        fill_missing: whether to impute missing values in the adjacency matrix using softimpute, otherwise missing
            values in the adjacency matrix are filled with zeros

    Returns:
        Pandas dataframe containing records with cluster id

    """
    graph = nx.Graph()
    for j, row in scored_pairs_table.iterrows():
        graph.add_node(row[f'{ROW_ID}_1'], **{col: row[f'{col}_1'] for col in col_names})
        graph.add_node(row[f'{ROW_ID}_2'], **{col: row[f'{col}_2'] for col in col_names})
        graph.add_edge(row[f'{ROW_ID}_1'], row[f'{ROW_ID}_2'], score=row['score'])

    components = nx.connected_components(graph)

    clustering = pd.DataFrame()
    cluster_counter = 0
    for component in components:
        subgraph = graph.subgraph(component)
        nodes = np.asarray(subgraph.nodes())
        if len(subgraph.nodes) > 1:
            adjacency = nx.to_numpy_array(subgraph, weight='score')
            if fill_missing:
                adjacency = fill_missing_links(adjacency)
            distances = (np.ones_like(adjacency) - np.eye(len(adjacency))) - adjacency
            condensed_distance = ssd.squareform(distances)
            linkage = hierarchy.linkage(condensed_distance, method='centroid')
            clusters = hierarchy.fcluster(linkage, t=1 - cluster_threshold, criterion='distance')
            central_cluster_nodes = find_central_cluster_nodes(clusters, nodes, distances)
        else:
            clusters = np.array([1])
            central_cluster_nodes = nodes
        to_add = pd.DataFrame.from_dict(dict(zip(subgraph.nodes(), clusters + cluster_counter)), orient='index')
        to_add[ROW_ID_CENTRAL] = central_cluster_nodes
        clustering = pd.concat([clustering, to_add])
        cluster_counter += len(component)
    clustering = clustering.reset_index()
    clustering.columns = [ROW_ID, DEDUPLICATION_ID_NAME, ROW_ID_CENTRAL]
    return clustering
