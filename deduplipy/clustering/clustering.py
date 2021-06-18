import re
from typing import List, Dict

import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

from deduplipy.config import DEDUPLICATION_ID_NAME, ROW_ID


def hierarchical_clustering(scored_pairs_table: pd.DataFrame, col_names: List,
                            cluster_threshold: float = 0.5, inspect: Dict = None) -> pd.DataFrame:
    """
    Apply hierarchical clustering to scored_pairs_table and perform the actual deduplication by adding a cluster id to
    each record

    Args:
        scored_pairs_table: Pandas dataframe containg all pairs and the similarity probability score
        col_names: name to use for deduplication
        cluster_threshold: threshold to apply in hierarchical clustering

    Returns:
        Pandas dataframe containing records with cluster id

    """
    if inspect:
        inspect_query = list(inspect.values())[0]
        inspect_regex = re.compile(inspect_query)
        inspect_col_name = list(inspect.keys())[0]
    graph = nx.Graph()
    for j, row in scored_pairs_table.iterrows():
        graph.add_node(row[f'{ROW_ID}_1'], **{col: row[f'{col}_1'] for col in col_names})
        graph.add_node(row[f'{ROW_ID}_2'], **{col: row[f'{col}_2'] for col in col_names})
        graph.add_edge(row[f'{ROW_ID}_1'], row[f'{ROW_ID}_2'], score=row['score'])

    components = nx.connected_components(graph)

    clustering = {}
    cluster_counter = 0
    plot_nr = 0
    for component in components:
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:
            adjacency = nx.to_numpy_matrix(subgraph, weight='score')
            distances = (np.ones_like(adjacency) - np.eye(len(adjacency))) - adjacency
            condensed_distance = ssd.squareform(distances)
            linkage = hierarchy.linkage(condensed_distance, method='centroid')
            clusters = hierarchy.fcluster(linkage, t=1 - cluster_threshold, criterion='distance')
            plot_nr = 0
            if inspect:
                inspect_strings = list(nx.get_node_attributes(subgraph, inspect_col_name).values())
                if any([inspect_regex.match(x) for x in inspect_strings]):
                    plot_nr += 1
                    fig, ax = plt.subplots(figsize=(10, 5))
                    pos = nx.circular_layout(subgraph)
                    nx.draw_networkx_nodes(subgraph, pos, cmap=plt.cm.Accent, node_size=700, ax=ax)

                    # node labels
                    name_dict = nx.get_node_attributes(subgraph, inspect_col_name)
                    labels_formatted = [format_string(name_dict[x]) for x in list(subgraph.nodes)]
                    labels = dict(zip(subgraph.nodes, labels_formatted))
                    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=10, ax=ax)

                    # edges and edge labels
                    edge_labels = nx.get_edge_attributes(subgraph, 'score')
                    # edges
                    scores = list(edge_labels.values())
                    edge_colors = np.digitize(scores, np.arange(0., 1.01, 0.2))
                    nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges, edge_color=edge_colors,
                                           edge_cmap=plt.cm.Blues,
                                           edge_vmin=0.,
                                           alpha=0.8, width=2, ax=ax)
                    edge_labels_filtered = dict()
                    for key, value in edge_labels.items():
                        edge_labels_filtered.update({key: str(round(value, 2))})
                    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels_filtered, font_color='grey',
                                                 ax=ax)

                    # adds space around graph such that labels are fully shown
                    l, r = plt.xlim()
                    b, t = plt.ylim()
                    border_size = 0.5
                    plt.xlim(l - border_size, r + border_size)
                    plt.ylim(b - border_size, t + border_size)
                    plt.axis('off')
                    plt.savefig(f'network_{inspect_query}_{plot_nr}.png')

                    fig = plt.figure()
                    hierarchy.dendrogram(linkage, color_threshold=1-cluster_threshold, labels=list(name_dict.values()),
                                         leaf_font_size=6)
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    plt.savefig(f'dendrogram_{inspect_query}_{plot_nr}.png')
        else:
            clusters = np.array([1])
        clustering.update(dict(zip(subgraph.nodes(), clusters + cluster_counter)))
        cluster_counter += len(component)
    df_clusters = pd.DataFrame.from_dict(clustering, orient='index', columns=[DEDUPLICATION_ID_NAME])
    df_clusters.sort_values(DEDUPLICATION_ID_NAME, inplace=True)
    df_clusters[ROW_ID] = df_clusters.index
    return df_clusters


def format_string(input_string, n=2):
    """
    Formats a string such that after `n` words a new line is started

    Args:
        input_string: string to be reformatted
        n: max number of words on a single line

    Returns: reformatted string

    """
    words = input_string.split()
    first_words = words[:n]
    last_words = words[n:]
    if len(last_words):
        return " ".join(first_words) + "\n" + format_string(" ".join(last_words), n)
    else:
        return " ".join(first_words)
