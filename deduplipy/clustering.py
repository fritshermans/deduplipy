import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd


def hierarchical_clustering(scored_pairs_table, cluster_threshold=0.5):
    graph = nx.Graph()
    for j, row in scored_pairs_table.iterrows():
        graph.add_edge(row['col_1'], row['col_2'], weight=row['score'])

    components = nx.connected_components(graph)

    clustering = {}
    cluster_counter = 0
    for component in components:
        subgraph = graph.subgraph(component)
        adjacency = nx.to_numpy_matrix(subgraph)
        condensed_distance = ssd.pdist(adjacency)
        z = linkage(condensed_distance, method='centroid')
        clusters = fcluster(z, t=cluster_threshold, criterion='distance')
        clustering.update(dict(zip(subgraph.nodes(), clusters + cluster_counter)))
        cluster_counter += len(component)
    df_clusters = pd.DataFrame.from_dict(clustering, orient='index', columns=['cluster_id'])
    df_clusters.sort_values('cluster_id', inplace=True)
    return df_clusters
