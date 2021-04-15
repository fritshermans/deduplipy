from deduplipy.clustering.clustering import hierarchical_clustering
from deduplipy.active_learning.active_learning import ActiveStringMatchLearner
from deduplipy.blocking.blocking import Blocking


class Deduplicator:
    def __init__(self, col_name, n_queries, rules=None):
        self.col_name = col_name
        self.n_queries = n_queries
        self.rules = rules
        self.myActiveLearner = ActiveStringMatchLearner(n_queries=self.n_queries, col=self.col_name)
        self.myBlocker = Blocking(self.col_name, rules)

    def fit(self, X):
        self.myActiveLearner.fit(X)
        self.myBlocker.fit(self.myActiveLearner.learner.X_training, self.myActiveLearner.learner.y_training)
        return self

    def predict(self, X):
        scored_pairs_table = self.myBlocker.transform(X[:, 0])
        scored_pairs_table['score'] = self.myActiveLearner.predict_proba(
            scored_pairs_table[[f'{self.col_name}_1', f'{self.col_name}_2']].values)[:, 1]
        scored_pairs_table.loc[scored_pairs_table.col_1 == scored_pairs_table.col_2, 'score'] = 1
        df_clusters = hierarchical_clustering(scored_pairs_table)
        return df_clusters
