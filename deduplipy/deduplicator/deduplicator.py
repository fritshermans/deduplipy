from deduplipy.clustering.clustering import hierarchical_clustering
from deduplipy.active_learning.active_learning import ActiveStringMatchLearner
from deduplipy.blocking.blocking import Blocking


class Deduplicator:
    def __init__(self, col_name, n_queries, rules=None, cache_tables=False):
        self.col_name = col_name
        self.n_queries = n_queries
        self.rules = rules
        self.cache_tables = cache_tables
        self.myActiveLearner = ActiveStringMatchLearner(n_queries=self.n_queries, col=self.col_name)
        self.myBlocker = Blocking(self.col_name, rules, cache_tables=self.cache_tables)

    def fit(self, X):
        self.myActiveLearner.fit(X)
        print('active learning finished')
        self.myBlocker.fit(self.myActiveLearner.learner.X_training, self.myActiveLearner.learner.y_training)
        print('blocking rules found')
        print(self.myBlocker.rules_selected)
        return self

    def predict(self, X, score_threshold=0.1):
        print('blocking started')
        scored_pairs_table = self.myBlocker.transform(X)
        print('blocking finished')
        print(f'Nr of pairs: {len(scored_pairs_table)}')
        print('scoring started')
        scored_pairs_table['score'] = self.myActiveLearner.predict_proba(
            scored_pairs_table[[f'{self.col_name}_1', f'{self.col_name}_2']].values)[:, 1]
        scored_pairs_table.loc[
            scored_pairs_table[f'{self.col_name}_1'] == scored_pairs_table[f'{self.col_name}_2'], 'score'] = 1
        print("scoring finished")
        scored_pairs_table = scored_pairs_table[scored_pairs_table['score']>=score_threshold]
        print(f'Nr of filtered pairs: {len(scored_pairs_table)}')
        if self.cache_tables:
            scored_pairs_table.to_csv('scored_pairs_table.csv')
        print('Clustering started')
        df_clusters = hierarchical_clustering(scored_pairs_table, col_name=self.col_name)
        print('Clustering finished')
        return df_clusters
