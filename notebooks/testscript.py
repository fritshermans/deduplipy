from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
from fuzzywuzzy.fuzz import ratio, token_set_ratio, token_sort_ratio, partial_ratio

if __name__ == "__main__":
    df_train = load_data(kind='childcare', return_pairs=False)

    myDedupliPy = Deduplicator(
        field_info={'name': [ratio, partial_ratio], 'address': [token_set_ratio, token_sort_ratio]}, interaction=False,
        n_queries=999, cache_tables=True, verbose=1)

    myDedupliPy.fit(df_train)
    res = myDedupliPy.predict(df_train)
    res.to_excel('res.xlsx', index=None)
    print(res)
