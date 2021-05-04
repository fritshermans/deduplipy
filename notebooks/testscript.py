from deduplipy.datasets import load_data
from deduplipy.deduplicator import Deduplicator
from deduplipy.config import DEDUPLICATION_ID_NAME
from fuzzywuzzy.fuzz import ratio, token_set_ratio, token_sort_ratio, partial_ratio

if __name__ == "__main__":
    df_train = load_data(kind='voters')

    myDedupliPy = Deduplicator(['name', 'suburb', 'postcode'], interaction=False, n_queries=999,
                               save_intermediate_steps=True, verbose=1)

    myDedupliPy.fit(df_train)
    res = myDedupliPy.predict(df_train)
    res.sort_values(DEDUPLICATION_ID_NAME).to_excel('res.xlsx', index=None)
    print(res)
