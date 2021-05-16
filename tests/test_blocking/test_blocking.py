import pandas as pd

from deduplipy.blocking import Blocking, all_rules

df = pd.DataFrame({'name': ['frits', 'frits h', 'frank', 'frank h', 'stan', 'stijn', 'ahmet', 'fred', 'frederik'],
                   'row_number': [0, 1, 2, 3, 4, 5, 6, 7, 8]})

df_pairs = pd.DataFrame({'name_1': ['frits', 'frank', 'stan', 'ahmet', 'fred'],
                         'name_2': ['frits h', 'frank h', 'stijn', 'ahmet', 'frederik']})

y = pd.DataFrame({'match': [1, 1, 0, 1, 1]})

myBlocker = Blocking(col_names=['name'], rules_info={'name': all_rules})
myBlocker.fit(df_pairs, y)


def test_base_case():
    result = myBlocker.transform(df)
    expected = pd.DataFrame({'name_1': ['frits', 'frank', 'fred'],
                             'row_number_1': [0, 2, 7],
                             'fingerprint': ['fri:0', 'fra:0', 'fre:0'],
                             'name_2': ['frits h', 'frank h', 'frederik'],
                             'row_number_2': [1, 3, 8]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


def test__fingerprint():
    result = myBlocker._fingerprint(df)
    assert result.columns.to_list() == ['name', 'row_number', 'fingerprint']
    assert ['frits', 0, 'fri:0'] in result.values
    assert ['stan', 4, 'sta:4'] in result.values
    assert ['ahmet', 6, 'ahme:1'] in result.values


def test__create_pairs_table():
    X_fingerprinted = myBlocker._fingerprint(df)
    result = myBlocker._create_pairs_table(X_fingerprinted)
    assert result.columns.to_list() == ['name_1', 'row_number_1', 'fingerprint', 'name_2', 'row_number_2']
    assert ['frits', 0, 'fri:0', 'frits h', 1] in result.values
    assert ['frank', 2, 'fra:0', 'frank h', 3] in result.values
