import os
from itertools import product
from pkg_resources import resource_filename

import pandas as pd
from sklearn.model_selection import train_test_split


def load_hotel_rooms(return_pairs=False):
    file_path = resource_filename('deduplipy', os.path.join('data', 'room_type.csv'))
    df = pd.read_csv(file_path).rename(columns={'Expedia': 'expedia', 'Booking.com': 'booking'})
    expedia = df.expedia.unique()
    booking = df.booking.unique()
    if return_pairs:
        df['match'] = 1
        df_all = pd.DataFrame(list(product(expedia, booking)), columns=['expedia', 'booking'])
        df = df.merge(df_all, on=['expedia', 'booking'], how='outer')
        df.fillna({'match': 0}, inplace=True)
        for col in ['expedia', 'booking']:
            df[col] = df[col].str.lower()
        X, y = df[['expedia', 'booking']].values, df['match'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        df = pd.concat((pd.DataFrame(expedia), pd.DataFrame(booking)))
        df.columns = ['room_description']
        df.reset_index(inplace=True, drop=True)
        return df


def load_stoxx50(return_pairs=False):
    file_path = resource_filename('deduplipy', os.path.join('data', 'stoxx50_extended_with_id.xlsx'))
    df = pd.read_excel(file_path, engine='openpyxl')
    if return_pairs:
        l = list(product(df.values.tolist(), df.values.tolist()))
        df_all = pd.DataFrame(list(map(lambda x: sum(x, []), l)),
                              columns=['name_address_1', 'id_1', 'name_address_2', 'id_2'])
        df_all['match'] = (df_all.id_1 == df_all.id_2).astype(int)

        df_train = df_all[(df_all.id_1 < 30) & (df_all.id_2) < 30]
        df_test = df_all[(df_all.id_1 >= 30) & (df_all.id_2 >= 30)]

        df_train = pd.concat((df_train[df_train.match == 1].sample(n=df_train.match.sum()),
                              df_train[df_train.match == 0].sample(n=df_train.match.sum())), axis=0)
        df_test = pd.concat((df_test[df_test.match == 1].sample(n=df_test.match.sum()),
                             df_test[df_test.match == 0].sample(n=df_test.match.sum())), axis=0)

        X_train, y_train = df_train[['name_address_1', 'name_address_2']].values, df_train['match'].values
        X_test, y_test = df_test[['name_address_1', 'name_address_2']].values, df_test['match'].values
        return X_train, X_test, y_train, y_test
    else:
        return df[['name']]


def load_chicago_childcare(return_pairs=False):
    file_path = resource_filename('deduplipy', os.path.join('data', 'csv_example_input_with_true_ids.csv'))
    df = pd.read_csv(file_path)
    df['name_address'] = df['Site name'] + " " + df['Address']
    df['name_address'] = (df['name_address'].str.lower()
                          .str.replace(',', ' ', regex=True)
                          .str.replace('\n', ' ', regex=True)
                          .str.replace('.', ' ', regex=True)
                          .str.replace('"', ' ', regex=True)
                          .str.replace(r'\s+', ' ', regex=True)
                          .str.strip()
                          )
    df = df.drop_duplicates(subset=['name_address'])
    if return_pairs:
        size = 10_000

        df = df[['True Id', 'name_address']]

        df_no_match = pd.DataFrame(list(
            product(df.iloc[:100][['True Id', 'name_address']].values,
                    df.iloc[:100][['True Id', 'name_address']].values)))
        df_no_match = pd.concat(
            (df_no_match, df_no_match[0].apply(pd.Series).rename(columns={0: 'True Id 1', 1: 'name_address_1'})),
            axis=1)
        df_no_match = pd.concat(
            (df_no_match, df_no_match[1].apply(pd.Series).rename(columns={0: 'True Id 2', 1: 'name_address_2'})),
            axis=1)
        df_no_match.drop(columns=[0, 1], inplace=True)
        df_no_match['match'] = df_no_match.apply(lambda row: int(row['True Id 1'] == row['True Id 2']), axis=1)
        df_no_match = df_no_match[['name_address_1', 'name_address_2', 'match']]

        df_match = df.merge(df, on='True Id', suffixes=('_1', '_2'))
        df_match = df_match[df_match.name_address_1 != df_match.name_address_2]
        df_match.drop(columns=['True Id'], inplace=True)
        df_match['match'] = 1

        df_equal = pd.concat((df_match.sample(n=size // 2), df_no_match[df_no_match.match == 0].sample(n=size // 2)),
                             axis=0)

        for col in ['name_address_1', 'name_address_2']:
            df_equal[col] = df_equal[col].str.lower()

        X, y = df_equal[['name_address_1', 'name_address_2']].values, df_equal['match'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        return df[['name_address']]


def load_data(kind='stoxx50', return_pairs=False):
    """
    Load data for experimentation. `kind` can be 'stoxx50', 'hotel_rooms' or 'childcare'. Function can return pairs
    or just a list of records.

    Args:
        kind: 'stoxx50', 'hotel_rooms' or 'childcare'
        return_pairs: boolean indicating whether to return pairs or list of records

    Returns:
        Pandas dataframe containing experimentation dataset
    """
    if kind == 'stoxx50':
        return load_stoxx50(return_pairs)
    elif kind == 'hotel_rooms':
        return load_hotel_rooms(return_pairs)
    elif kind == 'childcare':
        return load_chicago_childcare(return_pairs)