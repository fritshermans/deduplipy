import os
from pkg_resources import resource_filename

import pandas as pd


def load_stoxx50():
    file_path = resource_filename('deduplipy', os.path.join('data', 'stoxx50_extended_with_id.xlsx'))
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Column names: 'name'")
    return df[['name']]


def load_voters():
    file_path = resource_filename('deduplipy', os.path.join('data', 'voter_names.csv'))
    df = pd.read_csv(file_path)
    print("Column names: 'name', 'suburb', 'postcode'")
    return df


def load_data(kind='voters'):
    """
    Load data for experimentation. `kind` can be 'stoxx50' or 'voters'.

    Args:
        kind: 'stoxx50' or 'voters'

    Returns:
        Pandas dataframe containing experimentation dataset
    """
    if kind == 'stoxx50':
        return load_stoxx50()
    elif kind == 'voters':
        return load_voters()
