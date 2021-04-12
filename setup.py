from setuptools import setup, find_packages

setup(name='DedupliPy',
      version='0.1',
      packages=find_packages('deduplipy'),
      package_dir={'': 'deduplipy'},
      install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn', 'networkx', 'fuzzywuzzy', 'modAL'])
