from setuptools import setup, find_packages

base_packages = [
    'pandas',
    'numpy',
    'scipy',
    'scikit-learn',
    'networkx',
    'python-Levenshtein',
    'fuzzywuzzy',
    'modAL'
]

util_packages = [
    "matplotlib",
    "jupyterlab",
]

dev_packages = base_packages + util_packages

setup(name='DedupliPy',
      version='0.1',
      packages=find_packages('deduplipy', exclude=['notebooks']),
      package_dir={'': 'deduplipy'},
      install_requires=base_packages,
      extras_require={
          "base": base_packages,
          "dev": dev_packages,
      },
      )
