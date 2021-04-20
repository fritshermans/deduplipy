from setuptools import setup, find_packages

base_packages = [
    'pandas',
    'numpy',
    'scipy',
    'scikit-learn',
    'networkx',
    'python-Levenshtein',
    'fuzzywuzzy',
    'modAL',
    'openpyxl'
]

util_packages = [
    "matplotlib",
    "jupyterlab",
]

docs_packages = [
    "sphinx==3.5.4",
    "nbsphinx",
    'sphinx_rtd_theme'
]

dev_packages = base_packages + util_packages + docs_packages

setup(name='DedupliPy',
      version='0.1',
      packages=find_packages(exclude=['notebooks']),
      install_requires=base_packages,
      extras_require={
          "base": base_packages,
          "dev": dev_packages,
          "docs": docs_packages,
      },
      )
