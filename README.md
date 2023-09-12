<!--- BADGES: START --->
[![Version](https://img.shields.io/pypi/v/deduplipy)](https://pypi.org/project/deduplipy/)
![](https://img.shields.io/github/license/fritshermans/deduplipy)
[![Downloads](https://static.pepy.tech/badge/deduplipy)](https://pepy.tech/project/deduplipy)
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/deduplipy?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/deduplipy?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![Conda Recipe](https://img.shields.io/static/v1?logo=conda-forge&style=flat&color=green&label=recipe&message=deduplipy)][#conda-forge-feedstock]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=readthdocs&style=flat&color=pink&label=docs&message=deduplipy)][#docs-package]

[#pypi-package]: https://pypi.org/project/deduplipy/
[#conda-forge-package]: https://anaconda.org/conda-forge/deduplipy
[#conda-forge-feedstock]: https://github.com/conda-forge/deduplipy-feedstock
[#docs-package]: https://deduplipy.readthedocs.io/en/latest/
<!--- BADGES: END --->

# DedupliPy

<a href="https://deduplipy.readthedocs.io/en/latest/"><img src="https://deduplipy.readthedocs.io/en/latest/_images/logo.png" width="15%" height="15%" align="left" /></a>

Deduplication is the task to combine different representations of the same real world entity. This package implements
deduplication using active learning. Active learning allows for rapid training without having to provide a large,
manually labelled dataset.

DedupliPy is an end-to-end solution with advantages over existing solutions:

- active learning; no large manually labelled dataset required
- during active learning, the user gets notified when the model converged and training may be finished
- works out of the box, advanced users can choose settings as desired (custom blocking rules, custom metrics,
  interaction features)

Developed by [Frits Hermans](https://fritshermans.github.io/)

## Documentation

Documentation can be found [here](https://deduplipy.readthedocs.io/en/latest/)

## Installation

### Normal installation

**With pip**

Install directly from PyPI.

```
pip install deduplipy
```

**With conda**

Install using conda from conda-forge channel.

```
conda install -c conda-forge deduplipy
```


### Install to contribute

Clone this Github repo and install in editable mode:

```
python -m pip install -e ".[dev]"
python setup.py develop
```

## Usage

Apply deduplication your Pandas dataframe `df` as follows:

```python
myDedupliPy = Deduplicator(col_names=['name', 'address'])
myDedupliPy.fit(df)
```

This will start the interactive learning session in which you provide input on whether a pair is a match (y) or not (n).
During active learning you will get the message that training may be finished once algorithm training has converged.
Predictions on (new) data are obtained as follows:

```python
result = myDedupliPy.predict(df)
```
