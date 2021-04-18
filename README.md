# DedupliPy
Deduplication is the task to combine different representations of the same real world entity. 
This package implements deduplication using active learning.
DedupliPy is an end-to-end solution with advantages over existing solutions:
- active learning; no large manually labelled dataset required
- during active learning, the user gets notified when the model converged and training may stop
- works out of the box, advanced users can choose settings as desired (custom blocking rules, custom metrics)

Developed by [Frits Hermans](https://www.linkedin.com/in/frits-hermans-data-scientist/)

## Installation
Normal installation

```
pip install .
```

Install to contribute 
```
python -m pip install -e ".[dev]"
python setup.py develop
```