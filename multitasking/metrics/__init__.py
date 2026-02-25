"""Alignment metrics.

This module provides several metrics to evaluate the alignment between model and brain
representations. All metrics implement the same interface, e.g. for RSA:

```python
# Initialize the metric. Some metrics may require additional parameters.
metric = RSA()

# Apply the metric to two feature matrices with shapes (n_samples, n_features)
score, details = metric(features1, features2)

# The score is a scalar value, and the details is a dictionary containing additional
# information. Have a look at the documentation of the respective metric for more
# details.
```
"""

from .base import Metric
from .linear_predictivity import LinearPredictivity
from .rsa import RSA

__all__ = ["Metric", "LinearPredictivity", "RSA"]
