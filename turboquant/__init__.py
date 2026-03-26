"""
TurboQuant: Extreme KV Cache Quantization
Open-source implementation of Google TurboQuant (arXiv:2504.19874)

https://github.com/hackimov/turboquant-kv
"""

__version__ = "0.1.2"
__author__ = "hackimov"

from .calibration import (
    CalibrationMode,
    calibrate_turboquant_from_batches,
    calibrate_turboquant_from_tensor,
    kmeans_1d,
)
from .core import TurboQuantProd
from .transformers_integration import TurboQuantModel

__all__ = [
    "TurboQuantProd",
    "TurboQuantModel",
    "CalibrationMode",
    "calibrate_turboquant_from_tensor",
    "calibrate_turboquant_from_batches",
    "kmeans_1d",
    "__version__",
]