"""
The :mod:`gsmote` provides the implementation of
Geometric SMOTE algorithm.
"""

from .eg_smote import EGSmote

from ._version import __version__

__all__ = ['EGSmote', '__version__']
