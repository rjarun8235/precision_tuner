"""
PrecisionTuner - Instruction-Following Dataset Generator

A systematic constraint-based dataset generator for training
models with precise instruction-following capabilities.
"""

from precision_tuner.precision_tuner import ModelConfig, MemoryOptimizedOllamaClient
from precision_tuner.constraint_evaluator import ConstraintEvaluator
from precision_tuner.dataset_generator import PrecisionDatasetGenerator
from precision_tuner.dataset_saver import DatasetSaver

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "MemoryOptimizedOllamaClient",
    "ConstraintEvaluator",
    "PrecisionDatasetGenerator",
    "DatasetSaver"
]
