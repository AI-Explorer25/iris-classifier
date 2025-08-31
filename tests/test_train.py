# tests/test_train.py
from src.train import main

def test_accuracy():
    """Optional test: ensure Decision Tree achieves accuracy >= 0.9"""
    acc = main(test_size=0.2, random_state=42)
    assert acc >= 0.9