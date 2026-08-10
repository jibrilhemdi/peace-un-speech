"""Shared machinery for the annotation pipeline.

Nothing in this package knows the name of a codebook, a dimension or a code. Everything is
read from ``codebooks/*.yaml`` and ``config.yaml`` at runtime, so a new codebook is a new
file and never a code change.
"""
