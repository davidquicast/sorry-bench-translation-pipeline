"""
aya-safety: PhD-grade multilingual translation framework for NLP research.

Solves the LLM refusal problem for safety benchmark translation via a 3-tier
architecture: NLLB-200 backbone (no content filters) → LLM refinement (framed
as MT improvement) → multi-metric quality assurance (including GEMBA).
"""

__version__ = "0.1.0"
