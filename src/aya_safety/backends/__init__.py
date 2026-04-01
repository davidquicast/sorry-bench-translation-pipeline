"""
Backend registry and factory.

All backends register themselves here. The pipeline uses create_backend()
to instantiate backends by name from config.
"""

from __future__ import annotations

from typing import Any, Type

from aya_safety.backends.base import LLMBackend, MTBackend, QAMetric

# Registry: name → class
_MT_REGISTRY: dict[str, Type[MTBackend]] = {}
_QA_REGISTRY: dict[str, Type[QAMetric]] = {}


def register_backend(name: str) -> callable:
    """Decorator to register an MTBackend or LLMBackend class."""
    def decorator(cls: Type[MTBackend]) -> Type[MTBackend]:
        _MT_REGISTRY[name] = cls
        return cls
    return decorator


def register_metric(name: str) -> callable:
    """Decorator to register a QAMetric class."""
    def decorator(cls: Type[QAMetric]) -> Type[QAMetric]:
        _QA_REGISTRY[name] = cls
        return cls
    return decorator


def create_backend(name: str, **kwargs: Any) -> MTBackend:
    """Instantiate a translation backend by name."""
    if name not in _MT_REGISTRY:
        # Lazy import to trigger registration
        _lazy_import_backends()
    if name not in _MT_REGISTRY:
        available = sorted(_MT_REGISTRY.keys())
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")
    return _MT_REGISTRY[name](**kwargs)


def create_metric(name: str, **kwargs: Any) -> QAMetric:
    """Instantiate a QA metric by name."""
    if name not in _QA_REGISTRY:
        _lazy_import_metrics()
    if name not in _QA_REGISTRY:
        available = sorted(_QA_REGISTRY.keys())
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return _QA_REGISTRY[name](**kwargs)


def available_backends() -> list[str]:
    _lazy_import_backends()
    return sorted(_MT_REGISTRY.keys())


def available_metrics() -> list[str]:
    _lazy_import_metrics()
    return sorted(_QA_REGISTRY.keys())


def _lazy_import_backends() -> None:
    """Import backend modules to trigger registration decorators."""
    import importlib
    for module_name in [
        "aya_safety.backends.tier1.nllb",
        "aya_safety.backends.tier1.nllb_ct2",
        "aya_safety.backends.tier2.cohere_backend",
        "aya_safety.backends.tier2.openai_backend",
        "aya_safety.backends.tier2.anthropic_backend",
        "aya_safety.backends.tier2.gemini_backend",
        "aya_safety.backends.tier2.local_hf",
        "aya_safety.backends.tier2.google_translate",
        "aya_safety.backends.tier2.deepl_backend",
        "aya_safety.backends.tier2.azure_translate",
        "aya_safety.backends.tier2.amazon_translate",
    ]:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass  # Optional dependency not installed


def _lazy_import_metrics() -> None:
    """Import metric modules to trigger registration decorators."""
    import importlib
    for module_name in [
        "aya_safety.backends.tier3.comet_metric",
        "aya_safety.backends.tier3.chrf_metric",
        "aya_safety.backends.tier3.backtranslation",
        "aya_safety.backends.tier3.gemba",
        "aya_safety.backends.tier3.langdetect_metric",
        "aya_safety.backends.tier3.safety_consistency",
    ]:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass
