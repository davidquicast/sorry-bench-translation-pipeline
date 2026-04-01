"""
LanguageRegistry: lookup languages by any code format, register custom languages.
"""

from __future__ import annotations

import logging

from aya_safety.languages.definitions import ALL_LANGUAGES, GROUP_MAP, LanguageSpec

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """
    Central registry for language specifications.

    Supports lookup by ISO 639-1 code, BCP-47 tag, NLLB code, or display name.
    Initialized with 68 built-in languages; custom languages can be registered
    at runtime via config or CLI.
    """

    def __init__(self) -> None:
        # Primary index: ISO 639-1 code → LanguageSpec
        self._by_iso: dict[str, LanguageSpec] = {}
        # Secondary indices
        self._by_nllb: dict[str, LanguageSpec] = {}
        self._by_bcp47: dict[str, LanguageSpec] = {}
        self._by_name: dict[str, LanguageSpec] = {}

        for spec in ALL_LANGUAGES:
            self._register(spec)

    def _register(self, spec: LanguageSpec) -> None:
        self._by_iso[spec.iso_639_1] = spec
        self._by_nllb[spec.nllb_code] = spec
        self._by_bcp47[spec.bcp47] = spec
        self._by_name[spec.display_name.lower()] = spec

    def register(self, spec: LanguageSpec) -> None:
        """Register a custom language at runtime."""
        if spec.iso_639_1 in self._by_iso:
            logger.warning("Overriding existing language: %s", spec.iso_639_1)
        self._register(spec)
        logger.info("Registered custom language: %s (%s)", spec.display_name, spec.iso_639_1)

    def get(self, code: str) -> LanguageSpec:
        """
        Lookup by any code format: ISO 639-1, BCP-47, NLLB code, or display name.

        Raises KeyError if not found.
        """
        if code in self._by_iso:
            return self._by_iso[code]
        if code in self._by_nllb:
            return self._by_nllb[code]
        if code in self._by_bcp47:
            return self._by_bcp47[code]
        lowered = code.lower()
        if lowered in self._by_name:
            return self._by_name[lowered]
        raise KeyError(
            f"Language '{code}' not found. Register it via config custom_languages "
            f"or --add-language CLI flag."
        )

    def resolve_nllb_code(self, code: str) -> str:
        """Resolve any code format to the NLLB code needed by NLLB-200."""
        return self.get(code).nllb_code

    def resolve_display_name(self, code: str) -> str:
        """Resolve any code format to the human-readable display name."""
        return self.get(code).display_name

    def list_by_group(self, group: str) -> list[LanguageSpec]:
        """Return all languages in a regional group."""
        if group not in GROUP_MAP:
            raise KeyError(f"Unknown group '{group}'. Options: {list(GROUP_MAP.keys())}")
        return list(GROUP_MAP[group])

    def supported_codes(self) -> set[str]:
        """Return all registered ISO 639-1 codes."""
        return set(self._by_iso.keys())

    def __contains__(self, code: str) -> bool:
        try:
            self.get(code)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._by_iso)
