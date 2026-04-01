"""
68 built-in language definitions across 5 regional groups.

Source: Tiny Aya tech report (Abagyan et al., 2025; arxiv 2506.10766), Table 1.
NLLB codes from FLORES-200 language list.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageSpec:
    nllb_code: str
    iso_639_1: str
    bcp47: str
    display_name: str
    script: str
    resource_level: str  # "high", "medium", "low"
    group: str  # "europe", "west_asia", "south_asia", "asia_pacific", "african"
    recommended_models: tuple[str, ...] = field(default_factory=tuple)


# ===== Europe (31 languages) =====
EUROPE: tuple[LanguageSpec, ...] = (
    LanguageSpec("eng_Latn", "en", "en", "English", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B", "ALMA-R", "GemmaX2-28-9B")),
    LanguageSpec("nld_Latn", "nl", "nl", "Dutch", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B")),
    LanguageSpec("fra_Latn", "fr", "fr", "French", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B", "ALMA-R", "GemmaX2-28-9B")),
    LanguageSpec("ita_Latn", "it", "it", "Italian", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B")),
    LanguageSpec("por_Latn", "pt", "pt", "Portuguese", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B", "ALMA-R")),
    LanguageSpec("ron_Latn", "ro", "ro", "Romanian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("spa_Latn", "es", "es", "Spanish", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B", "ALMA-R", "GemmaX2-28-9B")),
    LanguageSpec("ces_Latn", "cs", "cs", "Czech", "Latn", "medium", "europe",
                 ("nllb-200-3.3B", "ALMA-R")),
    LanguageSpec("pol_Latn", "pl", "pl", "Polish", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("ukr_Cyrl", "uk", "uk", "Ukrainian", "Cyrl", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("rus_Cyrl", "ru", "ru", "Russian", "Cyrl", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B", "GemmaX2-28-9B")),
    LanguageSpec("ell_Grek", "el", "el", "Greek", "Grek", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("deu_Latn", "de", "de", "German", "Latn", "high", "europe",
                 ("nllb-200-3.3B", "TowerInstruct-13B", "ALMA-R", "GemmaX2-28-9B")),
    LanguageSpec("dan_Latn", "da", "da", "Danish", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("swe_Latn", "sv", "sv", "Swedish", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("nob_Latn", "nb", "nb", "Bokmal", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("cat_Latn", "ca", "ca", "Catalan", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("glg_Latn", "gl", "gl", "Galician", "Latn", "low", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("cym_Latn", "cy", "cy", "Welsh", "Latn", "low", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("gle_Latn", "ga", "ga", "Irish", "Latn", "low", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("eus_Latn", "eu", "eu", "Basque", "Latn", "low", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("hrv_Latn", "hr", "hr", "Croatian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("lvs_Latn", "lv", "lv", "Latvian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("lit_Latn", "lt", "lt", "Lithuanian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("slk_Latn", "sk", "sk", "Slovak", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("slv_Latn", "sl", "sl", "Slovenian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("est_Latn", "et", "et", "Estonian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("fin_Latn", "fi", "fi", "Finnish", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("hun_Latn", "hu", "hu", "Hungarian", "Latn", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("srp_Cyrl", "sr", "sr", "Serbian", "Cyrl", "medium", "europe",
                 ("nllb-200-3.3B",)),
    LanguageSpec("bul_Cyrl", "bg", "bg", "Bulgarian", "Cyrl", "medium", "europe",
                 ("nllb-200-3.3B",)),
)

# ===== West Asia (5 languages) =====
WEST_ASIA: tuple[LanguageSpec, ...] = (
    LanguageSpec("arb_Arab", "ar", "ar", "Arabic", "Arab", "medium", "west_asia",
                 ("nllb-200-3.3B", "GemmaX2-28-9B", "Aya-Expanse-8B")),
    LanguageSpec("pes_Arab", "fa", "fa", "Persian", "Arab", "medium", "west_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("tur_Latn", "tr", "tr", "Turkish", "Latn", "medium", "west_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("mlt_Latn", "mt", "mt", "Maltese", "Latn", "low", "west_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("heb_Hebr", "he", "he", "Hebrew", "Hebr", "medium", "west_asia",
                 ("nllb-200-3.3B",)),
)

# ===== South Asia (9 languages) =====
SOUTH_ASIA: tuple[LanguageSpec, ...] = (
    LanguageSpec("hin_Deva", "hi", "hi", "Hindi", "Deva", "medium", "south_asia",
                 ("nllb-200-3.3B", "GemmaX2-28-9B", "Aya-Expanse-8B")),
    LanguageSpec("mar_Deva", "mr", "mr", "Marathi", "Deva", "low", "south_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("ben_Beng", "bn", "bn", "Bengali", "Beng", "medium", "south_asia",
                 ("nllb-200-3.3B", "GemmaX2-28-9B", "Aya-Expanse-8B")),
    LanguageSpec("guj_Gujr", "gu", "gu", "Gujarati", "Gujr", "low", "south_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("pan_Guru", "pa", "pa", "Punjabi", "Guru", "low", "south_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("tam_Taml", "ta", "ta", "Tamil", "Taml", "medium", "south_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("tel_Telu", "te", "te", "Telugu", "Telu", "low", "south_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("npi_Deva", "ne", "ne", "Nepali", "Deva", "low", "south_asia",
                 ("nllb-200-3.3B",)),
    LanguageSpec("urd_Arab", "ur", "ur", "Urdu", "Arab", "medium", "south_asia",
                 ("nllb-200-3.3B",)),
)

# ===== Asia Pacific (13 languages) =====
ASIA_PACIFIC: tuple[LanguageSpec, ...] = (
    LanguageSpec("tgl_Latn", "tl", "tl", "Tagalog", "Latn", "medium", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("zsm_Latn", "ms", "ms", "Malay", "Latn", "medium", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("ind_Latn", "id", "id", "Indonesian", "Latn", "medium", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("vie_Latn", "vi", "vi", "Vietnamese", "Latn", "medium", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("jav_Latn", "jv", "jv", "Javanese", "Latn", "low", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("khm_Khmr", "km", "km", "Khmer", "Khmr", "low", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("tha_Thai", "th", "th", "Thai", "Thai", "medium", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("lao_Laoo", "lo", "lo", "Lao", "Laoo", "low", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("zho_Hans", "zh", "zh-Hans", "Chinese Simplified", "Hans", "high", "asia_pacific",
                 ("nllb-200-3.3B", "GemmaX2-28-9B", "ALMA-R")),
    LanguageSpec("zho_Hant", "zh-TW", "zh-Hant", "Chinese Traditional", "Hant", "high", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("mya_Mymr", "my", "my", "Burmese", "Mymr", "low", "asia_pacific",
                 ("nllb-200-3.3B",)),
    LanguageSpec("jpn_Jpan", "ja", "ja", "Japanese", "Jpan", "high", "asia_pacific",
                 ("nllb-200-3.3B", "GemmaX2-28-9B")),
    LanguageSpec("kor_Hang", "ko", "ko", "Korean", "Hang", "high", "asia_pacific",
                 ("nllb-200-3.3B", "TowerInstruct-13B")),
)

# ===== African (10 languages) =====
AFRICAN: tuple[LanguageSpec, ...] = (
    LanguageSpec("amh_Ethi", "am", "am", "Amharic", "Ethi", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("hau_Latn", "ha", "ha", "Hausa", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("ibo_Latn", "ig", "ig", "Igbo", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("plt_Latn", "mg", "mg", "Malagasy", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("sna_Latn", "sn", "sn", "Shona", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("swh_Latn", "sw", "sw", "Swahili", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("wol_Latn", "wo", "wo", "Wolof", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("xho_Latn", "xh", "xh", "Xhosa", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("yor_Latn", "yo", "yo", "Yoruba", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
    LanguageSpec("zul_Latn", "zu", "zu", "Zulu", "Latn", "low", "african",
                 ("nllb-200-3.3B",)),
)

ALL_LANGUAGES: tuple[LanguageSpec, ...] = EUROPE + WEST_ASIA + SOUTH_ASIA + ASIA_PACIFIC + AFRICAN

GROUP_MAP: dict[str, tuple[LanguageSpec, ...]] = {
    "europe": EUROPE,
    "west_asia": WEST_ASIA,
    "south_asia": SOUTH_ASIA,
    "asia_pacific": ASIA_PACIFIC,
    "african": AFRICAN,
}
