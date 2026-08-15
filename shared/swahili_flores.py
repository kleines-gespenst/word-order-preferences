"""
Swahili: FLORES / Hugging Face ``facebook/flores`` / Goldfish naming.

Your pipelines may use ``swh_Latn`` or ``swa_Latn`` as the logical ``--lang``; both count as Swahili.

For ``load_dataset("facebook/flores", ...)`` the Hub config is **only** ``swh_Latn`` (see HF ``flores.py`` ``_LANGUAGES``). Do not use ``swa_Latn`` there.

Goldfish checkpoints on the Hub use ``swa_Latn`` (e.g. ``goldfish-models/swa_Latn_5mb``).

Use :func:`is_swahili` / :func:`facebook_flores_hub_configs_to_try` / :func:`goldfish_repo_lang`
from this module when adding Swahili-specific behaviour elsewhere.
"""

from __future__ import annotations

from typing import Final, List, Tuple

# Any of these --lang / CSV values mean “Swahili” for our tooling.
SWAHILI_LOGICAL_CODES: Final[frozenset[str]] = frozenset({"swh_Latn", "swa_Latn"})

# ``load_dataset("facebook/flores", ...)``: single canonical config (matches HF flores.py).
FACEBOOK_FLORES_CONFIG_TRY_ORDER: Final[Tuple[str, ...]] = ("swh_Latn",)

# Second segment of ``goldfish-models/<name>_<size>`` for Swahili.
GOLDFISH_REPO_LANG: Final[str] = "swa_Latn"


def is_swahili(logical_lang: str) -> bool:
    return logical_lang in SWAHILI_LOGICAL_CODES


def facebook_flores_hub_configs_to_try() -> List[str]:
    """Config names to pass to ``load_dataset('facebook/flores', name, ...)``, in order."""
    return list(FACEBOOK_FLORES_CONFIG_TRY_ORDER)


def goldfish_repo_lang() -> str:
    """Return the Goldfish Hub repo language segment (before ``_5mb``, etc.)."""
    return GOLDFISH_REPO_LANG
