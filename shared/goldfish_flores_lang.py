"""
FLORES language codes: logical / CSV names vs Goldfish Hub and ``facebook/flores`` configs.

Swahili details live in :mod:`swahili_flores`.
"""

from __future__ import annotations

from typing import List

import swahili_flores


def goldfish_checkpoint_prefix(flores_lang: str) -> str:
    """Return the repo name segment before _5mb / _100mb (may differ from FLORES --lang)."""
    if swahili_flores.is_swahili(flores_lang):
        return swahili_flores.goldfish_repo_lang()
    return flores_lang


def facebook_flores_config_candidates(logical_lang: str) -> List[str]:
    """
    Config names to try for ``load_dataset('facebook/flores', ...)``, in order.

    Delegates to :mod:`swahili_flores` for Swahili.
    """
    if swahili_flores.is_swahili(logical_lang):
        return swahili_flores.facebook_flores_hub_configs_to_try()
    return [logical_lang]
