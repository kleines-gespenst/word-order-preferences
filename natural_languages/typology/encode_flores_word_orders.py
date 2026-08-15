"""
Generate [Base]-[5-char] directionality encoding for all FLORES-200 languages.

Encoding scheme:
  [Base]-[Pos1][Pos2][Pos3][Pos4][Pos5]

  Base   = SOV / SVO / VSO / VOS / OVS / OSV
  Pos 1  VP_Comp : L = CompV  (S_Comp before verb)   R = VComp  (verb before S_Comp)
                   proxy: OV→L, VO→R (indices 9,8 in syntax_average)
                   NOTE: only reliable when GB135=Yes (clausal objects pattern with nominals)
  Pos 2  Comp    : L = S Comp (subord. after clause)  R = Comp S (subord. before clause)
                   from syntax_average idx 82 (after) / 81 (before); WALS 94A fallback
  Pos 3  PP      : L = Postposition                   R = Preposition
                   from syntax_average idx 21/20; WALS 85A fallback
  Pos 4  NP      : L = AdjN                           R = NAdj
                   from syntax_average idx 24/25; WALS 87A fallback
  Pos 5  Rel     : L = RelN                           R = NRel
                   from syntax_average idx 32/33; WALS 90A fallback

  '?' = NoData or NoDominant for that position
  Full label e.g. SVO-RRLRR; partial e.g. SOV-LL?LL if one position is unknown.

Sources (priority order):
  1. lang2vec syntax_average  (aggregates WALS + SSWL + Ethnologue)
  2. WALS CLDF fallback       (for positions still NoData after lang2vec)
  3. URIEL+ / Grambank GB135  (validates VP_Comp: if GB135=No → mark Pos1 as '?')
  4. Manual overrides         (MANUAL_POS1 for Pos1; MANUAL_POSITIONS for Pos2–5;
                               applied when value is still '?' or lang2vec is the source)

Run with:
    conda activate <env>
    python encode_flores_word_orders.py
"""

import urllib.request
import csv
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import lang2vec.lang2vec as l2v
from urielplus import urielplus

DATA_ROOT = Path(os.environ.get("WORD_ORDER_DATA", "data"))

# ── FLORES-200 language list ──────────────────────────────────────────────
FLORES_NAMES = {
    "ace_Arab": "Acehnese (Arabic script)", "ace_Latn": "Acehnese (Latin script)",
    "acm_Arab": "Mesopotamian Arabic", "acq_Arab": "Ta'izzi-Adeni Arabic",
    "aeb_Arab": "Tunisian Arabic", "afr_Latn": "Afrikaans",
    "ajp_Arab": "South Levantine Arabic", "aka_Latn": "Akan",
    "amh_Ethi": "Amharic", "apc_Arab": "North Levantine Arabic",
    "arb_Arab": "Modern Standard Arabic", "arb_Latn": "Modern Standard Arabic (Romanized)",
    "ars_Arab": "Najdi Arabic", "ary_Arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic", "asm_Beng": "Assamese",
    "ast_Latn": "Asturian", "awa_Deva": "Awadhi",
    "ayr_Latn": "Central Aymara", "azb_Arab": "South Azerbaijani",
    "azj_Latn": "North Azerbaijani", "bak_Cyrl": "Bashkir",
    "bam_Latn": "Bambara", "ban_Latn": "Balinese",
    "bel_Cyrl": "Belarusian", "bem_Latn": "Bemba",
    "ben_Beng": "Bengali", "bho_Deva": "Bhojpuri",
    "bjn_Arab": "Banjar (Arabic script)", "bjn_Latn": "Banjar (Latin script)",
    "bod_Tibt": "Standard Tibetan", "bos_Latn": "Bosnian",
    "bug_Latn": "Buginese", "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan", "ceb_Latn": "Cebuano",
    "ces_Latn": "Czech", "cjk_Latn": "Chokwe",
    "ckb_Arab": "Central Kurdish", "crh_Latn": "Crimean Tatar",
    "cym_Latn": "Welsh", "dan_Latn": "Danish",
    "deu_Latn": "German", "dik_Latn": "Southwestern Dinka",
    "dyu_Latn": "Dyula", "dzo_Tibt": "Dzongkha",
    "ell_Grek": "Greek", "eng_Latn": "English",
    "epo_Latn": "Esperanto", "est_Latn": "Estonian",
    "eus_Latn": "Basque", "ewe_Latn": "Ewe",
    "fao_Latn": "Faroese", "pes_Arab": "Western Persian",
    "fij_Latn": "Fijian", "fin_Latn": "Finnish",
    "fon_Latn": "Fon", "fra_Latn": "French",
    "fur_Latn": "Friulian", "fuv_Latn": "Nigerian Fulfulde",
    "gla_Latn": "Scottish Gaelic", "gle_Latn": "Irish",
    "glg_Latn": "Galician", "grn_Latn": "Guarani",
    "guj_Gujr": "Gujarati", "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa", "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi", "hne_Deva": "Chhattisgarhi",
    "hrv_Latn": "Croatian", "hun_Latn": "Hungarian",
    "hye_Armn": "Armenian", "ibo_Latn": "Igbo",
    "ilo_Latn": "Ilocano", "ind_Latn": "Indonesian",
    "isl_Latn": "Icelandic", "ita_Latn": "Italian",
    "jav_Latn": "Javanese", "jpn_Jpan": "Japanese",
    "kab_Latn": "Kabyle", "kac_Latn": "Jingpho",
    "kam_Latn": "Kamba", "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri (Arabic script)", "kas_Deva": "Kashmiri (Devanagari script)",
    "kat_Geor": "Georgian", "knc_Arab": "Central Kanuri (Arabic script)",
    "knc_Latn": "Central Kanuri (Latin script)", "kaz_Cyrl": "Kazakh",
    "kbp_Latn": "Kabiyè", "kea_Latn": "Kabuverdianu",
    "khm_Khmr": "Khmer", "kik_Latn": "Kikuyu",
    "kin_Latn": "Kinyarwanda", "kir_Cyrl": "Kyrgyz",
    "kmb_Latn": "Kimbundu", "kmr_Latn": "Northern Kurdish",
    "kon_Latn": "Kikongo", "kor_Hang": "Korean",
    "lao_Laoo": "Lao", "lvs_Latn": "Standard Latvian",
    "lij_Latn": "Ligurian", "lim_Latn": "Limburgish",
    "lin_Latn": "Lingala", "lit_Latn": "Lithuanian",
    "lmo_Latn": "Lombard", "ltg_Latn": "Latgalian",
    "ltz_Latn": "Luxembourgish", "lua_Latn": "Luba-Kasai",
    "lug_Latn": "Ganda", "luo_Latn": "Luo",
    "lus_Latn": "Mizo", "mag_Deva": "Magahi",
    "mai_Deva": "Maithili", "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi", "min_Arab": "Minangkabau (Arabic script)",
    "min_Latn": "Minangkabau (Latin script)", "mkd_Cyrl": "Macedonian",
    "plt_Latn": "Plateau Malagasy", "mlt_Latn": "Maltese",
    "mni_Beng": "Meitei", "khk_Cyrl": "Halh Mongolian",
    "mos_Latn": "Mossi", "mri_Latn": "Maori",
    "zsm_Latn": "Standard Malay", "mya_Mymr": "Burmese",
    "nld_Latn": "Dutch", "nno_Latn": "Norwegian Nynorsk",
    "nob_Latn": "Norwegian Bokmål", "npi_Deva": "Nepali",
    "nso_Latn": "Northern Sotho", "nus_Latn": "Nuer",
    "nya_Latn": "Nyanja", "oci_Latn": "Occitan",
    "gaz_Latn": "West Central Oromo", "ory_Orya": "Odia",
    "pag_Latn": "Pangasinan", "pan_Guru": "Eastern Panjabi",
    "pap_Latn": "Papiamento", "pol_Latn": "Polish",
    "por_Latn": "Portuguese", "prs_Arab": "Dari",
    "pbt_Arab": "Southern Pashto", "quy_Latn": "Ayacucho Quechua",
    "ron_Latn": "Romanian", "run_Latn": "Rundi",
    "rus_Cyrl": "Russian", "sag_Latn": "Sango",
    "san_Deva": "Sanskrit", "sat_Beng": "Santali",
    "scn_Latn": "Sicilian", "shn_Mymr": "Shan",
    "sin_Sinh": "Sinhala", "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian", "smo_Latn": "Samoan",
    "sna_Latn": "Shona", "snd_Arab": "Sindhi",
    "som_Latn": "Somali", "sot_Latn": "Southern Sotho",
    "spa_Latn": "Spanish", "als_Latn": "Tosk Albanian",
    "srd_Latn": "Sardinian", "srp_Cyrl": "Serbian",
    "ssw_Latn": "Swati", "sun_Latn": "Sundanese",
    "swe_Latn": "Swedish", "swh_Latn": "Swahili",
    "szl_Latn": "Silesian", "tam_Taml": "Tamil",
    "tat_Cyrl": "Tatar", "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik", "tgl_Latn": "Tagalog",
    "tha_Thai": "Thai", "tir_Ethi": "Tigrinya",
    "taq_Latn": "Tamasheq (Latin script)", "taq_Tfng": "Tamasheq (Tifinagh script)",
    "tpi_Latn": "Tok Pisin", "tsn_Latn": "Tswana",
    "tso_Latn": "Tsonga", "tuk_Latn": "Turkmen",
    "tum_Latn": "Tumbuka", "tur_Latn": "Turkish",
    "twi_Latn": "Twi", "tzm_Tfng": "Central Atlas Tamazight",
    "uig_Arab": "Uyghur", "ukr_Cyrl": "Ukrainian",
    "umb_Latn": "Umbundu", "urd_Arab": "Urdu",
    "uzn_Latn": "Northern Uzbek", "vec_Latn": "Venetian",
    "vie_Latn": "Vietnamese", "war_Latn": "Waray",
    "wol_Latn": "Wolof", "xho_Latn": "Xhosa",
    "ydd_Hebr": "Eastern Yiddish", "yor_Latn": "Yoruba",
    "yue_Hant": "Yue Chinese", "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)", "zul_Latn": "Zulu",
}

# ── Constants ──────────────────────────────────────────────────────────────
ORDER_NAMES = ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV"]
BASE_ORDERS = set(ORDER_NAMES) | {"NoDominant"}
DOMINANT_THRESHOLD = 0.55

WALS_81A = {"1":"SOV","2":"SVO","3":"VSO","4":"VOS","5":"OVS","6":"OSV","7":"NoDominant"}
WALS_81B = {"1":"SOV/SVO","2":"SVO/VSO","3":"VSO/VOS","4":"SVO/VOS","5":"SOV/OVS"}
WALS_85A = {"1":"Postposition","2":"Preposition","3":"Inposition","4":"NoDominant","5":"NoAdpositions"}
WALS_87A = {"1":"AdjN","2":"NAdj","3":"NoDominant","4":"OnlyInternalRC"}
WALS_90A = {"1":"NRel","2":"RelN","3":"InternallyHeaded","4":"Correlative",
            "5":"Adjoined","6":"DoubleHeaded","7":"NoDominant"}
WALS_94A = {"1":"CompInitial","2":"CompFinal","3":"NoComp","4":"NoDominant"}

WALS_CLDF = "https://raw.githubusercontent.com/cldf-datasets/wals/v2020.3/cldf/"

OUTPUT_CSV = str(DATA_ROOT / "flores_clustering/flores200_encoding.csv")

GRAMBANK_RAW_CSV = Path(os.path.dirname(urielplus.__file__)) / \
    "database" / "urielplus_csvs" / "grambank_data.csv"

GB_RAW_FEATS = {
    "pos2_comp": ("S_POSTPOSED_CCOMP_THINKING", "S_PREPOSED_CCOMP_THINKING"),
    "pos3_pp":   ("S_ADPOSITION_AFTER_NOUN",    "S_ADPOSITION_BEFORE_NOUN"),
    "pos4_np":   ("S_ADJECTIVE_BEFORE_NOUN",    "S_ADJECTIVE_AFTER_NOUN"),
    "pos5_rel":  ("S_RELATIVE_BEFORE_NOUN",     "S_RELATIVE_AFTER_NOUN"),
}

WO_TO_POS1 = {"SVO": "R", "VSO": "R", "VOS": "R",
               "SOV": "L", "OVS": "L", "OSV": "L"}

# ── Manual pos1 (VP_Comp) overrides ───────────────────────────────────────
# Languages where OV/VO in lang2vec is NoDominant (free nominal placement),
# but the complement clause is known to be postposed (VComp = R).
# Covers Germanic V2, Slavic free-WO, Baltic, Romance, and related.
MANUAL_POS1 = {
    # Germanic V2 / free-WO
    "afr": "R", "deu": "R", "nld": "R", "dan": "R",
    "nob": "R", "nno": "R", "isl": "R", "fao": "R",
    "ltz": "R", "ydd": "R", "lim": "R",
    # Slavic free-WO
    "bel": "R", "rus": "R", "ukr": "R", "pol": "R",
    "ces": "R", "slk": "R", "slv": "R", "hrv": "R",
    "bos": "R", "srp": "R", "mkd": "R", "bul": "R",
    "szl": "R",
    # Baltic
    "lit": "R", "lvs": "R", "ltg": "R",
    # Romance free-WO
    "ita": "R", "spa": "R", "por": "R", "fra": "R",
    "ron": "R", "cat": "R", "oci": "R", "glg": "R",
    "ast": "R", "lmo": "R", "lij": "R", "fur": "R",
    "vec": "R", "scn": "R", "srd": "R",
    # Other European
    "ell": "R", "als": "R", "fin": "R", "est": "R",
    "hun": "R", "hye": "R", "mlt": "R", "epo": "R",
    # Austronesian SVO with GB135=No
    "min": "R",  # Minangkabau: VComp; GB135=No
    # Dravidian SOV with GB135=No
    "mal": "L",  # Malayalam: CompV (complement before verb); GB135=No
    # Kartvelian with GB135=No
    "kat": "R",  # Georgian: VComp (complement after verb); GB135=No
    # Semitic VSO with postposed anna/an-complementizer
    "arb": "R",  # Modern Standard Arabic: VComp; GB135=No
    # Indo-Aryan SOV with postposed ki-complementizer
    "mar": "R",  # Marathi: VComp via ki; GB135=No
    "pan": "R",  # Eastern Panjabi: VComp via ki; GB135=No
    # Creoles
    "tpi": "R",  # Tok Pisin: VComp (complement after verb); OV/VO NoDominant
}

# ── Manual base-order overrides ────────────────────────────────────────────
# Manual word order overrides for languages lang2vec does not resolve.
# Applied when lang2vec + WALS return NoData for base word order.
# "NoData" entries leave the language unlabeled (genuinely unknown).
MANUAL_BASE_ORDER = {
    "acq_Arab": "NoData",   # Ta'izzi-Adeni Arabic
    "aeb_Arab": "NoData",   # Tunisian Arabic
    "ajp_Arab": "NoData",   # South Levantine Arabic
    "ast_Latn": "SVO",      # Asturian
    "azj_Latn": "NoData",   # North Azerbaijani
    "bak_Cyrl": "SOV",      # Bashkir
    "ban_Latn": "SVO",      # Balinese
    "bel_Cyrl": "SVO",      # Belarusian
    "bjn_Arab": "SVO",      # Banjar (Arabic script)
    "bjn_Latn": "SVO",      # Banjar (Latin script)
    "cjk_Latn": "SVO",      # Chokwe
    "crh_Latn": "SOV",      # Crimean Tatar
    "dik_Latn": "NoData",   # Southwestern Dinka
    "dyu_Latn": "SOV",      # Dyula
    "dzo_Tibt": "SOV",      # Dzongkha
    "fur_Latn": "SVO",      # Friulian
    "fuv_Latn": "SVO",      # Nigerian Fulfulde
    "hne_Deva": "SOV",      # Chhattisgarhi
    "kea_Latn": "NoData",   # Kabuverdianu
    "kmb_Latn": "SVO",      # Kimbundu
    "lij_Latn": "NoData",   # Ligurian
    "lim_Latn": "NoData",   # Limburgish
    "lmo_Latn": "NoData",   # Lombard
    "ltg_Latn": "NoData",   # Latgalian
    "lua_Latn": "SVO",      # Luba-Kasai
    "lvs_Latn": "SVO",      # Standard Latvian
    "mlt_Latn": "SVO",      # Maltese
    "nno_Latn": "SVO",      # Norwegian Nynorsk
    "pap_Latn": "SVO",      # Papiamento
    "pbt_Arab": "SOV",      # Southern Pashto
    "prs_Arab": "SOV",      # Dari
    "run_Latn": "SVO",      # Rundi
    "slk_Latn": "SVO",      # Slovak
    "snd_Arab": "SOV",      # Sindhi
    "szl_Latn": "SVO",      # Silesian
    "tsn_Latn": "SVO",      # Tswana
    "tso_Latn": "SVO",      # Tsonga
    "tum_Latn": "SVO",      # Tumbuka
    "tur_Latn": "SOV",      # Turkish
    "umb_Latn": "SVO",      # Umbundu
    "vec_Latn": "SVO",      # Venetian
    "war_Latn": "VSO",      # Waray
}

# ── Manual pos2-pos5 overrides ────────────────────────────────────────────
# iso_code → {pos_name: L/R/N}.  Applied when the position is '?', 'N', or
# when the current source is 'lang2vec' (lang2vec can be unreliable for
# NP-internal features in free-WO languages).
# NoDominant (N) values from any source may be overridden when the dominant
# direction is known from typological knowledge.
MANUAL_POSITIONS = {
    # Germanic SVO: CompInitial(R) Preposition(R) AdjN(L) NRel(R)
    "afr": {"pos3": "R"},  # Afrikaans: Preposition; NoDominant in sources
    "fao": {"pos4": "L"},  # Faroese: AdjN (Germanic); NoDominant in sources
    "ydd": {"pos2": "R", "pos3": "R", "pos4": "L", "pos5": "R"},  # Eastern Yiddish: lang2vec NoData
    # Romance SVO: CompInitial(R) Preposition(R) NAdj(R) NRel(R)
    "ast": {"pos2": "R", "pos3": "R", "pos4": "R", "pos5": "R"},  # Asturian: lang2vec NoData
    "glg": {"pos4": "R"},  # Galician: NAdj dominant; NoDominant in sources
    "oci": {"pos4": "R"},  # Occitan: NAdj dominant; NoDominant in sources
    "scn": {"pos4": "R"},  # Sicilian: NAdj dominant; NoDominant in sources
    "vec": {"pos2": "R", "pos3": "R", "pos4": "R", "pos5": "R"},  # Venetian: lang2vec NoData
    # Other European
    "epo": {"pos2": "R"},  # Esperanto: CompInitial (ke); lang2vec NoData
    # Slavic SVO: CompInitial(R) Preposition(R) AdjN(L) NRel(R)
    "bel": {"pos3": "R"},  # Belarusian: Preposition; NoDominant in sources
    "szl": {"pos2": "R", "pos3": "R", "pos4": "L", "pos5": "R"},  # Silesian: lang2vec NoData
    "hrv": {"pos2": "R", "pos3": "R", "pos4": "L", "pos5": "R"},  # Croatian
    "slk": {"pos2": "R", "pos3": "R", "pos4": "L", "pos5": "R"},  # Slovak
    "srp": {"pos3": "R", "pos4": "L"},  # Serbian: fill pos3, correct lang2vec NAdj→AdjN
    "bos": {"pos5": "R"},               # Bosnian: NRel missing from lang2vec
    # Baltic SVO
    "lvs": {"pos2": "R", "pos3": "R", "pos4": "L", "pos5": "R"},  # Standard Latvian
    # Turkic SOV: CompFinal(L)
    "tuk": {"pos2": "L"},  # Turkmen: subordinator postposed; lang2vec NoData
    "uig": {"pos2": "L"},  # Uyghur: subordinator postposed; lang2vec NoData
    # Creoles (APiCS source; lang2vec wrong)
    "hat": {"pos4": "R"},  # Haitian Creole: NAdj dominant; NoDominant in sources
    "pap": {"pos2": "R", "pos4": "R"},  # Papiamento: CompInitial, NAdj dominant; APiCS NoDominant overridden
    "tpi": {"pos2": "R", "pos3": "R"},  # Tok Pisin: olsem CompInitial, long/bilong prepositions (APiCS)
    # Indo-Aryan SOV: RelN (lang2vec NoData, Grambank/APiCS no coverage)
    "san": {"pos2": "L", "pos4": "L", "pos5": "L"},  # Sanskrit: CompFinal, AdjN, RelN; lang2vec NoData
    "sin": {"pos2": "L", "pos5": "L"},  # Sinhala: CompFinal, RelN; lang2vec NoData/wrong
    "awa": {"pos5": "L"},  # Awadhi: prenominal relative clauses
    "ben": {"pos5": "L"},  # Bengali: prenominal relative clauses
    "bho": {"pos5": "L"},  # Bhojpuri: prenominal relative clauses
    "guj": {"pos5": "L"},  # Gujarati: prenominal relative clauses; lang2vec NoData
    "hin": {"pos5": "L"},  # Hindi: RelN dominant; NoDominant in sources
    "hne": {"pos2": "R", "pos3": "L", "pos4": "L", "pos5": "L"},  # Chhattisgarhi: lang2vec NoData
    "urd": {"pos5": "L"},  # Urdu: prenominal relative clauses (WALS: Correlative)
    # Dravidian SOV: CompFinal(L) RelN(L)
    "kan": {"pos5": "L"},  # Kannada: RelN dominant; NoDominant in sources
    "tam": {"pos2": "L"},  # Tamil: CompFinal; NoDominant in sources
    # Bantu
    "kon": {"pos5": "R"},  # Kikongo: postnominal relatives; lang2vec NoData
    "run": {"pos2": "R", "pos3": "R", "pos4": "R", "pos5": "R"},  # Rundi: lang2vec NoData
    "umb": {"pos2": "R"},  # Umbundu: CompInitial; lang2vec NoData
    "xho": {"pos2": "R"},  # Xhosa: CompInitial; lang2vec NoData
    # Sino-Tibetan
    "yue": {"pos2": "R"},  # Yue Chinese (Cantonese): CompInitial; lang2vec NoData
    # Austronesian
    "ban": {"pos2": "R"},  # Balinese: CompInitial; lang2vec NoData
    "jav": {"pos2": "R", "pos4": "R", "pos5": "R"},  # Javanese: lang2vec NoData
    "pag": {"pos4": "L"},  # Pangasinan: AdjN; NoDominant in sources
    "zsm": {"pos2": "R"},  # Standard Malay: CompInitial; lang2vec NoData
    # Semitic/Romance
    "mlt": {"pos2": "R"},  # Maltese: CompInitial; lang2vec NoData
    # Mande
    "bam": {"pos5": "L"},  # Bambara: prenominal relatives; WALS gives Correlative (unmapped)
}


# ── lang2vec helpers ───────────────────────────────────────────────────────
def _v(vec, idx):
    v = vec[idx]
    return None if v == "--" else v

def _is_on(vec, idx):
    v = _v(vec, idx)
    return v is not None and float(v) == 1.0

def _dominant_order(vec):
    w = [0.0 if _v(vec, i) is None else float(_v(vec, i)) for i in range(6)]
    pairs = [(n, v) for n, v in zip(ORDER_NAMES, w) if v > 0]
    if not pairs:
        return "NoData", ""
    pairs.sort(key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in pairs)
    top_share = pairs[0][1] / total
    dominant = "NoDominant" if (len(pairs) > 1 and
                                (pairs[0][1] == pairs[1][1] or top_share < DOMINANT_THRESHOLD)) \
                             else pairs[0][0]
    dist = ";".join(f"{n}:{v/total*100:.1f}%" for n, v in pairs)
    return dominant, dist

def _pair(vec, idx_a, idx_b, label_a, label_b):
    a, b = _is_on(vec, idx_a), _is_on(vec, idx_b)
    if a and b: return "NoDominant"
    if a: return label_a
    if b: return label_b
    return "NoData"

def _triple(vec, idx_a, idx_b, idx_c, la, lb, lc):
    active = [l for on, l in [(  _is_on(vec, idx_a), la),
                               (_is_on(vec, idx_b), lb),
                               (_is_on(vec, idx_c), lc)] if on]
    if len(active) == 1: return active[0]
    if len(active) > 1:  return "NoDominant"
    return "NoData"


# ── WALS CLDF fallback ─────────────────────────────────────────────────────
def load_wals():
    langs_txt  = urllib.request.urlopen(WALS_CLDF + "languages.csv").read().decode()
    values_txt = urllib.request.urlopen(WALS_CLDF + "values.csv").read().decode()

    param_maps = {"81A":WALS_81A,"81B":WALS_81B,"85A":WALS_85A,
                  "87A":WALS_87A,"90A":WALS_90A,"94A":WALS_94A}

    wals_id_to_iso = {r["ID"]: r["ISO639P3code"]
                      for r in csv.DictReader(StringIO(langs_txt))
                      if r.get("ISO639P3code")}

    entries = {}
    for r in csv.DictReader(StringIO(values_txt)):
        pid = r["Parameter_ID"]
        if pid not in param_maps: continue
        val = param_maps[pid].get(r["Value"])
        if val is None: continue
        iso = wals_id_to_iso.get(r["Language_ID"])
        if not iso: continue
        entries.setdefault(iso, {})[pid] = val

    print(f"  WALS loaded: {len(entries)} languages")
    return entries


# ── URIEL+ / Grambank ──────────────────────────────────────────────────────
URIELPLUS_GLOTTOCODE_MAP = os.path.join(
    os.path.dirname(urielplus.__file__),
    "database", "urielplus_csvs", "uriel_glottocode_map.csv"
)
GRAMBANK_SRC_IDX = 10  # index after integrate_grambank()
APICS_SRC_IDX    = 11  # index after integrate_grambank() + integrate_apics()

# Feature pairs: (feat_for_L_value, feat_for_R_value)
GRAMBANK_POS_FEATS = {
    # pos2: POSTPOSED comp = comp after clause = head-final = L
    #        PREPOSED comp = comp before clause = head-initial = R
    "pos2_comp": ("S_POSTPOSED_CCOMP_THINKING", "S_PREPOSED_CCOMP_THINKING"),
    "pos3_pp":   ("S_ADPOSITION_AFTER_NOUN",    "S_ADPOSITION_BEFORE_NOUN"),
    "pos4_np":   ("S_ADJECTIVE_BEFORE_NOUN",    "S_ADJECTIVE_AFTER_NOUN"),
    "pos5_rel":  ("S_RELATIVE_BEFORE_NOUN",     "S_RELATIVE_AFTER_NOUN"),
}

def load_grambank(iso_codes):
    """
    Return {iso: {'gb135': 'Yes'/'No'/'NoData',
                  'pos3_pp': 'L'/'R'/'?',   # Grambank
                  'pos4_np': 'L'/'R'/'?',
                  'pos5_rel': 'L'/'R'/'?',
                  'apics_pos3_pp': 'L'/'R'/'?',  # APiCS
                  'apics_pos4_np': 'L'/'R'/'?',
                  'apics_pos5_rel': 'L'/'R'/'?'}}
    """
    import numpy as np

    iso_to_glotto = {}
    with open(URIELPLUS_GLOTTOCODE_MAP) as f:
        for r in csv.DictReader(f):
            iso_to_glotto[r["code"]] = r["glottocode"]

    u = urielplus.URIELPlus()
    u.integrate_grambank()
    u.integrate_apics()
    feats = list(u.get_typological_features_array())
    langs = list(u.get_typological_languages_array())
    data  = u.get_typological_data_array()

    all_feat_names = (["S_CLAUSAL_NOMINAL_OBJ_POS"] +
                      [f for pair in GRAMBANK_POS_FEATS.values() for f in pair])
    feat_idx = {f: feats.index(f) for f in all_feat_names if f in feats}

    def _src(row, feat, src_idx):
        if feat not in feat_idx:
            return None
        raw = row[feat_idx[feat]]
        try:
            if isinstance(raw, (list, np.ndarray)):
                v = float(list(raw)[src_idx])
            else:
                v = float(raw)
            return v if v != -1.0 else None
        except (ValueError, TypeError):
            return None

    def _pair_lr(row, feat_l, feat_r, src_idx):
        vl = _src(row, feat_l, src_idx)
        vr = _src(row, feat_r, src_idx)
        if vl == 1.0 and vr != 1.0: return "L"
        if vr == 1.0 and vl != 1.0: return "R"
        if vl == 1.0 and vr == 1.0: return "N"  # NoDominant
        return "?"                               # NoData

    glotto_rows = {gc: data[li] for li, gc in enumerate(langs)}

    _nodata = {"gb135": "NoData",
               "pos2_comp": "?", "pos3_pp": "?", "pos4_np": "?", "pos5_rel": "?",
               "apics_pos2_comp": "?", "apics_pos3_pp": "?", "apics_pos4_np": "?", "apics_pos5_rel": "?"}
    result = {}
    for iso in iso_codes:
        gc = iso_to_glotto.get(iso)
        if not gc or gc not in glotto_rows:
            result[iso] = dict(_nodata); continue

        row = glotto_rows[gc]

        v = _src(row, "S_CLAUSAL_NOMINAL_OBJ_POS", GRAMBANK_SRC_IDX)
        gb135 = "Yes" if v == 1.0 else ("No" if v == 0.0 else "NoData")

        pos = {"gb135": gb135}
        for pos_name, (feat_l, feat_r) in GRAMBANK_POS_FEATS.items():
            pos[pos_name]              = _pair_lr(row, feat_l, feat_r, GRAMBANK_SRC_IDX)
            pos[f"apics_{pos_name}"]   = _pair_lr(row, feat_l, feat_r, APICS_SRC_IDX)

        result[iso] = pos
    return result


# ── Raw Grambank direct lookup ─────────────────────────────────────────────
def load_grambank_raw(iso_codes):
    """Read grambank_data.csv by glottocode; returns {iso: {pos: 'L'/'R'}} for
    positions where exactly one of the L/R features is 1. Skips NoDominant (both=1)
    and NoData (both=0/-1). Does not go through URIEL+'s source-index filter."""
    iso_to_glotto = {}
    with open(URIELPLUS_GLOTTOCODE_MAP) as f:
        for r in csv.DictReader(f):
            iso_to_glotto[r["code"]] = r["glottocode"]

    if not GRAMBANK_RAW_CSV.exists():
        print("  grambank_data.csv not found, skipping raw lookup")
        return {}

    df = pd.read_csv(GRAMBANK_RAW_CSV)
    gc_to_row = {row["code"]: row for _, row in df.iterrows()}

    result = {}
    for iso in iso_codes:
        gc = iso_to_glotto.get(iso)
        if not gc or gc not in gc_to_row:
            continue
        row = gc_to_row[gc]
        entry = {}
        for pos, (feat_l, feat_r) in GB_RAW_FEATS.items():
            vl = row.get(feat_l, -1)
            vr = row.get(feat_r, -1)
            if vl == 1 and vr != 1:
                entry[pos] = "L"
            elif vr == 1 and vl != 1:
                entry[pos] = "R"
        if entry:
            result[iso] = entry

    print(f"  Raw Grambank: {len(result)} langs with ≥1 L/R fill")
    return result


# ── Encoding ───────────────────────────────────────────────────────────────
LR = {
    # Pos 1 VP_Comp
    "OV": "L", "VO": "R",
    # Pos 2 Comp  (subordinator position)
    "CompFinal":   "L",   # subordinator after clause  → head-final
    "CompInitial": "R",   # subordinator before clause → head-initial
    # Pos 3 PP
    "Postposition": "L", "Preposition": "R",
    # Pos 4 NP
    "AdjN": "L", "NAdj": "R",
    # Pos 5 Rel
    "RelN": "L", "NRel": "R",
}

def to_lr(val):
    if val == "NoDominant":
        return "N"
    return LR.get(val, "?")

def make_label(base, positions):
    """Build [Base]-[P1P2P3P4P5]; None if base is unknown."""
    if base not in BASE_ORDERS:
        return None
    return f"{base}-{''.join(positions)}"


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    flores_codes = list(FLORES_NAMES.keys())
    iso_codes    = [fc.split("_")[0] for fc in flores_codes]
    unique_isos  = sorted(set(iso_codes))

    # 1. lang2vec
    l2v_feats = l2v.get_features(unique_isos, "syntax_average", header=True)

    # 2. WALS fallback
    wals = load_wals()

    # 3. URIEL+ / Grambank
    grambank = load_grambank(unique_isos)

    # 4. Raw Grambank CSV (direct glottocode lookup, bypasses URIEL+ source-index filter)
    gb_raw = load_grambank_raw(unique_isos)

    rows = []
    for fc, iso in zip(flores_codes, iso_codes):
        vec = l2v_feats.get(iso, {})
        w   = wals.get(iso, {})

        # ── Base word order ──────────────────────────────────────────────
        base, base_dist = _dominant_order(vec)
        base_src = "lang2vec"
        if base == "NoData" and "81A" in w:
            base = w["81A"]; base_dist = ""; base_src = "wals"
        if base in ("NoData", "NoDominant") and fc in MANUAL_BASE_ORDER:
            base = MANUAL_BASE_ORDER[fc]; base_dist = ""; base_src = "manual"

        # ── Pos 1: VP_Comp (OV/VO order, validated by GB135) ────────────
        ov = _pair(vec, 9, 8, "OV", "VO")
        ov_src = "lang2vec"
        if ov == "NoData" and "83A" in w:
            ov = w["83A"]; ov_src = "wals"
        gb = grambank[iso]["gb135"]
        # if GB135=No, the OV proxy is unreliable → mark as unknown
        pos1 = "?" if gb == "No" else to_lr(ov)
        # NoDominant OV/VO (both attested) is an artifact of free nominal placement,
        # not a reliable indicator of VP_Comp direction → treat as unknown
        if pos1 == "N":
            pos1 = "?"
        # manual override for languages where VP_Comp direction is known but OV/VO is free
        # applies even when GB135=No: MANUAL_POS1 entries are direct knowledge of VP_Comp,
        # not derived from the OV/VO proxy
        if pos1 == "?" and iso in MANUAL_POS1:
            pos1 = MANUAL_POS1[iso]; ov_src = "manual"

        # ── Pos 2: Comp (subordinator position) ─────────────────────────
        comp = _pair(vec, 82, 81, "CompFinal", "CompInitial")
        comp_src = "lang2vec"
        if comp == "NoData" and "94A" in w:
            comp = w["94A"]; comp_src = "wals"
        pos2 = to_lr(comp)
        if pos2 == "?" and grambank[iso]["pos2_comp"] != "?":
            pos2 = grambank[iso]["pos2_comp"]; comp_src = "grambank"
        if pos2 == "?" and grambank[iso]["apics_pos2_comp"] != "?":
            pos2 = grambank[iso]["apics_pos2_comp"]; comp_src = "apics"

        # ── Pos 3: PP ────────────────────────────────────────────────────
        adp = _pair(vec, 21, 20, "Postposition", "Preposition")
        adp_src = "lang2vec"
        if adp == "NoData" and "85A" in w:
            adp = w["85A"]; adp_src = "wals"
        pos3 = to_lr(adp)
        if pos3 == "?" and grambank[iso]["pos3_pp"] != "?":
            pos3 = grambank[iso]["pos3_pp"]; adp_src = "grambank"
        if pos3 == "?" and grambank[iso]["apics_pos3_pp"] != "?":
            pos3 = grambank[iso]["apics_pos3_pp"]; adp_src = "apics"

        # ── Pos 4: NP ────────────────────────────────────────────────────
        adj = _pair(vec, 24, 25, "AdjN", "NAdj")
        adj_src = "lang2vec"
        if adj == "NoData" and "87A" in w:
            adj = w["87A"]; adj_src = "wals"
        pos4 = to_lr(adj)
        if pos4 == "?" and grambank[iso]["pos4_np"] != "?":
            pos4 = grambank[iso]["pos4_np"]; adj_src = "grambank"
        if pos4 == "?" and grambank[iso]["apics_pos4_np"] != "?":
            pos4 = grambank[iso]["apics_pos4_np"]; adj_src = "apics"

        # ── Pos 5: Rel ───────────────────────────────────────────────────
        rel = _triple(vec, 32, 33, 34, "RelN", "NRel", "RelAroundN")
        rel_src = "lang2vec"
        if rel == "NoData" and "90A" in w:
            rel = w["90A"]; rel_src = "wals"
        pos5 = to_lr(rel)
        # tag unmapped WALS rel types so manual overrides carry the origin
        if pos5 == "?" and rel_src == "wals":
            rel_src = f"wals_{rel.lower()}"  # e.g. wals_correlative, wals_internallyheaded
        if pos5 == "?" and grambank[iso]["pos5_rel"] != "?":
            pos5 = grambank[iso]["pos5_rel"]; rel_src = "grambank"
        if pos5 == "?" and grambank[iso]["apics_pos5_rel"] != "?":
            pos5 = grambank[iso]["apics_pos5_rel"]; rel_src = "apics"

        # ── Manual pos2-pos5 fallback ────────────────────────────────────
        # Fills remaining '?' and corrects known lang2vec errors.
        # Grambank/WALS/APiCS sources are left untouched.
        man = MANUAL_POSITIONS.get(iso, {})
        if "pos2" in man and (pos2 in ("?", "N") or comp_src == "lang2vec"):
            pos2 = man["pos2"]; comp_src = "manual"
        if "pos3" in man and (pos3 in ("?", "N") or adp_src == "lang2vec"):
            pos3 = man["pos3"]; adp_src = "manual"
        if "pos4" in man and (pos4 in ("?", "N") or adj_src == "lang2vec"):
            pos4 = man["pos4"]; adj_src = "manual"
        if "pos5" in man and (pos5 in ("?", "N") or rel_src == "lang2vec"):
            pos5 = man["pos5"]
            rel_src = f"manual_{rel_src.split('_', 1)[1]}" if rel_src.startswith("wals_") else "manual"

        # ── Pos1 from base WO (pos1 is VO/OV order, same axis as base WO) ─────
        # Only when GB135 != No (clausal objects pattern with nominals).
        if gb != "No" and pos1 == "?" and base in WO_TO_POS1:
            pos1 = WO_TO_POS1[base]; ov_src = "base_wo"

        # ── Raw Grambank fallback (direct CSV, bypasses URIEL+ source filter) ──
        raw = gb_raw.get(iso, {})
        if pos2 == "?" and "pos2_comp" in raw:
            pos2 = raw["pos2_comp"]; comp_src = "grambank_raw"
        if pos3 == "?" and "pos3_pp" in raw:
            pos3 = raw["pos3_pp"]; adp_src = "grambank_raw"
        if pos4 == "?" and "pos4_np" in raw:
            pos4 = raw["pos4_np"]; adj_src = "grambank_raw"
        if pos5 == "?" and "pos5_rel" in raw:
            pos5 = raw["pos5_rel"]; rel_src = "grambank_raw"

        label = make_label(base, [pos1, pos2, pos3, pos4, pos5])

        rows.append({
            "flores_code": fc,
            "iso_code": iso,
            "language_name": FLORES_NAMES[fc],
            # encoding
            "label": label,
            "base_word_order": base,
            "pos1_vp_comp": pos1,
            "pos2_comp": pos2,
            "pos3_pp": pos3,
            "pos4_np": pos4,
            "pos5_rel": pos5,
            # raw values (for inspection)
            "raw_base_dist": base_dist,
            "raw_ov_order": ov,
            "raw_comp_order": comp,
            "raw_adp_order": adp,
            "raw_adj_order": adj,
            "raw_rel_order": rel,
            # sources
            "src_base": base_src,
            "src_pos1": ov_src,
            "src_pos2": comp_src,
            "src_pos3": adp_src,
            "src_pos4": adj_src,
            "src_pos5": rel_src,
            "gb135_clausal_same_as_nominal": gb,
        })

    df = pd.DataFrame(rows)

    total = len(df)
    labeled  = df["label"].notna().sum()
    full     = df["label"].str.match(r"[A-Z]+-[LR]{5}$", na=False).sum()
    has_n    = df["label"].str.contains("N", na=False).sum()
    has_q    = df["label"].str.contains(r"\?", na=False).sum()
    print(f"FLORES-200 encoding: {total} entries")
    print(f"  label assigned : {labeled} ({100*labeled/total:.0f}%)")
    print(f"    fully L/R    : {full} ({100*full/total:.0f}%)")
    print(f"    has N (NoDominant position) : {has_n}")
    print(f"    has ? (NoData position)     : {has_q}")
    print(f"  no label       : {total - labeled} (base order unknown)")

    print("\nLabel distribution (top 20):")
    for lbl, n in df["label"].value_counts().head(20).items():
        print(f"  {str(lbl):<15s} {n:>4d}")

    print(f"\nGB135 coverage: {(df['gb135_clausal_same_as_nominal']!='NoData').sum()}/{total}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
