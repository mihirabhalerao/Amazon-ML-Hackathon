# ============================================================
# STAGE 0: PREPROCESSING + STAGE 1: COUNTRY PARTITIONING
# Amazon ML Challenge — Business Entity Resolution
#
# Merge decision log (normalize_sample.py vs normalization.py):
#
#   KEPT from normalize_sample:
#     - canonicalize_dotted_legal_terms()  — handles "P.V.T", "L.L.C" etc.
#     - unicode_safe_punctuation_to_spaces() — unicode-aware, handles
#       non-ASCII punctuation that NOISE_CHARS regex misses
#     - INDIA_STATE_MAP + INDIA_NATIVE_STATE_TOKEN_MAP — full India state
#       abbreviation coverage (original had US only)
#     - _abbreviation_allowed_position() — position-aware state abbrev guard
#       prevents "HP" in "Opp HP Petrol Bunk" from being read as Himachal
#     - find_state_position() — positional state anchoring for postal code
#     - extract_postal_code() — state-anchored, position-aware extraction
#       (original used a naive 40%/60% tail heuristic)
#     - transliterate_address() run-based char-level approach — more
#       accurate than token-level for Devanagari that spans word boundaries
#     - ITRANS target scheme + maybe_use_dravidian_variant — better
#       Dravidian (Tamil/Telugu/Kannada/Malayalam) phonetic output
#     - name_transliterated_no_suffix column
#     - Oriya script range (0x0B00-0x0B7F) missing from original
#     - stratified_sample_from_tsv() — reservoir sampling (not in original)
#     - APPLY_OCR_CORRECTION flag (conservative default=False)
#
#   KEPT from normalization.py:
#     - base_clean() structure: M/s strip, & → and, apostrophe delete,
#       REPEATED_DASH, DOMAIN_SUFFIXES regex (mid-string matches)
#     - dedup_consecutive() / dedup_all_tokens()
#     - fix_ocr() — mixed-token-only digit correction
#     - nfkd_aggressive() — drops non-ASCII post-IAST/ITRANS
#     - full_latin_pipeline() — ordered composition of all Latin steps
#     - preprocess_record() / preprocess_dataframe() — master preprocessor
#     - partition_by_country() — open-set, France-safe country split
#     - save_processed_data() / load_processed() / load_partition() —
#       full I/O layer including JSON serialisation of list columns
#     - SUFFIX_CANON with IAST transliteration aliases (prāiveṭa, etc.)
#
#   RESOLVED CONFLICTS:
#     - Legal suffix handling: normalize_sample canonicalises before
#       stripping; normalization strips naively. MERGED: use the ordered
#       approach (canon → strip) and add dotted-form pre-pass from sample.
#     - & → and: normalization.py bug (missing). Fixed here with explicit
#       re.sub before NOISE_CHARS, consistent with both designs' intent.
#     - Apostrophe: normalization.py deleted via NOISE_CHARS (left space).
#       Fixed: explicit delete before NOISE_CHARS, same as sample intent.
#     - State abbrev for India: original only had US_STATE_MAP. Added
#       INDIA_STATE_MAP; position guard prevents mid-address false matches.
#     - Postal code: replaced tail-heuristic with state-anchored extraction.
#     - Transliteration target: switched from IAST to ITRANS for names
#       (ITRANS produces cleaner ASCII without needing nfkd_aggressive).
#       nfkd_aggressive retained as belt-and-suspenders cleanup pass.
# ============================================================

import os
import re
import json
import random
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
from indic_transliteration import sanscript


# ============================================================
# CONSTANTS & CONFIG
# ============================================================

APPLY_OCR_CORRECTION = False   # conservative default; set True to enable

DOMAIN_SUFFIXES_RE = re.compile(
    r'\.(com|net|org|in|co\.in|biz|info|edu|gov|io|www)\b',
    flags=re.IGNORECASE,
)

NOISE_CHARS = re.compile(r"[\[\]#\(\)\+@\\|,\u2018\u2019`]")

REPEATED_DASH = re.compile(r'-{2,}')

NUMERIC_HEAVY = re.compile(r'^\d[\d\-/]+$')

PIN_PATTERN = re.compile(r'\b([1-9]\d{5})\b')   # Indian PIN: non-zero first digit
ZIP_PATTERN = re.compile(r'\b(\d{5})(?:-\d{4})?\b')

OCR_MAP = str.maketrans({"6": "g", "0": "o", "5": "s", "1": "l"})

MULTI_SPACE = re.compile(r'\s+')

# Street suffixes that look like US state abbreviations — never expand these
STREET_SUFFIXES = {
    "ct", "st", "ave", "blvd", "dr", "ln", "rd", "pl", "ter",
    "hwy", "pkwy", "cir", "trl", "way", "sq", "crst", "loop", "fl",
}

NULL_STRINGS = {"", "null", "nan", "none", "n/a", "na", "-", "--"}


# ============================================================
# LEGAL SUFFIX TABLES
# ============================================================

# Dotted/spaced legal term normalisation (runs BEFORE tokenisation)
# e.g. "P.V.T" → "pvt", "L.L.C." → "llc", "Pvt. Ltd." → "private limited"
_DOTTED_LEGAL = [
    (re.compile(r'\bl\s*[\.\s]+\s*l\s*[\.\s]+\s*p\.?\b', re.IGNORECASE), 'llp'),
    (re.compile(r'\bl\s*[\.\s]+\s*l\s*[\.\s]+\s*c\.?\b', re.IGNORECASE), 'llc'),
    (re.compile(r'\bl\s*[\.\s]+\s*p\.?\b',               re.IGNORECASE), 'lp'),
    (re.compile(r'\bp\s*[\.\s]+\s*v\s*[\.\s]+\s*t\.?\b', re.IGNORECASE), 'pvt'),
    (re.compile(r'\bpvt\.?\s+ltd\.?\b',                  re.IGNORECASE), 'private limited'),
    (re.compile(r'\bprivate\s+limited\b',                 re.IGNORECASE), 'private limited'),
    (re.compile(r'\bltd\.?\b',                            re.IGNORECASE), 'ltd'),
    (re.compile(r'\bcorp\.?\b',                           re.IGNORECASE), 'corp'),
    (re.compile(r'\binc\.?\b',                            re.IGNORECASE), 'inc'),
]

# Token-level canonical forms
SUFFIX_CANON = {
    "corporation":   "corp",
    "incorporated":  "inc",
    "company":       "co",
    "limited":       "ltd",
    "and":           "and",
    "&":             "and",
    "associates":    "assoc",
    "group":         "group",
    "holdings":      "holdings",
    "ventures":      "ventures",
    "solutions":     "solutions",
    "services":      "services",
    "enterprises":   "enterprises",
    "international": "intl",
    "technologies":  "tech",
    "technology":    "tech",
    "private":       "pvt",
    "pvt":           "pvt",
    "llp":           "llp",
    "llc":           "llc",
    "lp":            "lp",
    "plc":           "plc",
    "pllc":          "llc",
    # IAST/ITRANS transliteration aliases
    "praiveta":      "pvt",
    "prāiveta":      "pvt",
    "prāiveṭa":      "pvt",
    "limiteda":      "ltd",
    "limiṭeḍa":      "ltd",
    "limiteḍa":      "ltd",
    "elaelapī":      "llp",
    "elaelapi":      "llp",
    "li":            "",
    "prā":           "",
    "pra":           "",
}

SUFFIX_REMOVE = {
    "pvt", "ltd", "llc", "llp", "lp", "inc", "corp", "co", "plc", "pc",
    "assoc", "public", "partners", "foundation", "trust", "society",
}

ADDRESS_NOISE = {
    "unit", "floor", "flat", "no", "plot", "building", "block", "sector",
    "road", "street", "avenue", "drive", "lane", "near", "opp", "behind",
    "opposite", "rd", "st", "ave", "dr", "ln", "blvd", "hwy", "highway",
    "apt", "suite", "ste", "fl", "bldg", "dept", "po", "box", "pob",
    "and", "the", "of", "first", "second", "third", "ground",
    "upper", "lower", "main", "new", "old",
    "east", "west", "north", "south",
    "only", "null", "true", "false", "none", "na", "nil",
    "room", "rm", "apartment", "number",
}

# Labels that introduce a street number token (e.g. "No 25", "Plot 358")
NUMBER_LABELS = {"no", "number", "plot", "flat", "unit", "suite", "ste",
                 "apt", "apartment", "door"}


# ============================================================
# STATE MAPS
# ============================================================

US_STATE_MAP = {
    "al": "alabama",        "ak": "alaska",         "az": "arizona",
    "ar": "arkansas",       "ca": "california",     "co": "colorado",
    "ct": "connecticut",    "de": "delaware",       "fl": "florida",
    "ga": "georgia",        "hi": "hawaii",         "id": "idaho",
    "il": "illinois",       "in": "indiana",        "ia": "iowa",
    "ks": "kansas",         "ky": "kentucky",       "la": "louisiana",
    "me": "maine",          "md": "maryland",       "ma": "massachusetts",
    "mi": "michigan",       "mn": "minnesota",      "ms": "mississippi",
    "mo": "missouri",       "mt": "montana",        "ne": "nebraska",
    "nv": "nevada",         "nh": "new hampshire",  "nj": "new jersey",
    "nm": "new mexico",     "ny": "new york",       "nc": "north carolina",
    "nd": "north dakota",   "oh": "ohio",           "ok": "oklahoma",
    "or": "oregon",         "pa": "pennsylvania",   "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota",   "tn": "tennessee",
    "tx": "texas",          "ut": "utah",           "vt": "vermont",
    "va": "virginia",       "wa": "washington",     "wv": "west virginia",
    "wi": "wisconsin",      "wy": "wyoming",        "dc": "district of columbia",
}

INDIA_STATE_MAP = {
    "ap": "andhra pradesh",     "ar": "arunachal pradesh",
    "as": "assam",              "br": "bihar",
    "cg": "chhattisgarh",       "ga": "goa",
    "gj": "gujarat",            "hr": "haryana",
    "hp": "himachal pradesh",   "jh": "jharkhand",
    "ka": "karnataka",          "kl": "kerala",
    "mp": "madhya pradesh",     "mh": "maharashtra",
    "mn": "manipur",            "ml": "meghalaya",
    "mz": "mizoram",            "nl": "nagaland",
    "od": "odisha",             "or": "odisha",
    "pb": "punjab",             "rj": "rajasthan",
    "sk": "sikkim",             "tn": "tamil nadu",
    "ts": "telangana",          "tg": "telangana",
    "tr": "tripura",            "up": "uttar pradesh",
    "uk": "uttarakhand",        "ut": "uttarakhand",
    "wb": "west bengal",
    # Union Territories
    "an": "andaman and nicobar islands",
    "ch": "chandigarh",         "dl": "delhi",
    "jk": "jammu and kashmir",  "la": "ladakh",
    "ld": "lakshadweep",        "py": "puducherry",
    "dn": "dadra and nagar haveli and daman and diu",
    "dd": "dadra and nagar haveli and daman and diu",
}

US_STATE_FULL_SET    = set(US_STATE_MAP.values())
INDIA_STATE_FULL_SET = set(INDIA_STATE_MAP.values())

# Native-script Indian state names — matched before transliteration
INDIA_NATIVE_STATE_MAP = {
    "ఆంధ్రప్రదేశ్": "andhra pradesh",
    "आंध्र प्रदेश": "andhra pradesh",
    "अरुणाचल प्रदेश": "arunachal pradesh",
    "অসম": "assam",
    "বিহার": "bihar",         "बिहार": "bihar",
    "छत्तीसगढ़": "chhattisgarh",
    "ગુજરાત": "gujarat",
    "हरियाणा": "haryana",     "ਹਰਿਆਣਾ": "haryana",
    "हिमाचल प्रदेश": "himachal pradesh",
    "झारखंड": "jharkhand",
    "ಕರ್ನಾಟಕ": "karnataka",
    "കേരളം": "kerala",
    "मध्य प्रदेश": "madhya pradesh",
    "महाराष्ट्र": "maharashtra",
    "মণিপুর": "manipur",
    "ओडिशा": "odisha",        "ଓଡ଼ିଶା": "odisha",
    "ਪੰਜਾਬ": "punjab",        "पंजाब": "punjab",
    "राजस्थान": "rajasthan",
    "தமிழ்நாடு": "tamil nadu",
    "తెలంగాణ": "telangana",
    "त्रिपुरा": "tripura",
    "उत्तर प्रदेश": "uttar pradesh",
    "उत्तराखंड": "uttarakhand",
    "পশ্চিমবঙ্গ": "west bengal",
    "पश्चिम बंगाल": "west bengal",
    "दिल्ली": "delhi",
    "जम्मू और कश्मीर": "jammu and kashmir",
    "ਚੰਡੀਗੜ੍ਹ": "chandigarh",
    "लद्दाख": "ladakh",
    "पुदुच्चेरी": "puducherry",
}

# Pre-normalised token tuples for fast matching
INDIA_NATIVE_STATE_TOKEN_MAP = {
    tuple(
        "".join(c for c in unicodedata.normalize("NFKD", part.lower())
                if not unicodedata.combining(c))
        for part in alias.split()
    ): canonical
    for alias, canonical in INDIA_NATIVE_STATE_MAP.items()
}


# ============================================================
# SCRIPT RANGES + TRANSLITERATION SCHEMES
# ============================================================

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali":    (0x0980, 0x09FF),
    "gurmukhi":   (0x0A00, 0x0A7F),
    "gujarati":   (0x0A80, 0x0AFF),
    "oriya":      (0x0B00, 0x0B7F),   # absent from original normalization.py
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
}

TRANSLITERATION_SCHEMES = {
    "devanagari": sanscript.DEVANAGARI,
    "bengali":    sanscript.BENGALI,
    "gurmukhi":   sanscript.GURMUKHI,
    "gujarati":   sanscript.GUJARATI,
    "oriya":      sanscript.ORIYA,
    "tamil":      sanscript.TAMIL,
    "telugu":     sanscript.TELUGU,
    "kannada":    sanscript.KANNADA,
    "malayalam":  sanscript.MALAYALAM,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_sources(
    train_dir: str = "dataset/train",
    test_dir:  str = "dataset/test",
) -> dict:
    data = {}
    for split, d in [("train", train_dir), ("test", test_dir)]:
        for src in ["source1", "source2", "source3"]:
            key  = f"{split}_{src}"
            path = f"{d}/{split}_{src}.tsv"
            df   = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            df.columns = df.columns.str.strip()
            data[key] = df
            print(f"  Loaded {key}: {len(df):>7} records")

    gt_path = f"{train_dir}/train_ground_truth.tsv"
    gt      = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    data["train_ground_truth"] = gt
    print(f"  Loaded train_ground_truth: {len(gt):>7} records")
    return data


# ============================================================
# STAGE 0A — BASE CLEANING
# ============================================================

def canonicalize_dotted_legal_terms(text: str) -> str:
    """Pre-tokenisation pass: normalise dotted/spaced legal abbreviations.
    'P.V.T' → 'pvt',  'L.L.C.' → 'llc',  'Pvt. Ltd.' → 'private limited'
    Must run on the raw string BEFORE splitting on dots/spaces.
    """
    for pattern, replacement in _DOTTED_LEGAL:
        text = pattern.sub(replacement, text)
    return text


def unicode_safe_punctuation_to_spaces(text: str) -> str:
    """Replace Unicode punctuation/symbols with spaces.
    Preserves Unicode letters, digits, and combining marks (needed for
    Indic scripts).  Strictly better than a byte-level regex for names
    that contain non-ASCII punctuation.
    """
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(" " if cat.startswith(("P", "S")) else ch)
    return "".join(out)


def base_clean(text: str) -> str:
    """
    Lowercase → M/s strip → dotted legal terms → & → and →
    apostrophe delete → domain suffixes → unicode punctuation →
    repeated dash → dot/dash split → collapse whitespace.
    """
    if not isinstance(text, str) or text.strip().lower() in NULL_STRINGS:
        return ""

    text = text.lower()

    # Strip Indian business prefix
    text = re.sub(r'\bm/s\b', '', text)
    text = re.sub(r'\bm\.s\.\b', '', text)

    # Normalise dotted legal terms BEFORE dots are split
    text = canonicalize_dotted_legal_terms(text)

    # & → and  (must come before NOISE_CHARS which would delete it)
    text = re.sub(r'\s*&\s*', ' and ', text)

    # Delete apostrophes (straight + curly) so "Candy's" → "candys"
    text = re.sub(r"['\u2018\u2019`]", "", text)

    # Remove domain suffixes before splitting on dots
    text = DOMAIN_SUFFIXES_RE.sub("", text)

    # Unicode-aware punctuation → space (handles non-ASCII punct)
    text = unicode_safe_punctuation_to_spaces(text)

    # Collapse repeated dashes that survive
    text = REPEATED_DASH.sub(" ", text)

    # Split remaining dots and dashes
    text = text.replace(".", " ").replace("-", " ")

    return MULTI_SPACE.sub(" ", text).strip()


# ============================================================
# STAGE 0B — CONSECUTIVE DEDUPLICATION
# ============================================================

def dedup_consecutive(text: str) -> str:
    """Remove immediately repeated tokens: 'Crestline Crestline' → 'Crestline'."""
    if not text:
        return text
    tokens, result = text.split(), []
    for tok in tokens:
        if not result or tok != result[-1]:
            result.append(tok)
    return " ".join(result)


def dedup_all_tokens(text: str) -> str:
    """Remove ALL duplicate tokens (order-preserving). Used for name_sorted."""
    tokens, seen, seen_set = text.split(), [], set()
    for t in tokens:
        if t not in seen_set:
            seen.append(t)
            seen_set.add(t)
    return " ".join(seen)


# ============================================================
# STAGE 0C — OCR CORRECTION  (optional, mixed tokens only)
# ============================================================

def fix_ocr(tokens: list[str]) -> list[str]:
    """Substitute digit/letter OCR confusions in mixed alphanumeric tokens only.
    Pure-digit tokens (phone, PIN) are never altered.
    Controlled by APPLY_OCR_CORRECTION flag.
    """
    if not APPLY_OCR_CORRECTION:
        return tokens
    out = []
    for tok in tokens:
        has_alpha = any(c.isalpha() for c in tok)
        has_digit = any(c.isdigit() for c in tok)
        out.append(tok.translate(OCR_MAP) if (has_alpha and has_digit) else tok)
    return out


# ============================================================
# STAGE 0D — LEGAL SUFFIX CANONICALISATION
# ============================================================

def fold_for_lookup(text: str) -> str:
    """NFKD decompose + drop combining marks. Used for suffix table lookups."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def canonicalize_legal_suffixes(tokens: list[str]) -> list[str]:
    return [SUFFIX_CANON.get(fold_for_lookup(t), t) for t in tokens]


def strip_legal_suffix_tokens(tokens: list[str]) -> list[str]:
    filtered = [t for t in tokens
                if fold_for_lookup(t) not in SUFFIX_REMOVE
                and not re.match(r'^\d+$', t)]
    # If everything was stripped (entity IS a suffix), fall back
    return filtered if filtered else tokens


# ============================================================
# STAGE 0E — NAME VARIANTS
# ============================================================

def generate_name_variants(name_clean: str) -> dict:
    tokens          = name_clean.split()
    no_suffix_tokens = strip_legal_suffix_tokens(tokens)
    sorted_str      = " ".join(sorted(no_suffix_tokens))
    return {
        "name_full":      name_clean,
        "name_no_suffix": " ".join(no_suffix_tokens),
        "name_sorted":    dedup_all_tokens(sorted_str),
        "name_first2":    " ".join(no_suffix_tokens[:2]) if len(no_suffix_tokens) >= 2 else " ".join(no_suffix_tokens),
    }


# ============================================================
# STAGE 0F — UNICODE NORMALISATION
# ============================================================

def nfkd_aggressive(text: str) -> str:
    """NFKD + drop combining marks + drop non-ASCII bytes.
    Converts IAST/ITRANS diacritics (ā→a, ṭ→t) to clean ASCII.
    """
    if not text:
        return text
    nfkd = unicodedata.normalize("NFKD", text)
    no_combining = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_combining.encode("ascii", errors="ignore").decode("ascii")


# ============================================================
# STAGE 0G — SCRIPT DETECTION + INDIC TRANSLITERATION
# ============================================================

def detect_script_type(text: str) -> str:
    """Return dominant Unicode script: 'latin', '<script_name>', or 'mixed'."""
    if not isinstance(text, str) or not text:
        return "latin"

    counts      = {s: 0 for s in SCRIPT_RANGES}
    latin_count = 0

    for ch in text:
        if not ch.isalpha():
            continue
        cp      = ord(ch)
        matched = False
        for script, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script] += 1
                matched = True
                break
        if not matched:
            latin_count += 1

    total_indic = sum(counts.values())
    if total_indic == 0:
        return "latin"
    if latin_count == 0:
        return max(counts, key=counts.get)
    return "mixed"


def transliterate_name(raw_name: str, script_type: str) -> str:
    """Transliterate an Indic-script name to ITRANS (clean ASCII output).
    Uses maybe_use_dravidian_variant for better Tamil/Telugu/Kannada/Malayalam.
    Returns empty string on failure — never raises.
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return ""
    if script_type == "latin":
        return ""

    source_scheme = TRANSLITERATION_SCHEMES.get(script_type)
    if source_scheme is None:
        return ""

    try:
        out = sanscript.transliterate(
            raw_name,
            source_scheme,
            sanscript.ITRANS,
            maybe_use_dravidian_variant="yes",
        )
        # Post-process: normalise punctuation that ITRANS may emit
        out = canonicalize_dotted_legal_terms(out)
        out = unicode_safe_punctuation_to_spaces(out)
        out = fold_for_lookup(out)
        # fold_for_lookup drops Mn (non-spacing) combining marks but keeps
        # Mc (spacing) marks such as U+0949 DEVANAGARI VOWEL SIGN CANDRA O
        # that NFKD does not decompose.  nfkd_aggressive drops them via
        # encode("ascii", errors="ignore"), guaranteeing ASCII output.
        out = nfkd_aggressive(out)
        out = out.lower()
        return MULTI_SPACE.sub(" ", out).strip()
    except Exception:
        return ""


def transliterate_address(raw_address: str) -> str:
    """Produce a transliterated lookup representation of an address.
    Processes character-level runs by script so Latin/digit spans are
    preserved exactly.  The original address is never modified.
    Returns empty string for null/missing input.
    """
    if (not isinstance(raw_address, str)
            or raw_address.strip().lower() in NULL_STRINGS):
        return ""

    # Fast-path: already pure ASCII
    try:
        raw_address.encode("ascii")
        return raw_address.lower()
    except UnicodeEncodeError:
        pass

    output, current_script, current_text = [], None, []

    def flush_run():
        nonlocal current_script, current_text
        if not current_text:
            return
        run = "".join(current_text)
        if current_script in TRANSLITERATION_SCHEMES:
            try:
                t = transliterate_name(run, current_script)
                if t:
                    run = t
            except Exception:
                pass
        output.append(run)
        current_text = []
        current_script = None

    for ch in raw_address:
        cp          = ord(ch)
        char_script = None
        for script, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                char_script = script
                break
        if char_script != current_script:
            flush_run()
            current_script = char_script
        current_text.append(ch)

    flush_run()

    out = "".join(output)
    out = unicode_safe_punctuation_to_spaces(out)
    out = fold_for_lookup(out)
    # Same Mc-category issue as transliterate_name: drop any surviving
    # non-ASCII bytes so address_transliterated is always ASCII-safe.
    out = out.encode("ascii", errors="ignore").decode("ascii")
    out = out.lower()
    return MULTI_SPACE.sub(" ", out).strip()


# ============================================================
# FULL LATIN PIPELINE
# ============================================================

def full_latin_pipeline(text: str) -> str:
    """Complete Latin normalisation pipeline. Order is intentional."""
    text   = base_clean(text)                       # 0A
    text   = dedup_consecutive(text)                # 0B
    tokens = text.split()
    tokens = fix_ocr(tokens)                        # 0C (flag-gated)
    tokens = canonicalize_legal_suffixes(tokens)    # 0D canon
    tokens = strip_legal_suffix_tokens(tokens)      # 0D strip
    text   = " ".join(tokens)
    text   = nfkd_aggressive(text)                  # 0F ASCII collapse
    return MULTI_SPACE.sub(" ", text).strip()


# ============================================================
# STAGE 0H — ADDRESS NORMALISATION
# ============================================================

def normalize_country(country) -> str:
    if not isinstance(country, str):
        return ""
    return country.strip().lower()


def _normalized_tokens(tokens: list[str]) -> list[str]:
    return [fold_for_lookup(t.lower()) for t in tokens]


def _find_native_india_state(tokens: list[str]) -> tuple[str, int]:
    """Match exact native-script Indian state/UT names (longest-match first)."""
    normalized = _normalized_tokens(tokens)
    aliases    = sorted(INDIA_NATIVE_STATE_TOKEN_MAP.items(),
                        key=lambda item: len(item[0]), reverse=True)
    for alias_tokens, canonical in aliases:
        width = len(alias_tokens)
        if width == 0 or width > len(normalized):
            continue
        for i in range(len(normalized) - width + 1):
            if tuple(normalized[i:i + width]) == alias_tokens:
                return canonical, i
    return "", -1


def _abbreviation_allowed_position(index: int, token_count: int) -> bool:
    """State abbreviations are ambiguous; only accept them at the start
    or within the final 3 tokens.  'HP' in 'Opp HP Petrol Bunk' is rejected.
    """
    return index == 0 or index >= max(0, token_count - 3)


def state_maps_for_country(country: str):
    c = normalize_country(country)
    if c == "us":
        return US_STATE_MAP, US_STATE_FULL_SET
    if c == "india":
        return INDIA_STATE_MAP, INDIA_STATE_FULL_SET
    # Open-set (France, etc.): no abbreviation map, union of full state sets
    return {}, US_STATE_FULL_SET | INDIA_STATE_FULL_SET


def normalize_state_token(tokens: list[str], country: str) -> str:
    country = normalize_country(country)

    # India: native-script exact match first
    if country == "india":
        native_state, _ = _find_native_india_state(tokens)
        if native_state:
            return native_state

    abbreviation_map, full_states = state_maps_for_country(country)
    normalized = _normalized_tokens(tokens)

    # Two-word full state names (scan from tail)
    for i in range(len(normalized) - 2, -1, -1):
        candidate = normalized[i] + " " + normalized[i + 1]
        if candidate in full_states:
            return candidate

    # One-word full state names
    for tok in reversed(normalized):
        if tok in full_states:
            return tok

    # Abbreviations — position-gated
    for i in range(len(normalized) - 1, -1, -1):
        if (normalized[i] in abbreviation_map
                and _abbreviation_allowed_position(i, len(normalized))):
            return abbreviation_map[normalized[i]]

    return ""


def find_state_position(tokens: list[str], country: str) -> int:
    country = normalize_country(country)

    if country == "india":
        _, native_position = _find_native_india_state(tokens)
        if native_position >= 0:
            return native_position

    abbreviation_map, full_states = state_maps_for_country(country)
    normalized = _normalized_tokens(tokens)

    for i in range(len(normalized) - 2, -1, -1):
        if normalized[i] + " " + normalized[i + 1] in full_states:
            return i

    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i] in full_states:
            return i

    for i in range(len(normalized) - 1, -1, -1):
        if (normalized[i] in abbreviation_map
                and _abbreviation_allowed_position(i, len(normalized))):
            return i

    return -1


def tokenize_address(raw_address: str) -> list[str]:
    cleaned = raw_address.lower()
    cleaned = re.sub(r"[,#;:./\\()\[\]]+", " ", cleaned)
    cleaned = MULTI_SPACE.sub(" ", cleaned).strip()
    return [t.strip(".,") for t in cleaned.split() if t.strip(".,")]


def extract_postal_code(
    tokens: list[str],
    country: str,
    state_position: int,
) -> tuple[str, str]:
    """State-anchored, position-aware postal code extraction.
    India: PIN (6 digits, non-zero first).
    US: ZIP (5 digits, context near state token preferred over address head).
    """
    country = normalize_country(country)

    if country == "india":
        pattern = re.compile(r'^[1-9]\d{5}$')
        candidate_indices = []
        if state_position >= 0:
            candidate_indices.extend(range(state_position + 1, len(tokens)))
            candidate_indices.extend(range(state_position - 1, -1, -1))
        else:
            candidate_indices.extend(range(len(tokens) - 1, -1, -1))
        seen = set()
        for idx in candidate_indices:
            if idx in seen:
                continue
            seen.add(idx)
            if pattern.fullmatch(tokens[idx]):
                return tokens[idx], ""
        return "", ""

    if country == "us":
        pattern = re.compile(r'^\d{5}(?:-\d{4})?$')
        # Prefer: ZIP immediately after state
        if state_position >= 0:
            for idx in range(state_position + 1, len(tokens)):
                if pattern.fullmatch(tokens[idx]):
                    return "", tokens[idx]
        # Then: ZIP immediately before state (small window)
        if state_position >= 0:
            for idx in range(state_position - 1,
                             max(-1, state_position - 4), -1):
                if idx == 0:
                    continue
                if pattern.fullmatch(tokens[idx]):
                    return "", tokens[idx]
        # Fallback: last few tokens only (avoid treating street num as ZIP)
        if state_position < 0:
            for idx in range(len(tokens) - 1,
                             max(-1, len(tokens) - 4), -1):
                if idx == 0:
                    continue
                if pattern.fullmatch(tokens[idx]):
                    return "", tokens[idx]
        return "", ""

    return "", ""


_STREET_NUM_PATTERN = re.compile(
    r'^(?:[a-z]+[-/]?)?\d+(?:[-/]\d+)*(?:[-/][a-z]+)?(?:[a-z])?$',
    re.IGNORECASE,
)


def extract_address_fields(raw_address: str, country: str) -> dict:
    """Extract structured address components. Call AFTER transliterate_address
    for non-Latin input so this function always sees ASCII-safe text.
    """
    NULL_RESULT = {
        "has_address":          False,
        "pin_code":             "",
        "zip_code":             "",
        "street_num":           "",
        "state_token":          "",
        "locality_tokens":      "",
        "address_transliterated": "",
    }

    if (not isinstance(raw_address, str)
            or raw_address.strip().lower() in NULL_STRINGS):
        return NULL_RESULT

    # Transliterated view (used for blocking; original preserved)
    addr_translit = transliterate_address(raw_address)

    tokens = tokenize_address(raw_address)
    if not tokens:
        return {**NULL_RESULT, "address_transliterated": addr_translit}

    # State recognition: try original first, fall back to transliterated
    state_token    = normalize_state_token(tokens, country)
    state_position = find_state_position(tokens, country)

    t_tokens = tokenize_address(addr_translit) if addr_translit else []
    if not state_token and t_tokens:
        state_token    = normalize_state_token(t_tokens, country)
        state_position = find_state_position(t_tokens, country)
        postal_tokens  = t_tokens
    else:
        postal_tokens = tokens

    pin_code, zip_code = extract_postal_code(
        postal_tokens, country, state_position
    )

    # If still no postal code, try transliterated fallback
    if not pin_code and not zip_code and t_tokens and postal_tokens is tokens:
        t_state_pos = find_state_position(t_tokens, country)
        pin_code, zip_code = extract_postal_code(t_tokens, country, t_state_pos)

    postal_values = {pin_code, zip_code} - {""}

    # Street number: regex match in first 12 tokens
    street_num = ""
    remaining  = [t for t in tokens if t not in postal_values]
    for i, tok in enumerate(remaining[:12]):
        candidate = tok.strip(".,;")
        lower     = candidate.lower()
        if lower in NUMBER_LABELS:
            continue
        if _STREET_NUM_PATTERN.fullmatch(lower):
            street_num = candidate
            break
        if (i > 0 and remaining[i - 1].lower() in NUMBER_LABELS
                and re.search(r'\d', candidate)):
            street_num = candidate
            break

    # Locality tokens: meaningful non-noise ASCII tokens
    country_norm = normalize_country(country)
    if country_norm == "us":
        state_words   = {w for s in US_STATE_FULL_SET for w in s.split()}
        abbrev_set    = set(US_STATE_MAP)
    elif country_norm == "india":
        state_words   = {w for s in INDIA_STATE_FULL_SET for w in s.split()}
        abbrev_set    = set(INDIA_STATE_MAP)
    else:
        state_words   = {w for s in (US_STATE_FULL_SET | INDIA_STATE_FULL_SET) for w in s.split()}
        abbrev_set    = set()

    # Indices of native-script state tokens to exclude
    native_indices = set()
    if country_norm == "india":
        _, native_start = _find_native_india_state(tokens)
        if native_start >= 0:
            norm = _normalized_tokens(tokens)
            for alias_tokens, _ in sorted(
                INDIA_NATIVE_STATE_TOKEN_MAP.items(),
                key=lambda x: len(x[0]), reverse=True
            ):
                w = len(alias_tokens)
                if (w > 0 and native_start + w <= len(norm)
                        and tuple(norm[native_start:native_start + w]) == alias_tokens):
                    native_indices = set(range(native_start, native_start + w))
                    break

    locality = []
    for orig_idx, tok in enumerate(tokens):
        if orig_idx in native_indices:
            continue
        if tok in postal_values or tok == street_num:
            continue
        lookup = fold_for_lookup(tok.lower())
        if (not lookup
                or lookup in ADDRESS_NOISE
                or lookup in abbrev_set
                or lookup in state_words
                or not tok.isascii()
                or NUMERIC_HEAVY.match(lookup)
                or lookup.isdigit()
                or len(lookup) <= 2):
            continue
        locality.append(tok)

    return {
        "has_address":            True,
        "pin_code":               pin_code,
        "zip_code":               zip_code,
        "street_num":             street_num,
        "state_token":            state_token,
        "locality_tokens":        " ".join(locality[:5]),
        "address_transliterated": addr_translit,
    }


# ============================================================
# STAGE 0 — MASTER RECORD PREPROCESSOR
# ============================================================

def preprocess_record(row: pd.Series) -> dict:
    entity_id   = str(row.get("entity_id", "")).strip()
    raw_name    = str(row.get("business_name", ""))
    raw_address = str(row.get("business_address", ""))
    country     = str(row.get("country", "")).strip()

    # Script detection on the original raw name
    script_type = detect_script_type(raw_name)

    # Domain-name flag (before cleaning)
    is_domain_name = bool(
        re.search(r'\.(com|net|org|in|co\.in)\b', raw_name, re.IGNORECASE)
    )

    # Name: Latin pipeline
    name_cleaned = full_latin_pipeline(raw_name)
    variants     = generate_name_variants(name_cleaned)

    # ASCII variant (Latin only)
    name_ascii = nfkd_aggressive(name_cleaned) if script_type == "latin" else ""

    # Transliteration (non-Latin records)
    if script_type not in ("latin", "unknown"):
        name_transliterated = transliterate_name(raw_name, script_type)
        if name_transliterated:
            trans_tokens              = name_transliterated.split()
            trans_no_suffix           = strip_legal_suffix_tokens(trans_tokens)
            name_transliterated_clean = " ".join(trans_no_suffix)
            # Sorted/first2 from transliterated form for Indic records
            name_sorted = dedup_all_tokens(" ".join(sorted(trans_no_suffix)))
            name_first2 = (
                " ".join(trans_no_suffix[:2]) if len(trans_no_suffix) >= 2
                else name_transliterated_clean
            )
        else:
            # transliterate_name returns "" for "mixed" script (Latin + Indic)
            # and for any script where the library raises.  Fall back to the
            # Latin-pipeline result so name_sorted / name_first2 are never empty.
            name_transliterated_clean = variants["name_no_suffix"]
            name_sorted               = variants["name_sorted"]
            name_first2               = variants["name_first2"]
    else:
        name_transliterated       = name_cleaned
        name_transliterated_clean = variants["name_no_suffix"]
        name_sorted               = variants["name_sorted"]
        name_first2               = variants["name_first2"]

    # Address
    addr = extract_address_fields(raw_address, country)

    return {
        "entity_id":                      entity_id,
        "country":                        country,
        "raw_name":                       raw_name,
        "raw_address":                    raw_address,
        # Name variants
        "name_full":                      name_cleaned if script_type == "latin" else raw_name,
        "name_no_suffix":                 variants["name_no_suffix"] if script_type == "latin" else name_transliterated_clean,
        "name_sorted":                    name_sorted,
        "name_first2":                    name_first2,
        "name_ascii":                     name_ascii,
        "name_transliterated":            name_transliterated,
        "name_transliterated_no_suffix":  name_transliterated_clean,
        # Metadata
        "script_type":                    script_type,
        "is_domain_name":                 is_domain_name,
        # Address components
        "has_address":                    addr["has_address"],
        "pin_code":                       addr["pin_code"],
        "zip_code":                       addr["zip_code"],
        "street_num":                     addr["street_num"],
        "locality_tokens":                addr["locality_tokens"],
        "state_token":                    addr["state_token"],
        "address_transliterated":         addr["address_transliterated"],
    }


def preprocess_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    print(f"  Preprocessing {source_label}: {len(df)} records ...")
    records = [preprocess_record(row) for _, row in df.iterrows()]
    result  = pd.DataFrame(records)
    print(f"  Done — {len(result)} rows, {len(result.columns)} columns")
    return result


# ============================================================
# STAGE 1 — COUNTRY PARTITIONING  (open-set safe)
# ============================================================

def partition_by_country(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> dict:
    all_countries = (
        set(s1["country"].unique())
        | set(s2["country"].unique())
        | set(s3["country"].unique())
    )
    all_countries.discard("")

    print(f"  Found {len(all_countries)} country partitions: {sorted(all_countries)}")
    partitions = {}

    for country in sorted(all_countries):
        partitions[country] = {
            "s1": s1[s1["country"] == country].reset_index(drop=True),
            "s2": s2[s2["country"] == country].reset_index(drop=True),
            "s3": s3[s3["country"] == country].reset_index(drop=True),
        }
        p = partitions[country]
        print(f"    [{country}]  S1={len(p['s1']):>6}  S2={len(p['s2']):>6}  S3={len(p['s3']):>6}")

    uk_s1 = s1[s1["country"] == ""].reset_index(drop=True)
    uk_s2 = s2[s2["country"] == ""].reset_index(drop=True)
    uk_s3 = s3[s3["country"] == ""].reset_index(drop=True)
    if len(uk_s1) + len(uk_s2) + len(uk_s3) > 0:
        partitions["__unknown__"] = {"s1": uk_s1, "s2": uk_s2, "s3": uk_s3}

    return partitions


# ============================================================
# STRATIFIED SAMPLING  (from normalize_sample.py)
# ============================================================

CHUNK_SIZE  = 100_000
RANDOM_STATE = 42


def get_country_counts(path: Path, chunk_size: int = CHUNK_SIZE) -> dict:
    counts = {}
    for chunk in pd.read_csv(path, sep="\t", usecols=["country"],
                             dtype={"country": "string"}, chunksize=chunk_size):
        for country, count in chunk["country"].value_counts(dropna=False).items():
            key = "__MISSING__" if pd.isna(country) else str(country)
            counts[key] = counts.get(key, 0) + int(count)
    return counts


def calculate_country_targets(country_counts: dict, sample_size: int) -> dict:
    total = sum(country_counts.values())
    if total == 0:
        return {}
    sample_size = min(sample_size, total)
    raw     = {c: v / total * sample_size for c, v in country_counts.items()}
    targets = {c: int(v) for c, v in raw.items()}
    remaining = sample_size - sum(targets.values())
    for country in sorted(country_counts, key=lambda c: raw[c] - targets[c],
                          reverse=True)[:remaining]:
        targets[country] += 1
    return {c: t for c, t in targets.items() if t > 0}


def stratified_sample_from_tsv(
    path: Path,
    sample_size: int,
    chunk_size: int = CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    print("  Pass 1/2: counting country distribution...")
    country_counts = get_country_counts(path, chunk_size)
    print("  Country counts:", country_counts)
    targets    = calculate_country_targets(country_counts, sample_size)
    print("  Target sample per country:", targets)
    reservoirs = {c: [] for c in targets}
    seen       = {c: 0 for c in targets}
    rngs       = {c: random.Random(random_state + i) for i, c in enumerate(targets)}

    print("  Pass 2/2: reservoir sampling...")
    for chunk_n, chunk in enumerate(
        pd.read_csv(path, sep="\t", dtype=str, chunksize=chunk_size)
    ):
        country_idx = chunk.columns.get_loc("country")
        for row in chunk.itertuples(index=False, name=None):
            country = row[country_idx]
            country = "__MISSING__" if pd.isna(country) else str(country)
            if country not in targets:
                continue
            seen[country] += 1
            target = targets[country]
            if len(reservoirs[country]) < target:
                reservoirs[country].append(row)
            else:
                j = rngs[country].randint(1, seen[country])
                if j <= target:
                    reservoirs[country][j - 1] = row
        if chunk_n % 10 == 0:
            print(f"    processed chunk {chunk_n:,}")

    columns = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    parts   = [pd.DataFrame(reservoirs[c], columns=columns)
               for c in targets if reservoirs[c]]
    if not parts:
        return pd.DataFrame(columns=columns)
    return (pd.concat(parts, ignore_index=True)
              .sample(frac=1, random_state=random_state)
              .reset_index(drop=True))


# ============================================================
# EXPORT UTILITIES
# ============================================================

def _save_tsv(df: pd.DataFrame, path: str, label: str) -> None:
    df.to_csv(path, sep="\t", index=False)
    print(f"    Saved {label}: {len(df):>7} rows  →  {path}")


def save_processed_data(
    processed:        dict,
    train_partitions: dict,
    test_partitions:  dict,
    output_dir:       str = "processed_data",
) -> None:
    stage0_dir       = os.path.join(output_dir, "stage0")
    stage1_train_dir = os.path.join(output_dir, "stage1", "train")
    stage1_test_dir  = os.path.join(output_dir, "stage1", "test")
    meta_dir         = os.path.join(output_dir, "meta")
    for d in [stage0_dir, stage1_train_dir, stage1_test_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n── Stage 0 export ──")
    stage0_keys = [
        "train_source1", "train_source2", "train_source3",
        "test_source1",  "test_source2",  "test_source3",
        "train_ground_truth",
    ]
    for key in stage0_keys:
        if key not in processed:
            continue
        _save_tsv(processed[key], os.path.join(stage0_dir, f"{key}.tsv"), key)

    print("\n── Stage 1 export ──")
    partition_summary = {}
    for split_label, partitions, out_dir in [
        ("train", train_partitions, stage1_train_dir),
        ("test",  test_partitions,  stage1_test_dir),
    ]:
        print(f"  {split_label.upper()}")
        partition_summary[split_label] = {}
        for country, srcs in partitions.items():
            slug   = re.sub(r"[^\w]", "_", country).strip("_").lower() or "unknown"
            counts = {}
            for src_key in ["s1", "s2", "s3"]:
                fname = f"{slug}_{src_key}.tsv"
                _save_tsv(srcs[src_key], os.path.join(out_dir, fname),
                          f"{split_label}/{country}/{src_key}")
                counts[src_key] = len(srcs[src_key])
            partition_summary[split_label][country] = counts

    summary_path = os.path.join(meta_dir, "partition_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(partition_summary, f, indent=2, ensure_ascii=False)

    FLAG_COLS   = ["entity_id", "country", "script_type", "is_domain_name", "has_address"]
    flag_frames = []
    for key in stage0_keys:
        if key == "train_ground_truth":
            continue
        df = processed.get(key)
        if df is None:
            continue
        sub = df[[c for c in FLAG_COLS if c in df.columns]].copy()
        sub["source_file"] = key
        flag_frames.append(sub)

    if flag_frames:
        flags_df = pd.concat(flag_frames, ignore_index=True)
        flags_df.to_csv(os.path.join(meta_dir, "subgroup_flags.tsv"),
                        sep="\t", index=False)
        print(f"\n  Subgroup breakdown:")
        print(f"    Non-ASCII names  : {(flags_df['script_type'] != 'latin').sum()}")
        print(f"    Missing address  : {(~flags_df['has_address']).sum()}")
        print(f"    Domain records   : {flags_df['is_domain_name'].sum()}")
        print(f"    Script types:\n{flags_df['script_type'].value_counts().to_string()}")

    print(f"\n  Export complete → {output_dir}/")


def load_processed(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_partition(
    country:    str,
    split:      str = "test",
    output_dir: str = "processed_data",
) -> dict:
    slug = re.sub(r"[^\w]", "_", country).strip("_").lower()
    base = os.path.join(output_dir, "stage1", split)
    out  = {}
    for src in ["s1", "s2", "s3"]:
        path = os.path.join(base, f"{slug}_{src}.tsv")
        if os.path.exists(path):
            out[src] = load_processed(path)
        else:
            print(f"  WARNING: {path} not found")
            out[src] = pd.DataFrame()
    return out


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    SAMPLE_MODE = True
    SAMPLE_SIZE = 10000

    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)
    data = load_sources()

    if SAMPLE_MODE:
        print(f"\n⚠️  SAMPLE MODE — {SAMPLE_SIZE} rows per source")
        for key in data:
            if key != "train_ground_truth":
                data[key] = data[key].head(SAMPLE_SIZE).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("STAGE 0: PREPROCESSING")
    print("=" * 60)
    processed = {}
    for key, df in data.items():
        if key == "train_ground_truth":
            processed[key] = df
        else:
            processed[key] = preprocess_dataframe(df, source_label=key)

    print("\n" + "=" * 60)
    print("STAGE 1: COUNTRY PARTITIONING")
    print("=" * 60)
    print("\n-- TRAIN --")
    train_partitions = partition_by_country(
        s1=processed["train_source1"],
        s2=processed["train_source2"],
        s3=processed["train_source3"],
    )
    print("\n-- TEST --")
    test_partitions = partition_by_country(
        s1=processed["test_source1"],
        s2=processed["test_source2"],
        s3=processed["test_source3"],
    )

    out_dir = "processed_data_sample" if SAMPLE_MODE else "processed_data"
    save_processed_data(
        processed=processed,
        train_partitions=train_partitions,
        test_partitions=test_partitions,
        output_dir=out_dir,
    )

    print(f"\n{'⚠️  SAMPLE COMPLETE' if SAMPLE_MODE else '✅ FULL RUN COMPLETE'}")
    print(f"Output: {out_dir}/")
    print("\nUseful name columns:")
    print("  name_full, name_no_suffix, name_sorted, name_first2,")
    print("  name_ascii, name_transliterated, name_transliterated_no_suffix,")
    print("  is_domain_name, script_type")
    print("\nUseful address columns:")
    print("  has_address, pin_code, zip_code, street_num,")
    print("  state_token, locality_tokens, address_transliterated")