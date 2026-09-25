# ============================================================
# STAGE 0: PREPROCESSING + STAGE 1: COUNTRY PARTITIONING
# Amazon ML Challenge — Business Entity Resolution
# ============================================================

import pandas as pd
import numpy as np
import re
import unicodedata
from indic_transliteration import sanscript
from indic_transliteration.detect import detect as detect_script
from metaphone import doublemetaphone

# ============================================================
# 0. LOAD RAW DATA
# ============================================================

def load_sources(train_dir="dataset/train", test_dir="dataset/test"):
    """Load all TSV files. Returns dict of DataFrames."""
    data = {}
    for split, d in [("train", train_dir), ("test", test_dir)]:
        for src in ["source1", "source2", "source3"]:
            key = f"{split}_{src}"
            path = f"{d}/{split}_{src}.tsv"
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
            df.columns = df.columns.str.strip()
            data[key] = df
            print(f"Loaded {key}: {len(df)} records")

    # Ground truth only for train
    gt = pd.read_csv(f"{train_dir}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    data["train_ground_truth"] = gt
    print(f"Loaded train_ground_truth: {len(gt)} records")
    return data


# ============================================================
# STAGE 0A: BASE CLEANING
# ============================================================

# Domain suffixes to strip from business names
DOMAIN_SUFFIXES = re.compile(
    r'\.(com|net|org|in|co\.in|biz|info|edu|gov|io|www)\b', flags=re.IGNORECASE
)

# Characters that are structural noise, not content
NOISE_CHARS = re.compile(r"[\[\]#\(\)\+@\\|,']")

# Repeated dashes used as separators
REPEATED_DASH = re.compile(r'-{2,}')


def base_clean(text: str) -> str:
    """
    0A: Lowercase, strip noise chars, strip domain suffixes,
        split domain-style names on . and -, collapse whitespace.
    """
    if not isinstance(text, str) or text.strip() in ("", "NULL", "NaN", "nan"):
        return ""

    # Lowercase
    text = text.lower()
    
    text = re.sub(r'\bm/s\b', '', text)
    text = re.sub(r'\bm\.s\.\b', '', text)

    # Strip domain suffixes before splitting (heassociates.com → heassociates)
    text = DOMAIN_SUFFIXES.sub("", text)

    # Remove noise characters
    text = NOISE_CHARS.sub(" ", text)

    # Replace repeated dashes with space
    text = REPEATED_DASH.sub(" ", text)
    
    # FIX 2: explicit apostrophe variants after regex pass
    text = text.replace("'", "").replace("'", "").replace("`", "")

    # Split remaining . and - (domain/hyphenated names → tokens)
    text = text.replace(".", " ").replace("-", " ")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# STAGE 0B: CONSECUTIVE WORD DEDUPLICATION
# ============================================================

def dedup_consecutive(text: str) -> str:
    """
    0B: Remove immediately repeated tokens.
    'Crestline Crestline Clean' → 'Crestline Clean'
    'FERRERO FERRERO DUKE' → 'FERRERO DUKE'
    Must run BEFORE suffix stripping.
    """
    if not text:
        return text
    tokens = text.split()
    result = []
    for tok in tokens:
        if not result or tok != result[-1]:
            result.append(tok)
    return " ".join(result)


# ============================================================
# STAGE 0C: OCR DIGIT → LETTER CORRECTION (NAME FIELD ONLY)
# ============================================================

# Only apply in name field. Never in address (digits are meaningful there).
OCR_MAP = str.maketrans({
    "6": "g",
    "0": "o",
    "5": "s",
    "1": "l",
})


def fix_ocr(text: str) -> str:
    """
    0C: Substitute common OCR digit/letter confusions.
    TRADIN6 → TRADING, N0se → Nose, 5uperior → Superior
    Only call on business_name, never on business_address.
    """
    if not text:
        return text
    return text.translate(OCR_MAP)


# ============================================================
# STAGE 0D: LEGAL SUFFIX CANONICALIZATION
# ============================================================

# Extensible dictionary — unrecognized tokens pass through unchanged
# This is open-set safe: French SARL/SAS/EURL just pass through
SUFFIX_CANON = {
    # English / US
    "corporation":  "corp",
    "incorporated": "inc",
    "company":      "co",
    "limited":      "ltd",
    "and":          "and",
    "&":            "and",
    "associates":   "assoc",
    "partners":     "partners",
    "group":        "group",
    "holdings":     "holdings",
    "ventures":     "ventures",
    "solutions":    "solutions",
    "services":     "services",
    "enterprises":  "enterprises",
    "international":"intl",
    "technologies": "tech",
    "technology":   "tech",
    # Indian
    "private":      "pvt",
    "pvt":          "pvt",
    # Keep "ltd" as-is, "limited" → "ltd" already above
    "llp":          "llp",
    "llc":          "llc",
    "lp":           "lp",
    "plc":          "plc",
    "pllc":         "llc",
}

SUFFIX_CANON.update({
    # IAST transliterations of pvt/private
    "praiveta":  "pvt",
    "prāiveta":  "pvt",
    "praiveṭa":  "pvt",
    "prāiveṭa":  "pvt",
    # IAST transliterations of limited/ltd
    "limiteda":  "ltd",
    "limiṭeḍa":  "ltd",
    "limiteḍa":  "ltd",
    # IAST transliterations of llp
    "elaelapī":  "llp",
    "elaelapi":  "llp",
    # IAST transliterations of ltd (Hindi लि / लिमिटेड short forms)
    "li":        "",    # drop standalone "li" (short for लिमिटेड)
    "prā":       "",    # drop standalone "prā" (short for प्राइवेट)
    "pra":       "",
})

# Legal suffix tokens to REMOVE entirely after canonicalization
# These add zero discriminative value
SUFFIX_REMOVE = {
    "pvt", "ltd", "llc", "llp", "lp", "inc", "corp",
    "co", "plc", "pc",
}

NUMERIC_HEAVY = re.compile(r'^\d[\d\-/]+$') 

STREET_SUFFIXES = {
    "ct", "st", "ave", "blvd", "dr", "ln", "rd", "pl", "ter",
    "hwy", "pkwy", "cir", "trl", "way", "sq", "crst", "loop",
}


def canonicalize_suffixes(text: str) -> str:
    """
    0D: Map known suffix variants to canonical form,
        then strip non-discriminative legal tokens.
    Unknown tokens (SARL, SAS, etc.) pass through unchanged.
    """
    if not text:
        return text
    tokens = text.split()
    canonical = [SUFFIX_CANON.get(tok, tok) for tok in tokens]
    filtered = [tok for tok in canonical if tok not in SUFFIX_REMOVE]
    return " ".join(filtered) if filtered else " ".join(canonical)

def dedup_all_tokens(text: str) -> str:
    """
    Remove ALL duplicate tokens (not just consecutive).
    Use for name_sorted generation only — preserves order otherwise.
    Applied after sorting so duplicates are adjacent and easy to remove.
    """
    tokens = text.split()
    seen = []
    seen_set = set()
    for t in tokens:
        if t not in seen_set:
            seen.append(t)
            seen_set.add(t)
    return " ".join(seen)

# ============================================================
# STAGE 0E: GENERATE FOUR NAME VARIANTS
# ============================================================

def generate_name_variants(name_clean: str) -> dict:
    """
    0E: From a single cleaned name string, produce four variants:
      - full:       full cleaned name (post 0A–0D)
      - no_suffix:  same as full (suffix already stripped in 0D)
      - sorted:     tokens sorted alphabetically (neutralizes word-order transpositions)
      - first2:     first two tokens only (for very noisy long names)
    """
    tokens = name_clean.split()
    sorted_str = " ".join(sorted(tokens))
    sorted_deduped = dedup_all_tokens(sorted_str)   # dedup after sort
    return {
        "name_full":      name_clean,
        "name_no_suffix": name_clean,
        "name_sorted":    sorted_deduped,
        "name_first2":    " ".join(tokens[:2]) if len(tokens) >= 2 else name_clean,
    }


# ============================================================
# STAGE 0F: NFKD DIACRITIC STRIPPING
# ============================================================

def nfkd_strip(text: str) -> str:
    """
    0F: Unicode NFKD decomposition + drop combining marks → ASCII.
    Handles: Sólar→Solar, Léarning→Learning, Nétwork→Network, Béque→Beque
    Safe for Latin-with-accents noise. Do NOT apply to genuine Indic scripts
    (detected in 0G) — this function is only called for Latin-script names.
    """
    if not text:
        return text
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in normalized if not unicodedata.combining(c))
    return ascii_text


# ============================================================
# STAGE 0G: SCRIPT DETECTION + INDIC TRANSLITERATION
# ============================================================

# Unicode ranges for Indic scripts
INDIC_RANGES = [
    (0x0900, 0x097F),   # Devanagari (Hindi, Marathi)
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0D00, 0x0D7F),   # Malayalam
    (0x0C80, 0x0CFF),   # Kannada
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
    (0x0980, 0x09FF),   # Bengali
]


def detect_script_type(text: str) -> str:
    """
    Detect dominant script in text.
    Returns: 'latin', 'devanagari', 'tamil', 'telugu',
             'malayalam', 'kannada', 'gujarati', 'gurmukhi',
             'bengali', or 'mixed'
    """
    if not text:
        return "latin"

    script_names = {
        (0x0900, 0x097F): "devanagari",
        (0x0B80, 0x0BFF): "tamil",
        (0x0C00, 0x0C7F): "telugu",
        (0x0D00, 0x0D7F): "malayalam",
        (0x0C80, 0x0CFF): "kannada",
        (0x0A80, 0x0AFF): "gujarati",
        (0x0A00, 0x0A7F): "gurmukhi",
        (0x0980, 0x09FF): "bengali",
    }

    counts = {name: 0 for name in script_names.values()}
    latin_count = 0

    for ch in text:
        cp = ord(ch)
        matched = False
        for (lo, hi), name in script_names.items():
            if lo <= cp <= hi:
                counts[name] += 1
                matched = True
                break
        if not matched and ch.isalpha():
            latin_count += 1

    total_indic = sum(counts.values())
    total = total_indic + latin_count

    if total == 0:
        return "latin"
    if total_indic == 0:
        return "latin"
    if latin_count == 0:
        return max(counts, key=counts.get)

    return "mixed"


def transliterate_indic(text: str, script_type: str) -> str:
    """
    0G: Transliterate Indic script text to Latin (IAST).
    For mixed scripts, attempt detection per-segment.
    For already-Latin text, returns text unchanged.
    """
    if script_type == "latin":
        return text
    if not text:
        return text

    SCRIPT_MAP = {
        "devanagari": sanscript.DEVANAGARI,
        "tamil":      sanscript.TAMIL,
        "telugu":     sanscript.TELUGU,
        "malayalam":  sanscript.MALAYALAM,
        "kannada":    sanscript.KANNADA,
        "gujarati":   sanscript.GUJARATI,
        "gurmukhi":   sanscript.GURMUKHI,
        "bengali":    sanscript.BENGALI,
    }

    if script_type == "mixed":
        # Try auto-detection via indic_transliteration
        try:
            detected = detect_script(text)
            if detected and detected in SCRIPT_MAP.values():
                return sanscript.transliterate(text, detected, sanscript.IAST)
        except Exception:
            pass
        return text  # fallback: return as-is for mixed

    src_script = SCRIPT_MAP.get(script_type)
    if src_script is None:
        return text

    try:
        transliterated = sanscript.transliterate(text, src_script, sanscript.IAST)
        return transliterated
    except Exception:
        return text  # never fail hard on transliteration


# ============================================================
# STAGE 0H: ADDRESS NORMALIZATION
# ============================================================

# Full US state name ↔ abbreviation map
US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Address noise words — remove before extracting locality tokens
ADDRESS_NOISE = {
    "unit", "floor", "flat", "no", "plot", "building", "block", "sector",
    "road", "street", "avenue", "drive", "lane", "near", "opp", "behind",
    "opposite", "rd", "st", "ave", "dr", "ln", "blvd", "hwy", "highway",
    "apt", "suite", "ste", "fl", "bldg", "dept", "po", "box", "pob",
    "c/o", "co", "h/o", "d/o", "w/o", "s/o", "and", "the", "of",
    "first", "second", "third", "ground", "upper", "lower", "main",
    "new", "old", "east", "west", "north", "south",
    "only", "null", "true", "false", "none", "na", "nil",
}

# Regex patterns for numeric extraction
PIN_PATTERN  = re.compile(r'\b(\d{6})\b')        # Indian PIN
ZIP_PATTERN  = re.compile(r'\b(\d{5})(?:-\d{4})?\b')  # US ZIP
STREET_NUM   = re.compile(r'\b(\d{1,5}[a-z]?)\b')     # Street/building number


def normalize_address(address: str, country: str = "") -> dict:
    """
    0H: Extract structured components from raw address string.
    Returns dict with:
      - address_missing: bool
      - pin_codes: list of 6-digit Indian PINs found
      - zip_codes: list of 5-digit US ZIPs found
      - street_num: first significant street/building number
      - locality_tokens: list of significant non-noise words
      - state_token: normalized state name (full form)
      - raw_normalized: lowercased, whitespace-collapsed address
    """
    NULL_VALS = {"", "null", "nan", "none", "n/a", "na", "-", "--"}

    if not isinstance(address, str) or address.strip().lower() in NULL_VALS:
        return {
            "address_missing":  True,
            "pin_codes":        [],
            "zip_codes":        [],
            "street_num":       "",
            "locality_tokens":  [],
            "state_token":      "",
            "raw_normalized":   "",
        }

    raw = address.lower()
    raw = re.sub(r"[#,./\\()\[\]]", " ", raw)   # add () and [] to address cleaning
    raw = re.sub(r"\s+", " ", raw).strip()

    # Extract PINs and ZIPs before removing digits
    pin_codes = PIN_PATTERN.findall(raw)
    # Avoid overlap: remove PIN matches before looking for ZIP
    raw_no_pin = PIN_PATTERN.sub(" ", raw)
    # FIX 4a: ZIP only from latter 60% of address string
    # Avoids capturing 5-digit street numbers (61573, 05131) as ZIPs
    addr_tail = raw_no_pin[int(len(raw_no_pin) * 0.4):]
    zip_codes = ZIP_PATTERN.findall(addr_tail)

    # Street number: first 1-5 digit token in first 60 chars
    street_num = ""
    early_text = raw[:60]
    sn_match = STREET_NUM.search(early_text)
    if sn_match:
        street_num = sn_match.group(1)

    # Normalize state abbreviations → full name
    # FIX 4b: state abbreviation only on pure standalone 2-letter alpha tokens
    # Prevents "341ND" → nd being parsed as North Dakota
    state_token = ""
    tokens_raw = raw.split()
    normalized_tokens = []
    for tok in tokens_raw:
        tok_clean = re.sub(r"[^a-z]", "", tok)
        is_us = country.strip().lower() in ("us", "united states")
        if (
            is_us                            # only expand for US records
            and tok_clean in US_STATE_MAP
            and len(tok_clean) == 2
            and tok.isalpha()
            and tok_clean not in STREET_SUFFIXES
        ):
            state_token = US_STATE_MAP[tok_clean]
            normalized_tokens.append(state_token)
        else:
            normalized_tokens.append(tok_clean if tok_clean else tok)
            
    normalized_tokens = [t for t in normalized_tokens if t != "null"]

    locality_tokens = [
        t for t in normalized_tokens
        if t not in ADDRESS_NOISE
        and len(t) > 3
        and not t.isdigit()
        and t.isascii()          # FIX: drop non-ASCII tokens from address
        and not NUMERIC_HEAVY.match(t)   # NEW: filter numeric-heavy tokens
    ]

    return {
        "address_missing":  False,
        "pin_codes":        pin_codes,
        "zip_codes":        zip_codes,
        "street_num":       street_num,
        "locality_tokens":  locality_tokens[:5],   # top 5 significant tokens
        "state_token":      state_token,
        "raw_normalized":   " ".join(normalized_tokens),
    }

def nfkd_aggressive(text: str) -> str:
    """
    NFKD strip that also handles IAST diacritics:
    ā→a, ī→i, ū→u, ṭ→t, ḍ→d, ṃ→m, ṅ→n, ś→s, ṣ→s, ḥ→h etc.
    Run this on transliterated output to get clean ASCII.
    """
    if not text:
        return text
    normalized = unicodedata.normalize("NFKD", text)
    # Drop all combining diacritical marks (Unicode category "Mn")
    ascii_text = "".join(
        c for c in normalized
        if not unicodedata.combining(c)
    )
    # Additionally map any remaining non-ASCII letters to closest ASCII
    # (handles ḷ, ṉ, ẓ etc. that NFKD alone doesn't fully resolve)
    result = ascii_text.encode("ascii", errors="ignore").decode("ascii")
    return result

def full_latin_pipeline(text: str) -> str:
    """
    Run the complete Latin normalization pipeline on any string.
    Used both for native Latin names AND for post-transliteration cleanup.
    Sequence matters — do not reorder.
    """
    text = base_clean(text)           # lowercase, strip noise, domain split
    text = dedup_consecutive(text)    # remove consecutive repeated tokens
    text = fix_ocr(text)              # digit→letter OCR fixes
    text = nfkd_aggressive(text)      # strip ALL diacritics including IAST
    text = canonicalize_suffixes(text) # suffix canon + removal
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fix_ocr(text: str) -> str:
    """
    Apply OCR digit→letter correction only to mixed tokens
    (tokens containing both letters and digits).
    Pure digit tokens (phone numbers, IDs) are left unchanged.
    """
    if not text:
        return text
    tokens = text.split()
    result = []
    for tok in tokens:
        has_alpha = any(c.isalpha() for c in tok)
        has_digit = any(c.isdigit() for c in tok)
        if has_alpha and has_digit:
            # Mixed token — apply OCR fix (TRADIN6, N0se, ART5)
            result.append(tok.translate(OCR_MAP))
        else:
            # Pure alpha or pure digit — leave unchanged
            result.append(tok)
    return " ".join(result)

# ============================================================
# ADDRESS TRANSLITERATION
# Handles mixed-script addresses where Indic state/city names
# appear inline with Latin text
# e.g. "Mumbai, महाराष्ट्र" → "Mumbai, maharastra"
# e.g. "Lucknow, उत्तर प्रदेश" → "Lucknow, uttar pradesa"
# ============================================================

def transliterate_address_token(token: str) -> str:
    """
    Attempt to transliterate a single address token if it contains
    non-Latin script. Returns ASCII-clean string.
    If token is already Latin/ASCII, returns it unchanged.
    """
    if not token:
        return token

    # Check if token contains any non-ASCII characters
    try:
        token.encode("ascii")
        return token  # already ASCII, skip transliteration
    except UnicodeEncodeError:
        pass

    # Detect script of this token
    script = detect_script_type(token)

    if script == "latin":
        # Has non-ASCII but detected as latin — just NFKD strip
        return nfkd_aggressive(token)

    # Transliterate to IAST Latin
    transliterated = transliterate_indic(token, script)

    # Then NFKD-strip the IAST diacritics to pure ASCII
    ascii_form = nfkd_aggressive(transliterated)

    # Clean up: lowercase, strip noise
    ascii_form = re.sub(r"[^a-z0-9\s]", "", ascii_form.lower())
    ascii_form = re.sub(r"\s+", " ", ascii_form).strip()

    return ascii_form


def transliterate_address(address: str) -> str:
    """
    Process a full address string that may contain mixed scripts.
    Splits on whitespace, transliterates each non-ASCII token
    individually, reassembles.

    Why token-by-token and not whole string:
    - Addresses mix Latin and Indic in the same field:
      "Mumbai, महाराष्ट्र" — "Mumbai" is Latin, "महाराष्ट्र" is Devanagari
    - Transliterating the whole string at once confuses the
      script detector since it sees mixed input
    - Token-level detection is more accurate and safer

    Returns a fully ASCII-clean address string.
    """
    if not isinstance(address, str) or not address.strip():
        return address

    # Check if transliteration is even needed
    try:
        address.encode("ascii")
        return address  # entirely ASCII already, skip
    except UnicodeEncodeError:
        pass  # contains non-ASCII, proceed

    # Split on whitespace, process each token
    tokens = address.split()
    result_tokens = []

    for tok in tokens:
        transliterated_tok = transliterate_address_token(tok)
        if transliterated_tok:
            result_tokens.append(transliterated_tok)

    return " ".join(result_tokens)


# ============================================================
# UPDATED normalize_address
# Now accepts pre-transliterated address string
# The transliteration happens in preprocess_record BEFORE
# calling normalize_address, so this function always sees ASCII
# ============================================================

def normalize_address(address: str, country: str = "") -> dict:
    """
    Normalize address string to structured components.
    Expects address to already be ASCII-clean (Indic tokens
    transliterated by transliterate_address() before this call).
    """
    NULL_VALS = {"", "null", "nan", "none", "n/a", "na", "-", "--"}

    if not isinstance(address, str) or address.strip().lower() in NULL_VALS:
        return {
            "address_missing":    True,
            "pin_codes":          [],
            "zip_codes":          [],
            "street_num":         "",
            "locality_tokens":    [],
            "state_token":        "",
            "raw_normalized":     "",
        }

    raw = address.lower()
    raw = re.sub(r"[#,./\\()\[\]]", " ", raw)   # includes () fix from Bug 8
    raw = re.sub(r"\s+", " ", raw).strip()

    # Extract PINs across full address
    pin_codes = PIN_PATTERN.findall(raw)
    raw_no_pin = PIN_PATTERN.sub(" ", raw)

    # ZIP only from latter 60% of address (Bug 4a fix)
    addr_tail = raw_no_pin[int(len(raw_no_pin) * 0.4):]
    zip_codes  = ZIP_PATTERN.findall(addr_tail)

    # Street number: first 1-5 digit token in first 60 chars
    street_num = ""
    sn_match = STREET_NUM.search(raw[:60])
    if sn_match:
        street_num = sn_match.group(1)

    # State abbreviation expansion — US only (Bug 1+2 fix)
    is_us = country.strip().lower() in ("us", "united states", "usa")
    state_token = ""
    tokens_raw = raw.split()
    normalized_tokens = []

    for tok in tokens_raw:
        tok_clean = re.sub(r"[^a-z]", "", tok)
        if (
            is_us
            and tok_clean in US_STATE_MAP
            and len(tok_clean) == 2
            and tok.isalpha()
        ):
            state_token = US_STATE_MAP[tok_clean]
            normalized_tokens.append(state_token)
        else:
            normalized_tokens.append(tok_clean if tok_clean else tok)

    # Locality tokens — now guaranteed ASCII after transliteration
    locality_tokens = [
        t for t in normalized_tokens
        if t not in ADDRESS_NOISE
        and len(t) > 3
        and not t.isdigit()
        and t.isascii()        # safety net — should all be ASCII now
    ]

    return {
        "address_missing":    False,
        "pin_codes":          pin_codes,
        "zip_codes":          zip_codes,
        "street_num":         street_num,
        "locality_tokens":    locality_tokens[:5],
        "state_token":        state_token,
        "raw_normalized":     " ".join(normalized_tokens),
    }


# ============================================================
# STAGE 0: MASTER PREPROCESSING FUNCTION
# ============================================================

def preprocess_record(row: pd.Series) -> dict:
    entity_id   = str(row.get("entity_id", "")).strip()
    raw_name    = str(row.get("business_name", ""))
    raw_address = str(row.get("business_address", ""))
    country     = str(row.get("country", "")).strip()

    # ── Script detection on ORIGINAL name
    script_type = detect_script_type(raw_name)

    # ── Name pipeline
    name_cleaned = full_latin_pipeline(raw_name)

    variants = generate_name_variants(name_cleaned)

    if script_type == "latin":
        name_ascii = nfkd_aggressive(name_cleaned)
    else:
        name_ascii = ""

    # ── Transliteration for non-Latin names
    if script_type != "latin":
        transliterated_raw  = transliterate_indic(raw_name, script_type)
        name_transliterated = full_latin_pipeline(transliterated_raw)

        # Populate sorted/first2 from transliterated form for Indic records
        trans_variants = generate_name_variants(name_transliterated)
        name_sorted = trans_variants["name_sorted"]
        name_first2 = trans_variants["name_first2"]
    else:
        name_transliterated = name_cleaned
        name_sorted = variants["name_sorted"]
        name_first2 = variants["name_first2"]

    is_domain_name = bool(
        re.search(r'\.(com|net|org|in|co\.in)\b', raw_name, re.IGNORECASE)
    )

    # ── Address: transliterate FIRST, then normalize
    # This converts महाराष्ट्र → maharastra, दिल्ली → dilli
    # BEFORE normalize_address tries to tokenize and extract components
    address_for_norm = transliterate_address(raw_address)

    addr = normalize_address(address_for_norm, country=country)

    return {
        "entity_id":              entity_id,
        "country":                country,
        "raw_name":               raw_name,
        "raw_address":            raw_address,

        "name_full":              name_cleaned if script_type == "latin" else raw_name,
        "name_no_suffix":         variants["name_no_suffix"] if script_type == "latin" else raw_name,
        "name_sorted":            name_sorted,
        "name_first2":            name_first2,
        "name_ascii":             name_ascii,
        "name_transliterated":    name_transliterated,

        "script_type":            script_type,
        "is_domain_name":         is_domain_name,

        "address_missing":        addr["address_missing"],
        "pin_codes":              addr["pin_codes"],
        "zip_codes":              addr["zip_codes"],
        "street_num":             addr["street_num"],
        "locality_tokens":        addr["locality_tokens"],
        "state_token":            addr["state_token"],
        "address_raw_normalized": addr["raw_normalized"],
    }

def preprocess_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Apply preprocess_record to every row in a source DataFrame.
    Returns a new DataFrame with all normalized fields.
    """
    print(f"  Preprocessing {source_label}: {len(df)} records...")
    records = [preprocess_record(row) for _, row in df.iterrows()]
    result = pd.DataFrame(records)
    print(f"  Done. Columns: {list(result.columns)}")
    return result


# ============================================================
# STAGE 1: COUNTRY PARTITIONING
# ============================================================

def partition_by_country(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> dict:
    """
    Stage 1: Split all three preprocessed sources by raw country string.
    Groups on the exact string as it appears — open-set safe.
    France (unseen in train) forms its own partition automatically.

    Returns:
        dict keyed by country string, each value is:
        {
          "s1": DataFrame,   # S1 records in this country
          "s2": DataFrame,   # S2 records in this country
          "s3": DataFrame,   # S3 records in this country
        }
    """
    # Collect all country strings across all sources
    all_countries = set(s1["country"].unique()) \
                  | set(s2["country"].unique()) \
                  | set(s3["country"].unique())

    # Remove empty-string country (records with missing country field)
    # These are partitioned together under key "__unknown__"
    all_countries.discard("")

    print(f"\nStage 1: Found {len(all_countries)} country partitions: {sorted(all_countries)}")

    partitions = {}

    for country in sorted(all_countries):
        s1_part = s1[s1["country"] == country].reset_index(drop=True)
        s2_part = s2[s2["country"] == country].reset_index(drop=True)
        s3_part = s3[s3["country"] == country].reset_index(drop=True)

        partitions[country] = {
            "s1": s1_part,
            "s2": s2_part,
            "s3": s3_part,
        }

        print(
            f"  [{country}] "
            f"S1={len(s1_part):>6}  "
            f"S2={len(s2_part):>6}  "
            f"S3={len(s3_part):>6}"
        )

    # Handle records with missing country
    s1_unknown = s1[s1["country"] == ""].reset_index(drop=True)
    s2_unknown = s2[s2["country"] == ""].reset_index(drop=True)
    s3_unknown = s3[s3["country"] == ""].reset_index(drop=True)

    if len(s1_unknown) + len(s2_unknown) + len(s3_unknown) > 0:
        partitions["__unknown__"] = {
            "s1": s1_unknown,
            "s2": s2_unknown,
            "s3": s3_unknown,
        }
        print(
            f"  [__unknown__] "
            f"S1={len(s1_unknown):>6}  "
            f"S2={len(s2_unknown):>6}  "
            f"S3={len(s3_unknown):>6}"
        )

    return partitions

# ============================================================
# EXPORT: SAVE PROCESSED DATA TO DISK
# ============================================================

import os
import json

def save_processed_data(
    processed: dict,
    train_partitions: dict,
    test_partitions: dict,
    output_dir: str = "processed_data",
):
    """
    Save all Stage 0 + Stage 1 outputs to disk.

    Directory structure:
        processed_data/
        ├── stage0/
        │   ├── train_source1.tsv
        │   ├── train_source2.tsv
        │   ├── train_source3.tsv
        │   ├── test_source1.tsv
        │   ├── test_source2.tsv
        │   ├── test_source3.tsv
        │   └── train_ground_truth.tsv
        ├── stage1/
        │   ├── train/
        │   │   ├── <country>_s1.tsv
        │   │   ├── <country>_s2.tsv
        │   │   └── <country>_s3.tsv
        │   └── test/
        │       ├── <country>_s1.tsv
        │       ├── <country>_s2.tsv
        │       └── <country>_s3.tsv
        └── meta/
            ├── partition_summary.json
            └── subgroup_flags.tsv
    """

    # ── Create directory structure
    stage0_dir = os.path.join(output_dir, "stage0")
    stage1_train_dir = os.path.join(output_dir, "stage1", "train")
    stage1_test_dir  = os.path.join(output_dir, "stage1", "test")
    meta_dir         = os.path.join(output_dir, "meta")

    for d in [stage0_dir, stage1_train_dir, stage1_test_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Helper: serialise list/dict columns to JSON strings before saving
    LIST_COLS = ["pin_codes", "zip_codes", "locality_tokens"]

    def prepare_for_export(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert list-typed columns to JSON strings so TSV is flat.
        Downstream code can json.loads() them back.
        """
        df = df.copy()
        for col in LIST_COLS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v) if isinstance(v, list) else v
                )
        return df

    def save_tsv(df: pd.DataFrame, path: str, label: str):
        df_out = prepare_for_export(df)
        df_out.to_csv(path, sep="\t", index=False)
        print(f"  Saved {label}: {len(df_out):>7} rows  →  {path}")

    # ──────────────────────────────────────────────
    # STAGE 0: full preprocessed sources
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPORT — STAGE 0 (full preprocessed sources)")
    print("=" * 60)

    stage0_keys = [
        "train_source1", "train_source2", "train_source3",
        "test_source1",  "test_source2",  "test_source3",
        "train_ground_truth",
    ]
    for key in stage0_keys:
        if key not in processed:
            print(f"  SKIP {key} — not found in processed dict")
            continue
        path = os.path.join(stage0_dir, f"{key}.tsv")
        save_tsv(processed[key], path, key)

    # ──────────────────────────────────────────────
    # STAGE 1: per-country partition files
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPORT — STAGE 1 (country partitions)")
    print("=" * 60)

    partition_summary = {}   # for meta JSON

    for split_label, partitions, out_dir in [
        ("train", train_partitions, stage1_train_dir),
        ("test",  test_partitions,  stage1_test_dir),
    ]:
        print(f"\n  -- {split_label.upper()} --")
        partition_summary[split_label] = {}

        for country, srcs in partitions.items():
            # Sanitise country string for use as filename
            country_slug = re.sub(r"[^\w]", "_", country).strip("_").lower()
            if not country_slug:
                country_slug = "unknown"

            counts = {}
            for src_key in ["s1", "s2", "s3"]:
                df = srcs[src_key]
                fname = f"{country_slug}_{src_key}.tsv"
                path  = os.path.join(out_dir, fname)
                save_tsv(df, path, f"{split_label}/{country}/{src_key}")
                counts[src_key] = len(df)

            partition_summary[split_label][country] = counts

    # ──────────────────────────────────────────────
    # META 1: partition_summary.json
    # ──────────────────────────────────────────────
    summary_path = os.path.join(meta_dir, "partition_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(partition_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved partition summary → {summary_path}")

    # ──────────────────────────────────────────────
    # META 2: subgroup_flags.tsv
    # Collects entity_id + all boolean/categorical flags
    # from every source for easy validation slicing later
    # ──────────────────────────────────────────────
    FLAG_COLS = [
        "entity_id", "country", "script_type",
        "is_domain_name", "address_missing",
    ]
    flag_frames = []
    for key in stage0_keys:
        if key == "train_ground_truth":
            continue
        df = processed.get(key)
        if df is None:
            continue
        available = [c for c in FLAG_COLS if c in df.columns]
        sub = df[available].copy()
        sub["source_file"] = key
        flag_frames.append(sub)

    if flag_frames:
        flags_df = pd.concat(flag_frames, ignore_index=True)
        flags_path = os.path.join(meta_dir, "subgroup_flags.tsv")
        flags_df.to_csv(flags_path, sep="\t", index=False)
        print(f"  Saved subgroup flags  → {flags_path}  ({len(flags_df)} rows)")

        # Print subgroup counts to console
        print("\n  Subgroup breakdown (across all sources):")
        print(f"    Non-ASCII names    : {(flags_df['script_type'] != 'latin').sum()}")
        print(f"    Missing address    : {flags_df['address_missing'].sum()}")
        print(f"    Domain-name records: {flags_df['is_domain_name'].sum()}")
        print(f"    Script types       :\n{flags_df['script_type'].value_counts().to_string()}")

    print("\n" + "=" * 60)
    print(f"Export complete. All files written under: {output_dir}/")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# LOADING UTILITY: read back any exported partition or stage0 file
# ──────────────────────────────────────────────────────────────

def load_processed(path: str) -> pd.DataFrame:
    """
    Load any TSV exported by save_processed_data().
    Deserialises list-typed columns (pin_codes, zip_codes,
    locality_tokens) back from JSON strings to Python lists.
    """
    LIST_COLS = ["pin_codes", "zip_codes", "locality_tokens"]
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.loads(v) if v.startswith("[") else []
            )
    return df


def load_partition(
    country: str,
    split: str = "test",
    output_dir: str = "processed_data",
) -> dict:
    """
    Convenience loader: given a country string and split,
    return {"s1": df, "s2": df, "s3": df} for that partition.

    Usage:
        india = load_partition("India", split="test")
        france = load_partition("France", split="test")
    """
    country_slug = re.sub(r"[^\w]", "_", country).strip("_").lower()
    base = os.path.join(output_dir, "stage1", split)
    result = {}
    for src in ["s1", "s2", "s3"]:
        path = os.path.join(base, f"{country_slug}_{src}.tsv")
        if os.path.exists(path):
            result[src] = load_processed(path)
        else:
            print(f"  WARNING: {path} not found — returning empty DataFrame")
            result[src] = pd.DataFrame()
    return result


# ============================================================
# MAIN: RUN STAGE 0 + STAGE 1
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # SAMPLE MODE — set to False for full run
    # ============================================================
    SAMPLE_MODE = True
    SAMPLE_SIZE = 10000       # rows per source file

    # ── Load raw data
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)
    data = load_sources()

    # ── Apply sampling if in sample mode
    if SAMPLE_MODE:
        print(f"\n⚠️  SAMPLE MODE ON — using {SAMPLE_SIZE} rows per source")
        for key in data:
            if key == "train_ground_truth":
                continue
            data[key] = data[key].head(SAMPLE_SIZE).reset_index(drop=True)

    # ── Stage 0: Preprocess
    print("\n" + "=" * 60)
    print("STAGE 0: PREPROCESSING")
    print("=" * 60)

    processed = {}
    for key, df in data.items():
        if key == "train_ground_truth":
            processed[key] = df
            continue
        processed[key] = preprocess_dataframe(df, source_label=key)

    # ── Stage 1: Country partition
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

    # ── Correctness checks
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECKS")
    print("=" * 60)

    source1 = processed["train_source1"]

    # Check 1: No raw name is lost — every input row has an entity_id
    assert len(source1) == len(data["train_source1"]), \
        "Row count mismatch after preprocessing"
    print("✓ Row count preserved")

    # Check 2: name_sorted tokens are alphabetically sorted
    for _, row in source1.head(10).iterrows():
        tokens = row["name_sorted"].split()
        assert tokens == sorted(tokens), \
            f"name_sorted not sorted for {row['entity_id']}: {row['name_sorted']}"
    print("✓ name_sorted is alphabetically sorted")

    # Check 3: name_first2 has at most 2 tokens
    for _, row in source1.iterrows():
        assert len(row["name_first2"].split()) <= 2, \
            f"name_first2 has >2 tokens for {row['entity_id']}"
    print("✓ name_first2 has ≤2 tokens")

    # Check 4: No legal suffixes remain in name_no_suffix
    SUFFIX_TOKENS = {"pvt", "ltd", "llc", "llp", "lp", "inc", "corp", "plc"}
    violations = []
    for _, row in source1.iterrows():
        tokens = set(row["name_no_suffix"].split())
        found = tokens & SUFFIX_TOKENS
        if found:
            violations.append((row["entity_id"], row["name_no_suffix"], found))
    if violations:
        print(f"⚠️  name_no_suffix still has suffix tokens in {len(violations)} records:")
        for eid, name, found in violations[:3]:
            print(f"     {eid}: '{name}' → {found}")
    else:
        print("✓ No legal suffixes remain in name_no_suffix")

    # Check 5: OCR digit fix — spot check known patterns
    test_cases = [
        ("TRADIN6",  "trading"),   # 6→g, then lowercase: TRADINg → trading ✓
        ("N0se",     "nose"),      # 0→o, then lowercase: Nose → nose ✓
        ("5uperior", "superior"),  # 5→s, then lowercase: superior ✓
        ("TR6ADI6G", "trgadigg"),  # multiple 6s each become g independently
    ]
    for raw, expected in test_cases:
        result = fix_ocr(base_clean(raw))
        assert result == expected, f"OCR fix failed: '{raw}' → '{result}', expected '{expected}'"
    print("✓ OCR digit correction working")
    
    # Fix 5: transliterated names must be clean ASCII, no IAST diacritics
    s2 = processed["train_source2"]
    indic_records = s2[s2["script_type"] != "latin"]
    for _, row in indic_records.iterrows():
        trans = row["name_transliterated"]
        # Must be pure ASCII
        try:
            trans.encode("ascii")
        except UnicodeEncodeError:
            print(f"  ✗ IAST diacritics remain in {row['entity_id']}: {trans}")
            break
        # Must not contain known IAST suffix fragments
        bad_fragments = ["prāiveṭa", "limiṭeḍa", "elaelapī", "prāi"]
        found = [f for f in bad_fragments if f in trans]
        if found:
            print(f"  ✗ Untranslated suffix in {row['entity_id']}: {trans} → {found}")
            break
    else:
        print("✓ Fix 5: All transliterated names are clean ASCII, suffixes stripped")

    # Fix 1+3: no pipe or comma in name_full
    pipe_comma = s2[
        s2["name_full"].str.contains(r"[|,]", regex=True, na=False)
    ]
    if len(pipe_comma):
        print(f"  ✗ Fix 1/3: pipe or comma still in name_full: "
              f"{pipe_comma[['entity_id','name_full']].head(3).to_string()}")
    else:
        print("✓ Fix 1+3: No pipe or comma in name_full")
        
    # Fix 4b: 341ND should not produce state_token = north dakota
    addr_test = normalize_address("WA, FEDERAL WAY, ##2326 341ND PLACE", "US")
    assert addr_test["state_token"] == "washington", \
        f"Fix 4b failed: state_token = {addr_test['state_token']}"
    print("✓ Fix 4b: Alphanumeric tokens not parsed as state abbreviations")

    # Fix 4a: 5-digit street number not captured as ZIP
    addr_test2 = normalize_address("61573 FRIENDSHIP LN, CNROE, TX", "US")
    assert "61573" not in addr_test2["zip_codes"], \
        f"Fix 4a failed: street number captured as ZIP: {addr_test2['zip_codes']}"
    print("✓ Fix 4a: Street-leading 5-digit numbers not captured as ZIP")

    # Fix 8: apostrophes stripped
    assert "'" not in full_latin_pipeline("Candy's Services"), \
        "Fix 8 failed: apostrophe survived"
    print("✓ Fix 8: Apostrophes stripped from names")
    
    # Check 5b: OCR fix must NOT touch address field
    # street numbers like "506" must survive unchanged
    addr_test = normalize_address("506 Main Street, Columbus, OH", "US")
    assert addr_test["street_num"] == "506", \
        f"OCR fix contaminated address: street_num = {addr_test['street_num']}"
    print("✓ OCR fix not applied to address field")

    # Check 6: NFKD stripping on accented Latin names
    accent_cases = [
        ("Sólar",    "Solar"),
        ("Léarning", "Learning"),
        ("Nétwork",  "Network"),
        ("Béque",    "Beque"),
    ]
    for raw, expected in accent_cases:
        result = nfkd_strip(raw.lower())
        assert result == expected.lower(), \
            f"NFKD failed: '{raw}' → '{result}', expected '{expected.lower()}'"
    print("✓ NFKD diacritic stripping working")

    # Check 7: Consecutive dedup
    dedup_cases = [
        ("crestline crestline clean",     "crestline clean"),
        ("ferrero ferrero duke",          "ferrero duke"),
        ("keystone odyssey odyssey llc",  "keystone odyssey llc"),
    ]
    for raw, expected in dedup_cases:
        result = dedup_consecutive(raw)
        assert result == expected, \
            f"Dedup failed: '{raw}' → '{result}', expected '{expected}'"
    print("✓ Consecutive word deduplication working")

    # Check 8: address_missing flag set correctly
    missing_addr = source1[source1["address_missing"] == True]
    present_addr = source1[source1["address_missing"] == False]
    print(f"✓ Address missing flag: {len(missing_addr)} missing, {len(present_addr)} present")

    # Check 9: Country partitions cover all S1 records
    train_s1_total = sum(
        len(v["s1"]) for v in train_partitions.values()
    )
    assert train_s1_total == len(processed["train_source1"]), \
        f"Partition S1 count {train_s1_total} ≠ total {len(processed['train_source1'])}"
    print(f"✓ All S1 records accounted for across partitions ({train_s1_total} total)")

    # Check 10: No entity_id is duplicated within a source
    for key in ["train_source1", "train_source2", "train_source3"]:
        dups = processed[key]["entity_id"].duplicated().sum()
        assert dups == 0, f"Duplicate entity_ids in {key}: {dups}"
    print("✓ No duplicate entity_ids within any source")
    
        # Address transliteration checks
    print("\n── Address Transliteration Checks ──")

    # Check: Devanagari state name in address becomes ASCII
    result = transliterate_address("Mumbai, महाराष्ट्र")
    assert result.isascii(), f"Address still has non-ASCII: {result}"
    assert "maharastra" in result or "maharashtra" in result, \
        f"महाराष्ट्र not transliterated correctly: {result}"
    print(f"✓ महाराष्ट्र → {result}")

    # Check: Mixed Latin+Indic address
    result2 = transliterate_address("Lucknow, उत्तर प्रदेश")
    assert result2.isascii(), f"Mixed address still has non-ASCII: {result2}"
    print(f"✓ उत्तर प्रदेश → {result2}")

    # Check: Pure ASCII address unchanged
    result3 = transliterate_address("123 Main Street, Columbus, OH")
    assert result3 == "123 Main Street, Columbus, OH", \
        f"Pure ASCII address was modified: {result3}"
    print(f"✓ Pure ASCII address unchanged")

    # Check: locality_tokens has no non-ASCII after full pipeline
    for key in ["train_source2", "train_source3"]:
        df = processed[key]
        for _, row in df.iterrows():
            for tok in row["locality_tokens"]:
                assert tok.isascii(), \
                    f"Non-ASCII in locality_tokens for {row['entity_id']}: {tok}"
    print("✓ All locality_tokens are ASCII across S2 and S3")

    # Check: Indic name records now have non-empty name_sorted/name_first2
    indic = processed["train_source2"][
        processed["train_source2"]["script_type"] != "latin"
    ]
    empty_sorted = indic[indic["name_sorted"] == ""]
    if len(empty_sorted):
        print(f"  ⚠ {len(empty_sorted)} Indic records still have empty name_sorted")
    else:
        print("✓ All Indic records have name_sorted populated from transliteration")

    # ── Pretty-print sample records for manual inspection
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS — MANUAL INSPECTION")
    print("=" * 60)

    INSPECT_COLS = [
        "entity_id", "raw_name",
        "name_full", "name_sorted", "name_transliterated", "name_ascii",
        "script_type", "is_domain_name",
        "address_missing", "street_num", "locality_tokens",
        "state_token", "pin_codes", "zip_codes", "country",
    ]

    # Pick interesting records: one Indic, one domain-name, one missing address,
    # one normal — whatever exists in the sample
    picks = []

    indic = source1[source1["script_type"] != "latin"]
    if len(indic): picks.append(("Indic script",   indic.iloc[0]))

    domain = source1[source1["is_domain_name"] == True]
    if len(domain): picks.append(("Domain name",   domain.iloc[0]))

    missing = source1[source1["address_missing"] == True]
    if len(missing): picks.append(("Missing addr",  missing.iloc[0]))

    picks.append(("Normal record", source1.iloc[0]))

    for label, row in picks:
        print(f"\n  ── {label} ──")
        for col in INSPECT_COLS:
            if col in row.index:
                print(f"    {col:<25}: {row[col]}")

    # ── Export (tagged as sample)
    out_dir = "processed_data_sample" if SAMPLE_MODE else "processed_data"
    save_processed_data(
        processed=processed,
        train_partitions=train_partitions,
        test_partitions=test_partitions,
        output_dir=out_dir,
    )

    print(f"\n{'⚠️  SAMPLE RUN COMPLETE' if SAMPLE_MODE else '✅ FULL RUN COMPLETE'}")
    print(f"Output directory: {out_dir}/")
    print("Set SAMPLE_MODE = False to run on full data.")