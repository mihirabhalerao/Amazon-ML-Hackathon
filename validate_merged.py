# ============================================================
# validate_merged.py
# Validates output of normalization_merged.py (Stage 0 + Stage 1).
#
# Usage:
#   python validate_merged.py                         # checks processed_data_sample/
#   python validate_merged.py --dir processed_data    # checks full run
#   python validate_merged.py --unit-only             # unit tests only
# ============================================================

import os
import sys
import json
import argparse
import unicodedata

import pandas as pd

# ── Import exactly what normalization_merged.py exports
from normalization_merged import (
    # Stage 0A
    base_clean,
    canonicalize_dotted_legal_terms,
    unicode_safe_punctuation_to_spaces,
    # Stage 0B
    dedup_consecutive,
    dedup_all_tokens,
    # Stage 0C  (takes list[str], returns list[str])
    fix_ocr,
    # Stage 0D
    fold_for_lookup,
    canonicalize_legal_suffixes,
    strip_legal_suffix_tokens,
    # Stage 0E
    generate_name_variants,
    # Stage 0F
    nfkd_aggressive,
    # Stage 0G
    detect_script_type,
    transliterate_name,       # ← not "transliterate_indic"
    transliterate_address,
    # Stage 0H
    extract_address_fields,   # ← not "normalize_address"
    normalize_state_token,
    # Pipeline
    full_latin_pipeline,
    # Constants
    SUFFIX_REMOVE,
    NULL_STRINGS,
    # I/O
    load_processed,
)


# ============================================================
# REPORTING HELPERS
# ============================================================

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

_results: list[tuple[str, str, str]] = []


def ok(category: str, msg: str) -> None:
    _results.append((PASS, category, msg))
    print(f"  {PASS}  [{category}]  {msg}")


def fail(category: str, msg: str) -> None:
    _results.append((FAIL, category, msg))
    print(f"  {FAIL}  [{category}]  {msg}")


def warn(category: str, msg: str) -> None:
    _results.append((WARN, category, msg))
    print(f"  {WARN}  [{category}]  {msg}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ============================================================
# UNIT TESTS
# ============================================================

def test_base_clean() -> None:
    section("base_clean")
    cases = [
        ("-- Holloway Peak Inc",  "holloway peak inc",    "leading dashes stripped"),
        ("[INCORPORATED] Peak",   "incorporated peak",    "square brackets stripped"),
        ("heassociates.com",      "heassociates",         "domain suffix stripped"),
        ("SHIVSHAKTI | www.x",    "shivshakti www x",     "pipe stripped"),
        ("Hail, Harris & Co",     "hail harris and co",   "comma + ampersand"),
        ("Candy's-Services",      "candys services",      "apostrophe + dash"),
        ("M/s Sandeep Software",  "sandeep software",     "M/s prefix stripped"),
        ("Crestline.Corp",        "crestline corp",       "dot split"),
        ("",                      "",                     "empty string"),
        ("NULL",                  "",                     "literal NULL"),
    ]
    for inp, expected, desc in cases:
        result = base_clean(inp)
        if result == expected:
            ok("base_clean", f"{desc}: '{inp}' → '{result}'")
        else:
            fail("base_clean", f"{desc}: '{inp}' → '{result}'  (expected '{expected}')")


def test_dedup_consecutive() -> None:
    section("dedup_consecutive")
    cases = [
        ("crestline crestline clean",    "crestline clean"),
        ("ferrero ferrero duke",         "ferrero duke"),
        ("keystone odyssey odyssey llc", "keystone odyssey llc"),
        ("a b c",                        "a b c"),
        ("",                             ""),
    ]
    for inp, expected in cases:
        result = dedup_consecutive(inp)
        if result == expected:
            ok("dedup_consecutive", f"'{inp}' → '{result}'")
        else:
            fail("dedup_consecutive", f"'{inp}' → '{result}'  (expected '{expected}')")


def test_dedup_all_tokens() -> None:
    section("dedup_all_tokens")
    cases = [
        ("capital capital snow snow", "capital snow"),
        ("a a b b c",                "a b c"),
        ("apple banana",             "apple banana"),
    ]
    for inp, expected in cases:
        result = dedup_all_tokens(inp)
        if result == expected:
            ok("dedup_all_tokens", f"'{inp}' → '{result}'")
        else:
            fail("dedup_all_tokens", f"'{inp}' → '{result}'  (expected '{expected}')")


def test_fix_ocr() -> None:
    """fix_ocr takes list[str] and returns list[str].
    With APPLY_OCR_CORRECTION=False (the default) it is a no-op.
    Tests here verify the list-in/list-out contract and that pure-digit
    tokens are never altered when OCR correction IS applied.
    """
    section("fix_ocr — list[str] contract")

    from normalization_merged import APPLY_OCR_CORRECTION

    # Always: pure-digit tokens must survive unchanged regardless of flag
    result = fix_ocr(["2067865001", "trading", "n0se"])
    if isinstance(result, list):
        ok("fix_ocr", f"Returns list: {result}")
    else:
        fail("fix_ocr", f"Expected list, got {type(result)}: {result}")
        return

    if APPLY_OCR_CORRECTION:
        # Mixed token correction
        cases = [
            (["tradin6"],    ["trading"],    "6→g in mixed token"),
            (["n0se"],       ["nose"],       "0→o in mixed token"),
            (["5uperior"],   ["superior"],   "5→s in mixed token"),
            (["art5"],       ["arts"],       "5→s suffix"),
            (["fog1eman"],   ["fogleman"],   "1→l in mixed token"),
            # Pure-digit — must NOT be touched
            (["2067865001"], ["2067865001"], "pure digit unchanged"),
            (["98825"],      ["98825"],      "pure digit unchanged"),
            (["trading"],    ["trading"],    "pure alpha unchanged"),
        ]
        for inp, expected, desc in cases:
            r = fix_ocr(inp)
            if r == expected:
                ok("fix_ocr", f"{desc}: {inp} → {r}")
            else:
                fail("fix_ocr", f"{desc}: {inp} → {r}  (expected {expected})")
    else:
        # Flag is off — must be a pure pass-through
        tokens = ["tradin6", "2067865001", "trading"]
        r = fix_ocr(tokens)
        if r == tokens:
            ok("fix_ocr", f"APPLY_OCR_CORRECTION=False → identity: {r}")
        else:
            fail("fix_ocr", f"Expected identity with flag off, got {r}")

        # Confirm pure-digit is still safe
        r2 = fix_ocr(["2067865001"])
        if r2 == ["2067865001"]:
            ok("fix_ocr", "Pure-digit token unchanged with flag off")
        else:
            fail("fix_ocr", f"Pure-digit altered: {r2}")


def test_nfkd_aggressive() -> None:
    section("nfkd_aggressive")
    cases = [
        ("Sólar",     "solar"),
        ("Léarning",  "learning"),
        ("rāma",      "rama"),
        ("limiṭeḍa",  "limiteda"),
        ("prāiveṭa",  "praiveta"),
        ("",          ""),
    ]
    for inp, expected in cases:
        result = nfkd_aggressive(inp.lower())
        if result == expected:
            ok("nfkd_aggressive", f"'{inp}' → '{result}'")
        else:
            fail("nfkd_aggressive", f"'{inp}' → '{result}'  (expected '{expected}')")


def test_dotted_legal_terms() -> None:
    """canonicalize_dotted_legal_terms() is designed to run BEFORE dots are
    split by base_clean().  It canonicalises dotted patterns like 'P.V.T'
    and 'L.L.C' that appear mid-string.

    Important: the regex patterns use \b word boundaries.  A trailing '.'
    that comes AFTER the last letter (e.g. 'Corp.') is outside the word
    boundary, so it is NOT consumed by the pattern — it remains in the
    string to be removed later by base_clean's dot-split pass.
    These tests reflect the actual contract of the function in isolation.
    """
    section("canonicalize_dotted_legal_terms")

    # Mid-string / no trailing punctuation — fully replaced
    cases_exact = [
        ("P.V.T",          "pvt",             "P.V.T without trailing dot"),
        ("L.L.C",          "llc",             "L.L.C without trailing dot"),
        ("L.L.P",          "llp",             "L.L.P without trailing dot"),
        ("Pvt Ltd",        "private limited", "Pvt Ltd with space separator"),
    ]
    for inp, expected, desc in cases_exact:
        result = canonicalize_dotted_legal_terms(inp).lower().strip()
        if result == expected:
            ok("dotted_legal", f"{desc}: '{inp}' → '{result}'")
        else:
            fail("dotted_legal", f"{desc}: '{inp}' → '{result}'  (expected '{expected}')")

    # Trailing dot AFTER the pattern is outside \b — the trailing dot survives
    # in the isolated function; base_clean() removes it downstream.
    cases_trailing = [
        ("L.L.C.",   "llc."),
        ("Pvt. Ltd.", "private limited."),
        ("Corp.",     "corp."),
    ]
    for inp, expected in cases_trailing:
        result = canonicalize_dotted_legal_terms(inp).lower().strip()
        if result == expected:
            ok("dotted_legal", f"trailing dot survives (removed by base_clean): '{inp}' → '{result}'")
        else:
            fail("dotted_legal", f"'{inp}' → '{result}'  (expected '{expected}')")

    # Verify full_latin_pipeline handles trailing dot correctly end-to-end.
    # Note: strip_legal_suffix_tokens has a fallback — when the ENTIRE name
    # is a suffix token, it returns it unchanged (avoids empty output).
    # So bare-suffix names like "Corp." → "corp" (not "").
    from normalization_merged import full_latin_pipeline
    e2e_cases = [
        ("L.L.C.",    "llc"),    # fallback: whole name is a suffix → kept
        ("Corp.",     "corp"),   # fallback: same
        ("Apex Corp.", "apex"),  # non-suffix survives, Corp stripped
    ]
    for inp, expected in e2e_cases:
        result = full_latin_pipeline(inp)
        if result == expected:
            ok("dotted_legal", f"full_latin_pipeline end-to-end: '{inp}' → '{result}'")
        else:
            fail("dotted_legal", f"full_latin_pipeline: '{inp}' → '{result}'  (expected '{expected}')")


def test_canonicalize_legal_suffixes() -> None:
    """canonicalize_legal_suffixes takes list[str], returns list[str]."""
    section("canonicalize_legal_suffixes + strip_legal_suffix_tokens")
    cases = [
        (["shree", "infracon", "pvt", "ltd"],          ["shree", "infracon"],         "pvt ltd stripped"),
        (["summit", "inc"],                            ["summit"],                    "inc stripped"),
        (["asset", "building", "coalition", "llc"],    ["asset", "building", "coalition"], "llc stripped"),
        (["center", "grand", "pllc"],                  ["center", "grand"],           "pllc→llc then stripped"),
        (["mw", "management", "2067865001"],           ["mw", "management"],          "trailing numeric-ID stripped"),
    ]
    for tokens_in, expected_out, desc in cases:
        canonicalized = canonicalize_legal_suffixes(tokens_in)
        stripped      = strip_legal_suffix_tokens(canonicalized)
        if stripped == expected_out:
            ok("suffix", f"{desc}: {tokens_in} → {stripped}")
        else:
            fail("suffix", f"{desc}: {tokens_in} → {stripped}  (expected {expected_out})")


def test_generate_variants() -> None:
    section("generate_name_variants")
    v = generate_name_variants("apex summit global")
    tokens_sorted = v["name_sorted"].split()
    if tokens_sorted == sorted(tokens_sorted):
        ok("variants", f"name_sorted is alphabetically sorted: '{v['name_sorted']}'")
    else:
        fail("variants", f"name_sorted not sorted: '{v['name_sorted']}'")

    if len(v["name_first2"].split()) <= 2:
        ok("variants", f"name_first2 has ≤2 tokens: '{v['name_first2']}'")
    else:
        fail("variants", f"name_first2 has >2 tokens: '{v['name_first2']}'")

    # name_sorted deduplicates after sorting
    v2 = generate_name_variants("capital snow capital snow")
    if v2["name_sorted"] == "capital snow":
        ok("variants", f"name_sorted dedupes after sort: '{v2['name_sorted']}'")
    else:
        fail("variants", f"name_sorted did not dedupe: '{v2['name_sorted']}'")

    # name_full is exactly the input
    v3 = generate_name_variants("apex summit global")
    if v3["name_full"] == "apex summit global":
        ok("variants", f"name_full preserved: '{v3['name_full']}'")
    else:
        fail("variants", f"name_full wrong: '{v3['name_full']}'")


def test_script_detection() -> None:
    section("detect_script_type")
    cases = [
        ("Apex Summit",                          "latin"),
        ("राम मार्केटिंग प्राइवेट",             "devanagari"),
        ("குளோபல் பிசினஸ்",                      "tamil"),
        ("కృష్ణా ఇంపెక్స్",                      "telugu"),
        ("ಡಿಜಿಟಲ್ ಬಿಲ್ಡರ್ಸ್",                  "kannada"),
        ("സിൽവർ കൺസൾട്ടൻസി",                    "malayalam"),
        ("ਸਕਾਈ ਅਰਿਹੰਤ",                          "gurmukhi"),
        ("ইনোভেটিভ প্রোডাক্টস",                 "bengali"),
        ("Gujarat Logistics లిమిటెడ్",           "mixed"),
    ]
    for text, expected in cases:
        result = detect_script_type(text)
        if result == expected:
            ok("script_detect", f"'{text[:30]}' → {result}")
        else:
            fail("script_detect", f"'{text[:30]}' → {result}  (expected {expected})")


def test_transliterate_name() -> None:
    """transliterate_name(raw_name, script_type) → ASCII string."""
    section("transliterate_name")
    cases = [
        # (raw, script_type)
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "devanagari"),
        ("மும்பை வணிகம்",                   "tamil"),
        ("కృష్ణా ఇంపెక్స్",                  "telugu"),
        ("ಡಿಜಿಟಲ್ ಬಿಲ್ಡರ್ಸ್",              "kannada"),
    ]
    for raw, script in cases:
        result = transliterate_name(raw, script)
        if not isinstance(result, str):
            fail("transliterate_name", f"Expected str, got {type(result)}")
            continue
        if result == "" or result.isascii():
            ok("transliterate_name", f"[{script}] '{raw[:20]}' → '{result[:30]}' (ascii={result.isascii()})")
        else:
            fail("transliterate_name", f"[{script}] non-ASCII in output: '{result}'")

    # Latin input → empty string (transliterate_name skips latin)
    r = transliterate_name("Apex Summit", "latin")
    if r == "":
        ok("transliterate_name", "Latin input → '' (correct)")
    else:
        fail("transliterate_name", f"Latin input should return '', got '{r}'")


def test_transliterate_address() -> None:
    section("transliterate_address")
    cases = [
        ("Mumbai, महाराष्ट्र",    True,  ["maharastra", "maharashtra"]),
        ("Lucknow, उत्तर प्रदेश", True,  ["uttara"]),
        ("123 Main Street, OH",    False, []),
    ]
    for addr, should_change, fragments in cases:
        result = transliterate_address(addr)
        if not result.isascii():
            fail("addr_translit", f"Non-ASCII remains: '{result}'")
            continue
        if not should_change:
            if result == addr.lower():
                ok("addr_translit", f"Pure ASCII unchanged: '{addr}'")
            else:
                # allow minor normalisation differences (whitespace/punct)
                warn("addr_translit", f"Pure ASCII slightly modified: '{addr}' → '{result}'")
        else:
            found = [f for f in fragments if f in result]
            if found:
                ok("addr_translit", f"'{addr}' → '{result}'  (found: {found})")
            else:
                warn("addr_translit",
                     f"'{addr}' → '{result}'  (expected fragments {fragments} not found)")


def test_extract_address_fields() -> None:
    """Tests for extract_address_fields(raw_address, country)."""
    section("extract_address_fields")

    # CT = Court, not Connecticut (US address)
    r = extract_address_fields("5780 Fawn Ct, Fort Worth, Texas", "US")
    if r["state_token"] == "texas":
        ok("addr_fields", "State=Texas extracted correctly; CT not mistaken for Connecticut")
    else:
        fail("addr_fields", f"state_token='{r['state_token']}' (expected 'texas')")
    if "connecticut" not in r["locality_tokens"]:
        ok("addr_fields", "CT (Court) not in locality_tokens as Connecticut")
    else:
        fail("addr_fields", "CT ended up in locality_tokens as 'connecticut'")

    # FL (Floor) not Florida in Indian address
    r = extract_address_fields("B-242 F/F, Surajmal Vihar, Delhi", "India")
    if r["state_token"] != "florida":
        ok("addr_fields", "FL not parsed as Florida for Indian address")
    else:
        fail("addr_fields", "FL parsed as Florida for Indian address")

    # TN = Tamil Nadu, not Tennessee
    r = extract_address_fields("Chennai, TN", "India")
    if r["state_token"] != "tennessee":
        ok("addr_fields", "TN → Tamil Nadu (not Tennessee) for India")
    else:
        fail("addr_fields", "TN incorrectly parsed as Tennessee for Indian address")

    # 341ND must not trigger North Dakota
    r = extract_address_fields("WA, FEDERAL WAY, 2326 341ND PLACE", "US")
    if r["state_token"] == "washington":
        ok("addr_fields", "341ND not parsed as North Dakota; WA=Washington")
    else:
        fail("addr_fields", f"state_token='{r['state_token']}' (expected 'washington')")

    # 5-digit street number not captured as ZIP
    r = extract_address_fields("61573 FRIENDSHIP LN, CNROE, TX", "US")
    if r["zip_code"] != "61573":
        ok("addr_fields", "5-digit street-leading number not captured as ZIP")
    else:
        fail("addr_fields", f"Street number 61573 wrongly captured as zip_code")

    # Hyphenated plot numbers not in locality_tokens
    r = extract_address_fields("8-2-595/3, Eden Gardens, Khairatabad", "India")
    loc_tokens = r["locality_tokens"].split()
    bad = [t for t in loc_tokens if "-" in t and t[0].isdigit()]
    if not bad:
        ok("addr_fields", "Hyphenated plot numbers filtered from locality_tokens")
    else:
        fail("addr_fields", f"Hyphenated numbers in locality_tokens: {bad}")

    # Null literal → has_address=False
    for missing in ["", "NULL", "nan", "N/A"]:
        r = extract_address_fields(missing, "")
        if not r["has_address"]:
            ok("addr_fields", f"has_address=False for '{missing}'")
        else:
            fail("addr_fields", f"has_address=True for '{missing}'")

    # locality_tokens is a space-joined string, not a list
    r = extract_address_fields("123 Main Street, Springfield, IL", "US")
    if isinstance(r["locality_tokens"], str):
        ok("addr_fields", f"locality_tokens is a str: '{r['locality_tokens']}'")
    else:
        fail("addr_fields", f"locality_tokens is {type(r['locality_tokens'])}, expected str")

    # pin_code and zip_code are strings, not lists
    r_india = extract_address_fields("Andheri, Mumbai, Maharashtra 400058", "India")
    if isinstance(r_india["pin_code"], str):
        ok("addr_fields", f"pin_code is str: '{r_india['pin_code']}'")
    else:
        fail("addr_fields", f"pin_code type={type(r_india['pin_code'])}")

    r_us = extract_address_fields("100 Main St, Springfield, IL 62701", "US")
    if isinstance(r_us["zip_code"], str):
        ok("addr_fields", f"zip_code is str: '{r_us['zip_code']}'")
    else:
        fail("addr_fields", f"zip_code type={type(r_us['zip_code'])}")

    # "null" literal must NOT appear in locality_tokens (noise-filtered).
    # address_transliterated is the raw lowercased address string — it is NOT
    # noise-filtered; "null" may appear there and that is correct behaviour.
    r = extract_address_fields("Mulberry Dr, NULL, Buckeye, Arizona", "US")
    loc = r["locality_tokens"].lower()
    if "null" not in loc:
        ok("addr_fields", "Literal NULL filtered from locality_tokens")
    else:
        fail("addr_fields", f"'null' survived in locality_tokens: '{loc}'")


# ============================================================
# DATA QUALITY CHECKS — run on exported TSV files
# ============================================================

# Expected columns produced by preprocess_record()
EXPECTED_COLUMNS = {
    "entity_id", "country", "raw_name", "raw_address",
    "name_full", "name_no_suffix", "name_sorted", "name_first2",
    "name_ascii", "name_transliterated", "name_transliterated_no_suffix",
    "script_type", "is_domain_name",
    "has_address", "pin_code", "zip_code", "street_num",
    "locality_tokens", "state_token", "address_transliterated",
}

VALID_SCRIPT_TYPES = {
    "latin", "mixed",
    "devanagari", "bengali", "gurmukhi", "gujarati", "oriya",
    "tamil", "telugu", "kannada", "malayalam",
}


def check_dataframe(df: pd.DataFrame, label: str) -> None:
    section(f"DataFrame checks: {label}  ({len(df)} rows)")

    # ── Schema
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if not missing_cols:
        ok("schema", f"All {len(EXPECTED_COLUMNS)} expected columns present")
    else:
        fail("schema", f"Missing columns: {missing_cols}")

    # ── Row count
    if len(df) > 0:
        ok("df", f"{len(df)} rows loaded")
    else:
        warn("df", "File is empty")
        return

    # ── No duplicate entity_ids
    dups = df["entity_id"].duplicated().sum()
    if dups == 0:
        ok("df", "No duplicate entity_ids")
    else:
        fail("df", f"{dups} duplicate entity_ids")

    # ── name_sorted: alphabetically sorted tokens for every row
    bad_sorted = []
    for idx, row in df.iterrows():
        tokens = row["name_sorted"].split()
        if tokens and tokens != sorted(tokens):
            bad_sorted.append(row["entity_id"])
    if not bad_sorted:
        ok("df", "name_sorted is alphabetically sorted for all rows")
    else:
        fail("df", f"name_sorted not sorted in {len(bad_sorted)} rows "
                   f"(e.g. entity_id={bad_sorted[:3]})")

    # ── name_first2: at most 2 tokens
    bad_f2 = df["name_first2"].apply(lambda x: len(x.split()) > 2).sum()
    if bad_f2 == 0:
        ok("df", "name_first2 has ≤2 tokens for all rows")
    else:
        fail("df", f"name_first2 has >2 tokens in {bad_f2} rows")

    # ── No legal suffixes in name_no_suffix (Latin records only).
    # strip_legal_suffix_tokens has an intentional fallback: when the ENTIRE
    # name is composed only of suffix tokens (e.g. entity named "PC" or "Corp"),
    # it returns the tokens unchanged rather than producing an empty string.
    # Those single-suffix rows are therefore expected and logged as warnings,
    # not failures.  Rows where a suffix survives alongside non-suffix tokens
    # are genuine bugs and are logged as failures.
    latin_df = df[df["script_type"] == "latin"]
    fallback_rows  = []   # entire name is suffix(es) — expected behaviour
    violation_rows = []   # suffix coexists with real tokens — pipeline bug
    for _, row in latin_df.iterrows():
        tokens     = row["name_no_suffix"].split()
        suffix_set = set(tokens) & SUFFIX_REMOVE
        if not suffix_set:
            continue
        non_suffix = set(tokens) - SUFFIX_REMOVE
        if non_suffix:
            violation_rows.append((row["entity_id"], suffix_set))
        else:
            fallback_rows.append((row["entity_id"], suffix_set))
    if not violation_rows:
        ok("df", f"No illegal suffix survivors in name_no_suffix ({len(latin_df)} Latin rows)")
    else:
        fail("df", f"Suffix+non-suffix coexistence in {len(violation_rows)} Latin rows "
                   f"(e.g. {violation_rows[0]})")
    if fallback_rows:
        warn("df", f"{len(fallback_rows)} rows are single-suffix entities (fallback expected, "
                   f"e.g. {fallback_rows[0]})")

    # ── name_transliterated is ASCII for all records
    bad_trans = df[~df["name_transliterated"].apply(lambda x: x.isascii())]
    if len(bad_trans) == 0:
        ok("df", "name_transliterated is ASCII for all rows")
    else:
        fail("df", f"Non-ASCII name_transliterated in {len(bad_trans)} rows: "
                   f"{bad_trans['entity_id'].head(3).tolist()}")

    # ── name_transliterated_no_suffix is ASCII
    bad_tns = df[~df["name_transliterated_no_suffix"].apply(lambda x: x.isascii())]
    if len(bad_tns) == 0:
        ok("df", "name_transliterated_no_suffix is ASCII for all rows")
    else:
        fail("df", f"Non-ASCII name_transliterated_no_suffix in {len(bad_tns)} rows")

    # ── Indic records have non-empty name_sorted
    indic_df   = df[df["script_type"] != "latin"]
    if len(indic_df) == 0:
        warn("df", "No Indic-script records in this file")
    else:
        empty_sort = (indic_df["name_sorted"] == "").sum()
        if empty_sort == 0:
            ok("df", f"All {len(indic_df)} Indic records have name_sorted populated")
        else:
            fail("df", f"{empty_sort}/{len(indic_df)} Indic records have empty name_sorted")

    # ── locality_tokens: string (space-joined), all tokens ASCII
    # (normalization_merged stores it as space-joined str, not a list)
    bad_loc_type = 0
    bad_loc_ascii = 0
    for _, row in df.iterrows():
        loc = row["locality_tokens"]
        if not isinstance(loc, str):
            bad_loc_type += 1
            continue
        for tok in loc.split():
            if not tok.isascii():
                bad_loc_ascii += 1
    if bad_loc_type == 0:
        ok("df", "locality_tokens is a string for all rows")
    else:
        fail("df", f"locality_tokens is not a string in {bad_loc_type} rows")
    if bad_loc_ascii == 0:
        ok("df", "All locality_tokens tokens are ASCII")
    else:
        fail("df", f"{bad_loc_ascii} non-ASCII tokens found in locality_tokens")

    # ── has_address is a valid boolean-like value
    valid_has_address = df["has_address"].isin([True, False, "True", "False"])
    if valid_has_address.all():
        missing_count = df["has_address"].isin([False, "False"]).sum()
        ok("df", f"has_address valid for all rows ({missing_count} missing)")
    else:
        bad_vals = df.loc[~valid_has_address, "has_address"].unique()
        fail("df", f"has_address has unexpected values: {bad_vals[:5]}")

    # ── pin_code: empty or 6-digit string starting with non-zero (India rows)
    india_df = df[df["country"].str.lower() == "india"]
    if len(india_df) > 0:
        bad_pin = india_df[
            india_df["pin_code"].apply(
                lambda x: bool(x) and (not x.isdigit() or len(x) != 6 or x[0] == "0")
            )
        ]
        if len(bad_pin) == 0:
            ok("df", f"pin_code format valid for all {len(india_df)} India rows")
        else:
            fail("df", f"Malformed pin_code in {len(bad_pin)} India rows: "
                       f"{bad_pin['pin_code'].head(3).tolist()}")

    # ── zip_code: empty or 5-digit (or 5+4) string (US rows)
    us_df = df[df["country"].str.lower() == "us"]
    if len(us_df) > 0:
        import re as _re
        zip_pat = _re.compile(r'^\d{5}(?:-\d{4})?$')
        bad_zip = us_df[
            us_df["zip_code"].apply(lambda x: bool(x) and not zip_pat.fullmatch(x))
        ]
        if len(bad_zip) == 0:
            ok("df", f"zip_code format valid for all {len(us_df)} US rows")
        else:
            fail("df", f"Malformed zip_code in {len(bad_zip)} US rows: "
                       f"{bad_zip['zip_code'].head(3).tolist()}")

    # ── script_type values are from the known set
    unknown_scripts = set(df["script_type"].unique()) - VALID_SCRIPT_TYPES
    if not unknown_scripts:
        ok("df", f"All script_type values are recognised: {df['script_type'].value_counts().to_dict()}")
    else:
        fail("df", f"Unknown script_type values: {unknown_scripts}")

    # ── No pipe or comma in name_full (cleaned names)
    bad_nc = df["name_full"].str.contains(r"[|,]", regex=True, na=False).sum()
    if bad_nc == 0:
        ok("df", "No pipe or comma in name_full")
    else:
        fail("df", f"Pipe or comma in name_full for {bad_nc} rows")

    # ── address_transliterated is ASCII
    bad_at = df[~df["address_transliterated"].apply(lambda x: x.isascii())]
    if len(bad_at) == 0:
        ok("df", "address_transliterated is ASCII for all rows")
    else:
        fail("df", f"Non-ASCII address_transliterated in {len(bad_at)} rows")


def check_partition_coverage(
    full_df: pd.DataFrame,
    output_dir: str,
    split: str,
    label: str,
) -> None:
    section(f"Partition coverage: {label} ({split})")

    part_dir = os.path.join(output_dir, "stage1", split)
    if not os.path.isdir(part_dir):
        warn("partition", f"Partition directory not found: {part_dir}")
        return

    collected_ids: set[str] = set()
    file_count = 0
    for fname in os.listdir(part_dir):
        if not fname.endswith(".tsv"):
            continue
        fpath = os.path.join(part_dir, fname)
        try:
            part_df = pd.read_csv(fpath, sep="\t", dtype=str).fillna("")
            if "entity_id" in part_df.columns:
                collected_ids.update(part_df["entity_id"].tolist())
            file_count += 1
        except Exception as e:
            warn("partition", f"Could not read {fname}: {e}")

    ok("partition", f"Read {file_count} partition files from {part_dir}")

    full_ids = set(full_df["entity_id"].tolist())
    missing  = full_ids - collected_ids
    extra    = collected_ids - full_ids

    if not missing:
        ok("partition", f"All {len(full_ids)} entity_ids present in partition files")
    else:
        fail("partition", f"{len(missing)} entity_ids missing from partitions "
                          f"(e.g. {list(missing)[:3]})")

    if not extra:
        ok("partition", "No unexpected entity_ids in partition files")
    else:
        warn("partition", f"{len(extra)} extra entity_ids in partitions "
                          f"(e.g. {list(extra)[:3]})")


def check_partition_files_individually(output_dir: str, split: str) -> None:
    """Run per-file DataFrame checks on every stage1 partition TSV."""
    part_dir = os.path.join(output_dir, "stage1", split)
    if not os.path.isdir(part_dir):
        warn("partition_files", f"Stage1 {split} dir not found: {part_dir}")
        return

    tsv_files = sorted(f for f in os.listdir(part_dir) if f.endswith(".tsv"))
    if not tsv_files:
        warn("partition_files", f"No TSV files found in {part_dir}")
        return

    section(f"Stage1 {split.upper()} — per-file checks ({len(tsv_files)} files)")
    for fname in tsv_files:
        fpath = os.path.join(part_dir, fname)
        try:
            df = load_processed(fpath)
            check_dataframe(df, f"stage1/{split}/{fname}")
        except Exception as e:
            fail("partition_files", f"Failed to load {fname}: {e}")


def check_meta_files(output_dir: str) -> None:
    section("Meta files")
    meta_dir = os.path.join(output_dir, "meta")

    summary_path = os.path.join(meta_dir, "partition_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            ok("meta", f"partition_summary.json loaded: {list(summary.keys())}")
            for split in ["train", "test"]:
                if split in summary:
                    ok("meta", f"  {split}: {len(summary[split])} country partitions")
        except Exception as e:
            fail("meta", f"partition_summary.json parse error: {e}")
    else:
        warn("meta", f"partition_summary.json not found: {summary_path}")

    flags_path = os.path.join(meta_dir, "subgroup_flags.tsv")
    if os.path.exists(flags_path):
        try:
            flags = pd.read_csv(flags_path, sep="\t", dtype=str).fillna("")
            ok("meta", f"subgroup_flags.tsv loaded: {len(flags)} rows, "
                       f"cols={list(flags.columns)}")
        except Exception as e:
            fail("meta", f"subgroup_flags.tsv parse error: {e}")
    else:
        warn("meta", f"subgroup_flags.tsv not found: {flags_path}")


# ============================================================
# SUMMARY
# ============================================================

def print_summary() -> None:
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    passes = sum(1 for s, _, _ in _results if s == PASS)
    fails  = sum(1 for s, _, _ in _results if s == FAIL)
    warns  = sum(1 for s, _, _ in _results if s == WARN)
    total  = len(_results)
    print(f"  Total : {total}")
    print(f"  {PASS} Pass  : {passes}")
    print(f"  {FAIL} Fail  : {fails}")
    print(f"  {WARN} Warn  : {warns}")
    if fails == 0:
        print("\n  ✅ ALL CHECKS PASSED")
    else:
        print(f"\n  ❌ {fails} CHECKS FAILED — see details above")
        print("\n  Failed checks:")
        for s, cat, msg in _results:
            if s == FAIL:
                print(f"    [{cat}] {msg}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate output of normalization_merged.py (Stage 0 + Stage 1)"
    )
    parser.add_argument(
        "--dir", default="processed_data_sample",
        help="Root output directory to validate (default: processed_data_sample)",
    )
    parser.add_argument(
        "--unit-only", action="store_true",
        help="Run unit tests only, skip file checks",
    )
    args = parser.parse_args()

    # ── Unit tests (no files needed)
    test_base_clean()
    test_dedup_consecutive()
    test_dedup_all_tokens()
    test_fix_ocr()
    test_nfkd_aggressive()
    test_dotted_legal_terms()
    test_canonicalize_legal_suffixes()
    test_generate_variants()
    test_script_detection()
    test_transliterate_name()
    test_transliterate_address()
    test_extract_address_fields()

    if not args.unit_only:
        # ── Stage 0 file checks
        stage0_dir = os.path.join(args.dir, "stage0")
        stage0_files = [
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv",
            "test_source1.tsv",
            "test_source2.tsv",
            "test_source3.tsv",
        ]
        for fname in stage0_files:
            fpath = os.path.join(stage0_dir, fname)
            if not os.path.exists(fpath):
                warn("file", f"Not found: {fpath}")
                continue
            df = load_processed(fpath)
            check_dataframe(df, fname)

        # ── Partition coverage: do all entity_ids from stage0 appear in stage1?
        for split, src_fname in [
            ("train", "train_source1.tsv"),
            ("test",  "test_source1.tsv"),
        ]:
            fpath = os.path.join(stage0_dir, src_fname)
            if os.path.exists(fpath):
                df = load_processed(fpath)
                check_partition_coverage(df, args.dir, split, src_fname)
            else:
                warn("partition", f"Source file not found for partition check: {fpath}")

        # ── Per-file checks on every stage1 partition TSV (all 10 000 rows each)
        check_partition_files_individually(args.dir, "train")
        check_partition_files_individually(args.dir, "test")

        # ── Meta files
        check_meta_files(args.dir)

    print_summary()
    sys.exit(0 if all(s != FAIL for s, _, _ in _results) else 1)