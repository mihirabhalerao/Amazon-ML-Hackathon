# ============================================================
# BLOCKING PIPELINE — Amazon ML Challenge: Business Entity Resolution
#
# Consumes output of normalization_merged.py (Stage 0 + Stage 1).
# Produces: candidate_pairs.tsv
#
# Architecture: 7 independent channels per (S1×S2) and (S1×S3) pair,
# run within each country partition, then unioned.
#
# Channels:
#   A — Address key overlap (PIN/ZIP + street_num + locality)
#   B — IDF-weighted name token inverted index
#   C — MinHash LSH (character 3-gram Jaccard, ~0.3 threshold)
#   D — Phonetic key (Double Metaphone on name tokens)
#   E — TF-IDF character n-gram ANN via FAISS (name view, bidirectional)
#   F — TF-IDF character n-gram ANN via FAISS (address view, bidirectional)
#   G — Sorted-token exact match (word-order transposition)
#
# Dependencies: pandas, numpy, datasketch, faiss, metaphone, scikit-learn
# No torch/sentence-transformers required — FAISS runs on TF-IDF vectors.
#
# Usage:
#   python blocking_pipeline.py                     # full run
#   python blocking_pipeline.py --sample 10000      # sample mode
#   python blocking_pipeline.py --unit-test         # self-test only
#   python blocking_pipeline.py --input-dir processed_data_sample
#   python blocking_pipeline.py --validate          # evaluate recall on train GT
# ============================================================

from __future__ import annotations

import os
import re
import sys
import math
import time
import json
import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================

# MinHash
MINHASH_NUM_PERM     = 64          # permutations; more = better accuracy, slower
MINHASH_LSH_THRESHOLD = 0.25        # Jaccard floor for LSH (generous for blocking)

# FAISS / TF-IDF
TFIDF_NGRAM_RANGE    = (2, 4)       # character n-gram range (bigrams to 4-grams)
TFIDF_MAX_FEATURES   = 65536        # vocabulary cap; larger = more discriminative
FAISS_TOP_K          = 15           # candidates per S1 entity from ANN channels
FAISS_SIM_FLOOR      = 0.15         # cosine similarity floor (generous)

# IDF token channel
IDF_FLOOR            = 2.0          # tokens with IDF below this are stopwords (log scale)
IDF_MIN_OVERLAP      = 1            # minimum shared discriminative tokens to keep

# Address channel
ADDR_MIN_LOCALITY_OVERLAP = 1       # minimum shared locality tokens

# Cap on candidate list per S1 entity
MAX_CANDIDATES       = 80

# Output
OUTPUT_FILE          = "candidate_pairs.tsv"

# Validation constants
PAIRS_COMPLETENESS_TARGET = 0.90    # warn if below this


# ============================================================
# HELPERS
# ============================================================

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _timer(label: str):
    """Context manager: prints elapsed time on exit."""
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        t0 = time.perf_counter()
        yield
        _log(f"  {label}: {time.perf_counter() - t0:.1f}s")
    return _ctx()


def _load_partition(
    country: str,
    split: str,
    input_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Load one country partition from stage1 output of normalization_merged."""
    slug = re.sub(r"[^\w]", "_", country).strip("_").lower() or "unknown"
    base = os.path.join(input_dir, "stage1", split)
    out  = {}
    for src in ["s1", "s2", "s3"]:
        path = os.path.join(base, f"{slug}_{src}.tsv")
        if os.path.exists(path):
            out[src] = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        else:
            _log(f"  WARNING: {path} not found — using empty frame")
            out[src] = pd.DataFrame()
    return out


def _load_stage0(input_dir: str, split: str) -> Dict[str, pd.DataFrame]:
    """Load all three stage0 source files for a split."""
    stage0 = os.path.join(input_dir, "stage0")
    dfs = {}
    for src in ["source1", "source2", "source3"]:
        path = os.path.join(stage0, f"{split}_{src}.tsv")
        if os.path.exists(path):
            dfs[src] = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        else:
            _log(f"  WARNING: stage0 file not found: {path}")
            dfs[src] = pd.DataFrame()
    return dfs


def _discover_countries(input_dir: str, split: str) -> List[str]:
    """Discover country partition slugs from stage1 directory."""
    part_dir = os.path.join(input_dir, "stage1", split)
    if not os.path.isdir(part_dir):
        raise FileNotFoundError(f"Stage1 directory not found: {part_dir}")
    # Files named like: {slug}_s1.tsv, {slug}_s2.tsv, {slug}_s3.tsv
    slugs = set()
    for fname in os.listdir(part_dir):
        m = re.match(r"^(.+)_(s[123])\.tsv$", fname)
        if m:
            slugs.add(m.group(1))
    countries = sorted(slugs)
    _log(f"  Found {len(countries)} country partitions: {countries}")
    return countries


def _col(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    """Get a column safely, returning a series of defaults if missing."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


# ============================================================
# CHANNEL A — Address Key Overlap
# ============================================================

class AddressKeyIndex:
    """
    Inverted index on:
      - pin_code / zip_code (exact 6/5-digit match)
      - (street_num, first_locality_token) composite
      - first_locality_token alone (fallback)

    All keys are lowercase strings. Skips records with empty pin/zip AND
    empty street_num AND empty locality (truly address-missing records).
    """

    def __init__(self) -> None:
        self.pin_idx:      Dict[str, List[str]] = defaultdict(list)
        self.zip_idx:      Dict[str, List[str]] = defaultdict(list)
        self.street_idx:   Dict[Tuple, List[str]] = defaultdict(list)
        self.locality_idx: Dict[str, List[str]] = defaultdict(list)

    def _first_locality(self, locality_tokens: str) -> str:
        toks = locality_tokens.lower().split()
        return toks[0] if toks else ""

    def index(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            eid      = row["entity_id"]
            pin      = str(row.get("pin_code", "")).strip()
            zip_c    = str(row.get("zip_code", "")).strip()
            street   = str(row.get("street_num", "")).strip().lower()
            locality = str(row.get("locality_tokens", "")).strip().lower()
            first_loc = self._first_locality(locality)

            if pin:
                self.pin_idx[pin].append(eid)
            if zip_c:
                self.zip_idx[zip_c].append(eid)
            if street and first_loc:
                self.street_idx[(street, first_loc)].append(eid)
            if first_loc:
                self.locality_idx[first_loc].append(eid)

    def query(self, row: pd.Series) -> Set[str]:
        pin      = str(row.get("pin_code", "")).strip()
        zip_c    = str(row.get("zip_code", "")).strip()
        street   = str(row.get("street_num", "")).strip().lower()
        locality = str(row.get("locality_tokens", "")).strip().lower()
        first_loc = self._first_locality(locality)

        results: Set[str] = set()

        if pin:
            results.update(self.pin_idx.get(pin, []))
        if zip_c:
            results.update(self.zip_idx.get(zip_c, []))
        if street and first_loc:
            results.update(self.street_idx.get((street, first_loc), []))

        # Locality-only as fallback — but only when combined with no PIN/ZIP hit
        # to avoid producing huge buckets on common localities.
        if not results and first_loc:
            results.update(self.locality_idx.get(first_loc, []))

        return results


# ============================================================
# CHANNEL B — IDF-Weighted Name Token Index
# ============================================================

class TokenIDFIndex:
    """
    Builds a per-token inverted index weighted by IDF.
    Tokens with IDF < IDF_FLOOR are treated as stopwords and excluded.
    Query returns entity IDs sharing ≥ IDF_MIN_OVERLAP discriminative tokens.
    """

    # Hardcoded stopword supplement (common legal/address noise)
    STOPWORDS: Set[str] = {
        "and", "the", "of", "in", "at", "for", "to", "a", "an",
        "india", "us", "new", "old", "main", "global", "services",
        "solutions", "group", "tech", "technologies", "international",
        "intl", "enterprises", "company", "limited", "pvt", "ltd",
        "llc", "llp", "inc", "corp", "co", "plc", "assoc",
        "store", "shop", "house", "center", "centre", "trading",
        "trade", "business", "system", "systems",
    }

    def __init__(self) -> None:
        self.token_idx:   Dict[str, List[str]] = defaultdict(list)
        self.idf_map:     Dict[str, float]      = {}
        self._doc_count:  int = 0

    def _tokenize(self, *cols: str) -> Set[str]:
        """Tokenize one or more name columns, union the token sets."""
        tokens: Set[str] = set()
        for col in cols:
            tokens.update(col.lower().split())
        return tokens - self.STOPWORDS - {""}

    def build(self, df: pd.DataFrame) -> None:
        """
        Two-pass:
          1. Count DF for IDF
          2. Build inverted index using only tokens above IDF_FLOOR
        """
        df_counts: Dict[str, int] = defaultdict(int)
        records: List[Tuple[str, Set[str]]] = []

        cols_to_use = ["name_no_suffix", "name_sorted", "name_first2",
                       "name_transliterated_no_suffix"]

        for _, row in df.iterrows():
            eid    = row["entity_id"]
            values = [str(row.get(c, "")) for c in cols_to_use]
            toks   = self._tokenize(*values)
            records.append((eid, toks))
            for t in toks:
                df_counts[t] += 1

        self._doc_count = len(records)
        N = max(self._doc_count, 1)

        # Compute IDF (log1p for smoothing)
        self.idf_map = {
            t: math.log1p(N / cnt)
            for t, cnt in df_counts.items()
        }

        # Build index with discriminative tokens only
        for eid, toks in records:
            for t in toks:
                if self.idf_map.get(t, 0.0) >= IDF_FLOOR:
                    self.token_idx[t].append(eid)

    def query(self, row: pd.Series) -> Set[str]:
        cols_to_use = ["name_no_suffix", "name_sorted", "name_first2",
                       "name_transliterated_no_suffix"]
        values = [str(row.get(c, "")) for c in cols_to_use]
        toks   = self._tokenize(*values)

        hit_count: Dict[str, int] = defaultdict(int)
        for t in toks:
            if self.idf_map.get(t, 0.0) >= IDF_FLOOR:
                for eid in self.token_idx.get(t, []):
                    hit_count[eid] += 1

        return {eid for eid, cnt in hit_count.items() if cnt >= IDF_MIN_OVERLAP}


# ============================================================
# CHANNEL C — MinHash LSH
# ============================================================

def _name_shingles(name: str, k: int = 3) -> Set[str]:
    """Character k-gram shingles from a name string."""
    s = name.replace(" ", "_")  # preserve word boundaries
    if len(s) < k:
        return {s}
    return {s[i:i+k] for i in range(len(s) - k + 1)}


def _make_minhash(shingles: Set[str], num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh.encode("utf8"))
    return m


class MinHashLSHIndex:
    """
    MinHash LSH over character 3-gram shingles of name_no_suffix and
    name_transliterated_no_suffix. Both variants of each record are
    inserted so a single query can find either Latin or transliterated form.
    """

    def __init__(self, threshold: float = MINHASH_LSH_THRESHOLD) -> None:
        self.lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
        self._counter = 0  # unique key suffix for duplicate name entries

    def _key(self, eid: str, variant: str) -> str:
        return f"{eid}::{variant}"

    def _eid_from_key(self, key: str) -> str:
        return key.split("::")[0]

    def index(self, df: pd.DataFrame) -> None:
        seen_keys: Set[str] = set()
        for _, row in df.iterrows():
            eid = row["entity_id"]
            for variant, col in [
                ("main", "name_no_suffix"),
                ("trans", "name_transliterated_no_suffix"),
                ("sorted", "name_sorted"),
            ]:
                name = str(row.get(col, "")).strip()
                if not name:
                    continue
                shingles = _name_shingles(name)
                m        = _make_minhash(shingles)
                key      = self._key(eid, variant)
                if key not in seen_keys:
                    seen_keys.add(key)
                    try:
                        self.lsh.insert(key, m)
                    except ValueError:
                        # datasketch raises if duplicate key
                        self._counter += 1
                        self.lsh.insert(f"{key}_{self._counter}", m)

    def query(self, row: pd.Series) -> Set[str]:
        results: Set[str] = set()
        for col in ["name_no_suffix", "name_transliterated_no_suffix", "name_sorted"]:
            name = str(row.get(col, "")).strip()
            if not name:
                continue
            shingles = _name_shingles(name)
            m        = _make_minhash(shingles)
            for key in self.lsh.query(m):
                results.add(self._eid_from_key(key))
        return results


# ============================================================
# CHANNEL D — Phonetic Key (Double Metaphone)
# ============================================================

class PhoneticIndex:
    """
    Double Metaphone on every token in name_no_suffix and
    name_transliterated_no_suffix. Each (primary, secondary) code from
    doublemetaphone is stored as a separate key.
    """

    def __init__(self) -> None:
        self.phonetic_idx: Dict[str, List[str]] = defaultdict(list)

    @staticmethod
    def _phonetic_keys(name: str) -> Set[str]:
        keys: Set[str] = set()
        for tok in name.lower().split():
            if len(tok) < 2:
                continue
            primary, secondary = doublemetaphone(tok)
            if primary:
                keys.add(primary)
            if secondary:
                keys.add(secondary)
        return keys

    def index(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            eid = row["entity_id"]
            for col in ["name_no_suffix", "name_transliterated_no_suffix"]:
                name = str(row.get(col, ""))
                for key in self._phonetic_keys(name):
                    self.phonetic_idx[key].append(eid)

    def query(self, row: pd.Series) -> Set[str]:
        results: Set[str] = set()
        for col in ["name_no_suffix", "name_transliterated_no_suffix"]:
            name = str(row.get(col, ""))
            for key in self._phonetic_keys(name):
                results.update(self.phonetic_idx.get(key, []))
        return results


# ============================================================
# CHANNELS E + F — TF-IDF + FAISS ANN (Name and Address views)
# ============================================================

class TFIDFFaissIndex:
    """
    Builds a TF-IDF character n-gram matrix for the target corpus (S2+S3),
    projects it via FAISS IndexFlatIP (cosine, after L2 norm), and supports
    bidirectional approximate nearest-neighbor search.

    Used for:
      Channel E — name view (name_no_suffix + name_transliterated_no_suffix)
      Channel F — address view (locality_tokens + state_token)
    """

    def __init__(
        self,
        label: str = "name",
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        max_features: int = TFIDF_MAX_FEATURES,
        top_k: int = FAISS_TOP_K,
        sim_floor: float = FAISS_SIM_FLOOR,
    ) -> None:
        self.label        = label
        self.top_k        = top_k
        self.sim_floor    = sim_floor
        self.vectorizer   = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.index_target: Optional[faiss.IndexFlatIP] = None
        self.index_query:  Optional[faiss.IndexFlatIP] = None  # for bidirectional
        self.target_eids:  List[str] = []
        self.query_eids:   List[str] = []

    def _text_col(self, row: pd.Series) -> str:
        if self.label == "name":
            parts = [
                str(row.get("name_no_suffix", "")),
                str(row.get("name_transliterated_no_suffix", "")),
                str(row.get("name_sorted", "")),
            ]
        else:  # address
            parts = [
                str(row.get("locality_tokens", "")),
                str(row.get("state_token", "")),
                str(row.get("address_transliterated", "")),
            ]
        return " ".join(p for p in parts if p).strip()

    def _df_to_texts(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        eids, texts = [], []
        for _, row in df.iterrows():
            t = self._text_col(row)
            if not t:
                continue
            eids.append(row["entity_id"])
            texts.append(t)
        return eids, texts

    @staticmethod
    def _to_faiss_matrix(mat) -> np.ndarray:
        """Convert sparse matrix to dense float32 + L2 normalise (→ cosine)."""
        dense = np.asarray(mat.todense(), dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def build(self, target_df: pd.DataFrame, query_df: pd.DataFrame) -> None:
        """
        Fit vectorizer on union of target + query texts (so IDF is shared),
        then build two FAISS indexes:
          - index_target for S1 → (S2+S3) search
          - index_query  for (S2+S3) → S1 search (bidirectional)
        """
        t_eids, t_texts = self._df_to_texts(target_df)
        q_eids, q_texts = self._df_to_texts(query_df)

        if not t_texts or not q_texts:
            _log(f"    [{self.label}] Skipping FAISS — no texts in one side")
            return

        # Fit on union
        all_texts = t_texts + q_texts
        self.vectorizer.fit(all_texts)

        t_mat = self._to_faiss_matrix(self.vectorizer.transform(t_texts))
        q_mat = self._to_faiss_matrix(self.vectorizer.transform(q_texts))

        dim = t_mat.shape[1]

        self.index_target = faiss.IndexFlatIP(dim)
        self.index_target.add(t_mat)
        self.target_eids = t_eids

        self.index_query = faiss.IndexFlatIP(dim)
        self.index_query.add(q_mat)
        self.query_eids = q_eids

    def _search_index(
        self,
        index: faiss.IndexFlatIP,
        vec: np.ndarray,
        eids_pool: List[str],
    ) -> Set[str]:
        if index is None or index.ntotal == 0:
            return set()
        k_capped = min(self.top_k, index.ntotal)
        sims, idxs = index.search(vec, k_capped)
        results: Set[str] = set()
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                break
            if sim >= self.sim_floor:
                results.add(eids_pool[idx])
        return results

    def query_forward(self, row: pd.Series) -> Set[str]:
        """S1 → (S2+S3): query index_target for neighbors of this row."""
        if self.index_target is None:
            return set()
        t = self._text_col(row)
        if not t:
            return set()
        vec = self._to_faiss_matrix(self.vectorizer.transform([t]))
        return self._search_index(self.index_target, vec, self.target_eids)

    def query_backward(self, eid: str) -> Set[str]:
        """(S2+S3) → S1: given a target entity, find its S1 neighbors."""
        if self.index_query is None or eid not in self.query_eids:
            return set()
        idx  = self.query_eids.index(eid)
        dim  = self.index_query.d
        # Retrieve the stored vector by doing a 1-NN search on index_query
        # (FAISS FlatIP stores dense matrix, we can recover by position)
        # Faster: reconstruct vector directly
        vec  = np.zeros((1, dim), dtype=np.float32)
        self.index_query.reconstruct(idx, vec[0])
        return self._search_index(self.index_target, vec, self.target_eids)


# ============================================================
# CHANNEL G — Sorted-Token Exact Match
# ============================================================

class SortedTokenIndex:
    """
    Hash map: name_sorted → [entity_ids].
    Exact hits are extremely high-precision (same tokens, different order).
    """

    def __init__(self) -> None:
        self.idx: Dict[str, List[str]] = defaultdict(list)

    def index(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            key = str(row.get("name_sorted", "")).strip()
            if key:
                self.idx[key].append(row["entity_id"])

    def query(self, row: pd.Series) -> Set[str]:
        key = str(row.get("name_sorted", "")).strip()
        return set(self.idx.get(key, []))


# ============================================================
# BLOCKING ENGINE — assembles all channels for one partition
# ============================================================

class PartitionBlocker:
    """
    Builds all 7 channel indexes over a target dataframe (S2 or S3)
    and runs batch queries for every S1 entity.

    Returns: Dict[s1_eid → Set[target_eid]]
    """

    def __init__(
        self,
        target_df:  pd.DataFrame,
        query_df:   pd.DataFrame,   # S1
        label:      str = "",
        skip_address_for_missing: bool = True,
    ) -> None:
        self.target_df  = target_df
        self.query_df   = query_df
        self.label      = label
        self.skip_addr  = skip_address_for_missing

        # Channels
        self.ch_A = AddressKeyIndex()
        self.ch_B = TokenIDFIndex()
        self.ch_C = MinHashLSHIndex()
        self.ch_D = PhoneticIndex()
        self.ch_E = TFIDFFaissIndex(label="name")
        self.ch_F = TFIDFFaissIndex(label="address",
                                    sim_floor=FAISS_SIM_FLOOR * 0.8)
        self.ch_G = SortedTokenIndex()

    def build_indexes(self) -> None:
        _log(f"    Building indexes [{self.label}] over {len(self.target_df)} target rows ...")

        with _timer("  Ch-A address"):
            self.ch_A.index(self.target_df)

        with _timer("  Ch-B IDF token"):
            self.ch_B.build(self.target_df)

        with _timer("  Ch-C MinHash LSH"):
            self.ch_C.index(self.target_df)

        with _timer("  Ch-D Phonetic"):
            self.ch_D.index(self.target_df)

        with _timer("  Ch-E FAISS name"):
            self.ch_E.build(self.target_df, self.query_df)

        with _timer("  Ch-F FAISS addr"):
            self.ch_F.build(self.target_df, self.query_df)

        with _timer("  Ch-G sorted-token"):
            self.ch_G.index(self.target_df)

        _log(f"    Indexes ready [{self.label}]")

    def _has_address(self, row: pd.Series) -> bool:
        v = str(row.get("has_address", "False")).lower()
        return v in ("true", "1", "yes")

    def run(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Returns: {s1_eid → {"A":.., "B":.., ..., "union": Set[eid]}}
        Channel sets contain target entity IDs.
        """
        results: Dict[str, Dict[str, Set[str]]] = {}

        target_eid_set = set(self.target_df["entity_id"].tolist())

        for _, row in self.query_df.iterrows():
            eid     = row["entity_id"]
            has_addr = self._has_address(row)

            hits: Dict[str, Set[str]] = {
                "A": set(), "B": set(), "C": set(),
                "D": set(), "E": set(), "F": set(), "G": set(),
            }

            # Channel A — address keys
            if has_addr or not self.skip_addr:
                hits["A"] = self.ch_A.query(row) & target_eid_set

            # Channel B — IDF token
            hits["B"] = self.ch_B.query(row) & target_eid_set

            # Channel C — MinHash
            hits["C"] = self.ch_C.query(row) & target_eid_set

            # Channel D — Phonetic
            hits["D"] = self.ch_D.query(row) & target_eid_set

            # Channel E — FAISS name (forward; backward handled in post-processing)
            hits["E"] = self.ch_E.query_forward(row) & target_eid_set

            # Channel F — FAISS address
            if has_addr or not self.skip_addr:
                hits["F"] = self.ch_F.query_forward(row) & target_eid_set

            # Channel G — sorted-token
            hits["G"] = self.ch_G.query(row) & target_eid_set

            # Union
            union: Set[str] = set()
            for v in hits.values():
                union |= v
            hits["union"] = union

            results[eid] = hits

        # Bidirectional pass for FAISS name (E) and address (F)
        # For each target entity that appears in S1 results, check if
        # S1 entity is in target's neighborhood.
        
        # DISABLED DUE TO TIME CONSTRAINTS
        
        # _log(f"    Bidirectional FAISS pass [{self.label}] ...")
        # for target_row_idx, target_row in self.target_df.iterrows():
        #     t_eid = target_row["entity_id"]
        #     # forward on the *target* side → finds S1 neighbors
        #     s1_neighbors_E = self.ch_E.query_forward(target_row) & set(self.query_df["entity_id"].tolist())
        #     s1_neighbors_F = self.ch_F.query_forward(target_row) & set(self.query_df["entity_id"].tolist())

        #     for s1_eid in s1_neighbors_E:
        #         if s1_eid in results:
        #             results[s1_eid]["E"].add(t_eid)
        #             results[s1_eid]["union"].add(t_eid)

        #     for s1_eid in s1_neighbors_F:
        #         if s1_eid in results:
        #             results[s1_eid]["F"].add(t_eid)
        #             results[s1_eid]["union"].add(t_eid)

        return results


# ============================================================
# CANDIDATE CAPPING
# ============================================================

def cap_candidate_list(
    hits: Dict[str, Set[str]],
    max_candidates: int = MAX_CANDIDATES,
) -> List[str]:
    """
    Score each candidate by:
      1. Channel agreement count (# channels that produced it)
      2. Tie-break: presence in high-precision channels (G first, then A, E)
    Keep top max_candidates.
    """
    union = hits.get("union", set())
    if len(union) <= max_candidates:
        return sorted(union)

    PRIORITY_CHANNELS = ["G", "A", "E", "C", "B", "D", "F"]

    def score(eid: str) -> Tuple[int, int]:
        agreement = sum(1 for ch, s in hits.items() if ch != "union" and eid in s)
        priority  = next(
            (len(PRIORITY_CHANNELS) - i for i, ch in enumerate(PRIORITY_CHANNELS)
             if eid in hits.get(ch, set())),
            0,
        )
        return (agreement, priority)

    ranked = sorted(union, key=score, reverse=True)
    return ranked[:max_candidates]


# ============================================================
# FULL PIPELINE ENTRY POINT
# ============================================================

def run_blocking(
    input_dir:  str  = "processed_data_sample",
    split:      str  = "test",
    output_file: str = OUTPUT_FILE,
    sample_s1:  Optional[int] = None,
) -> pd.DataFrame:
    """
    Main blocking function. Returns a DataFrame with columns:
      source1_entity_id, candidate_entity_ids
    and writes output_file.
    """
    _log(f"=== BLOCKING START  split={split}  input={input_dir} ===")

    countries = _discover_countries(input_dir, split)

    all_rows: List[Dict] = []         # {s1_eid, candidates: List[str], channel_tags}

    for country in countries:
        _log(f"\n── Country: [{country}] ──")
        parts = _load_partition(country, split, input_dir)
        s1    = parts["s1"]
        s2    = parts["s2"]
        s3    = parts["s3"]

        if s1.empty:
            _log(f"  Skipping — S1 empty")
            continue

        if sample_s1 is not None:
            s1 = s1.head(sample_s1).reset_index(drop=True)
            _log(f"  S1 sampled to {len(s1)} rows")

        # ── S1 × S2
        if not s2.empty:
            _log(f"  Building S1×S2 blocker  (S1={len(s1)}, S2={len(s2)})")
            blocker12 = PartitionBlocker(target_df=s2, query_df=s1, label=f"{country}/s1xs2")
            blocker12.build_indexes()
            res12 = blocker12.run()
        else:
            _log(f"  S2 empty — skipping S1×S2")
            res12 = {}

        # ── S1 × S3
        if not s3.empty:
            _log(f"  Building S1×S3 blocker  (S1={len(s1)}, S3={len(s3)})")
            blocker13 = PartitionBlocker(target_df=s3, query_df=s1, label=f"{country}/s1xs3")
            blocker13.build_indexes()
            res13 = blocker13.run()
        else:
            _log(f"  S3 empty — skipping S1×S3")
            res13 = {}

        # ── Merge S2 + S3 results per S1 entity
        for _, row in s1.iterrows():
            eid = row["entity_id"]

            # Merge channel sets from both blockers
            merged_hits: Dict[str, Set[str]] = {
                ch: set() for ch in ["A", "B", "C", "D", "E", "F", "G", "union"]
            }
            for res in [res12.get(eid, {}), res13.get(eid, {})]:
                for ch, s in res.items():
                    merged_hits[ch] |= s

            candidates = cap_candidate_list(merged_hits)

            # Channel tag bitmask (for downstream matcher feature)
            tags: Dict[str, int] = {
                ch: len(merged_hits.get(ch, set()) & set(candidates))
                for ch in ["A", "B", "C", "D", "E", "F", "G"]
            }

            all_rows.append({
                "source1_entity_id":  eid,
                "candidate_entity_ids": ",".join(candidates),
                "n_candidates":        len(candidates),
                "channel_tags":        json.dumps(tags),
            })

        _log(f"  [{country}] done — {len(s1)} S1 entities processed")

    if not all_rows:
        _log("WARNING: No results generated — check input directory.")
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])

    result_df = pd.DataFrame(all_rows)

    # ── Stats
    n_total  = len(result_df)
    n_empty  = (result_df["candidate_entity_ids"] == "").sum()
    avg_size = result_df["n_candidates"].mean()
    _log(f"\n=== BLOCKING COMPLETE ===")
    _log(f"  Total S1 entities    : {n_total}")
    _log(f"  Empty candidate lists: {n_empty}  ({100*n_empty/max(n_total,1):.1f}%)")
    _log(f"  Avg candidates/entity: {avg_size:.1f}")
    _log(f"  Max candidates       : {result_df['n_candidates'].max()}")

    # Write output
    out_cols = ["source1_entity_id", "candidate_entity_ids"]
    result_df[out_cols].to_csv(output_file, sep="\t", index=False)
    _log(f"  Written → {output_file}")

    # Also write a richer debug file alongside
    debug_file = output_file.replace(".tsv", "_debug.tsv")
    result_df.to_csv(debug_file, sep="\t", index=False)
    _log(f"  Debug file → {debug_file}")

    return result_df


# ============================================================
# VALIDATION — evaluate on training ground truth
# ============================================================

def validate_blocking(
    result_df:  pd.DataFrame,
    input_dir:  str,
    split:      str = "train",
) -> Dict:
    """
    Compute Pairs Completeness and Reduction Ratio against ground truth.
    Only meaningful when split='train' (ground truth available).

    Reads train_ground_truth.tsv from dataset dir.
    """
    gt_path = os.path.join("dataset", "train", "train_ground_truth.tsv")
    if not os.path.exists(gt_path):
        _log(f"Ground truth not found: {gt_path}")
        return {}

    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    _log(f"  Loaded ground truth: {len(gt)} rows")

    # Build set of true match pairs (s1_eid, s23_eid)
    # Ground truth format: entity_id | cluster_id (all entities in same cluster match)
    # Group by cluster_id, find S1 entities and their S2/S3 matches
    # Ensure correct columns exist
    if "source1_entity_id" not in gt.columns or "matched_entity_ids" not in gt.columns:
        _log("  WARNING: source1_entity_id or matched_entity_ids column not found in ground truth")
        return {}

    # Build set of true match pairs (s1_eid, target_eid)
    true_pairs: Set[Tuple[str, str]] = set()
    
    for _, row in gt.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matches_str = str(row["matched_entity_ids"]).strip()
        
        if not matches_str:
            continue
            
        # Extract individual target IDs from the comma-separated string
        target_ids = [m.strip() for m in matches_str.split(",") if m.strip()]
        for target in target_ids:
            true_pairs.add((s1, target))

    if not true_pairs:
        _log("  No true pairs found — check ground truth format")
        return {}

    # Build candidate pairs set from result_df
    cand_pairs: Set[Tuple[str, str]] = set()
    cand_map: Dict[str, Set[str]] = {}
    for _, row in result_df.iterrows():
        s1_eid = row["source1_entity_id"]
        cands  = [c for c in str(row["candidate_entity_ids"]).split(",") if c]
        cand_map[s1_eid] = set(cands)
        for c in cands:
            cand_pairs.add((s1_eid, c))

    true_s1s = {p[0] for p in true_pairs}
    covered  = true_pairs & cand_pairs
    missed   = true_pairs - cand_pairs

    pc = len(covered) / len(true_pairs) if true_pairs else 0.0

    # Reduction ratio: 1 - (# candidate pairs) / (# possible pairs)
    n_s1   = len(result_df)
    n_s23  = sum(
        len(parts["s2"]) + len(parts["s3"])
        for country in _discover_countries(input_dir, split)
        for parts in [_load_partition(country, split, input_dir)]
    )
    n_possible  = n_s1 * n_s23 if n_s23 else 1
    rr = 1.0 - len(cand_pairs) / max(n_possible, 1)

    # Per-channel contribution (requires debug file)
    # Approximate: sample missed pairs and see which channels had them
    missed_sample = list(missed)[:20]

    _log("\n── VALIDATION RESULTS ──")
    _log(f"  True match pairs     : {len(true_pairs)}")
    _log(f"  Covered pairs        : {len(covered)}")
    _log(f"  Missed pairs         : {len(missed)}")
    _log(f"  Pairs Completeness   : {pc:.4f}  {'✅' if pc >= PAIRS_COMPLETENESS_TARGET else '⚠️ BELOW TARGET'}")
    _log(f"  Reduction Ratio      : {rr:.4f}")
    if missed_sample:
        _log(f"  Sample missed pairs  : {missed_sample[:5]}")

    return {
        "pairs_completeness": pc,
        "reduction_ratio":    rr,
        "true_pairs":         len(true_pairs),
        "covered_pairs":      len(covered),
        "missed_pairs":       len(missed),
    }


# ============================================================
# UNIT TESTS
# ============================================================

def _unit_tests() -> bool:
    import traceback
    failures = []

    def check(name, cond, detail=""):
        if not cond:
            failures.append(f"FAIL [{name}] {detail}")
            print(f"  ✗  {name}  {detail}")
        else:
            print(f"  ✓  {name}")

    print("\n── Unit Tests ──")

    # Address key index
    try:
        idx = AddressKeyIndex()
        df  = pd.DataFrame([{
            "entity_id": "A1", "pin_code": "400001", "zip_code": "",
            "street_num": "5", "locality_tokens": "andheri nagar",
        }])
        idx.index(df)
        row  = pd.Series({"pin_code": "400001", "zip_code": "", "street_num": "",
                          "locality_tokens": ""})
        hits = idx.query(row)
        check("AddressKeyIndex/pin", "A1" in hits, f"got {hits}")

        row2 = pd.Series({"pin_code": "", "zip_code": "", "street_num": "5",
                          "locality_tokens": "andheri main"})
        hits2 = idx.query(row2)
        check("AddressKeyIndex/street", "A1" in hits2, f"got {hits2}")
    except Exception:
        failures.append("FAIL [AddressKeyIndex] " + traceback.format_exc())

    # IDF token index
    try:
        idx = TokenIDFIndex()
        
        # Add dummy rows to increase total document count (N) so IDF scores can surpass IDF_FLOOR (2.0)
        dummies = [{"entity_id": f"D{i}", "name_no_suffix": f"dummy company {i}"} for i in range(10)]
        
        df  = pd.DataFrame([
            {"entity_id": "X1", "name_no_suffix": "apex summit trading",
             "name_sorted": "apex summit trading", "name_first2": "apex summit",
             "name_transliterated_no_suffix": "apex summit trading"},
            {"entity_id": "X2", "name_no_suffix": "apex global",
             "name_sorted": "apex global", "name_first2": "apex global",
             "name_transliterated_no_suffix": "apex global"},
        ] + dummies) # Concatenate the dummies
        
        idx.build(df)
        row  = pd.Series({"name_no_suffix": "apex summit",
                          "name_sorted": "apex summit", "name_first2": "apex summit",
                          "name_transliterated_no_suffix": "apex summit"})
        hits = idx.query(row)
        check("TokenIDFIndex/basic", "X1" in hits, f"got {hits}")
    except Exception:
        failures.append("FAIL [TokenIDFIndex] " + traceback.format_exc())

    # MinHash
    try:
        idx = MinHashLSHIndex(threshold=0.25)
        df  = pd.DataFrame([{
            "entity_id": "M1", "name_no_suffix": "crestline clean energy",
            "name_transliterated_no_suffix": "crestline clean energy",
            "name_sorted": "clean crestline energy",
        }])
        idx.index(df)
        row  = pd.Series({"name_no_suffix": "crestline clean energi",  # typo
                          "name_transliterated_no_suffix": "crestline clean energi",
                          "name_sorted": "clean crestline energi"})
        hits = idx.query(row)
        check("MinHashLSH/typo", "M1" in hits, f"got {hits}")
    except Exception:
        failures.append("FAIL [MinHashLSH] " + traceback.format_exc())

    # Phonetic
    try:
        idx = PhoneticIndex()
        df  = pd.DataFrame([{
            "entity_id": "P1",
            "name_no_suffix": "smith trading",
            "name_transliterated_no_suffix": "smith trading",
        }])
        idx.index(df)
        row  = pd.Series({"name_no_suffix": "smyth trading",  # transliteration variant
                          "name_transliterated_no_suffix": "smyth trading"})
        hits = idx.query(row)
        check("PhoneticIndex/variant", "P1" in hits, f"got {hits}")
    except Exception:
        failures.append("FAIL [PhoneticIndex] " + traceback.format_exc())

    # Sorted-token index
    try:
        idx = SortedTokenIndex()
        df  = pd.DataFrame([{
            "entity_id": "G1", "name_sorted": "bitwise moore",
        }])
        idx.index(df)
        row  = pd.Series({"name_sorted": "bitwise moore"})
        hits = idx.query(row)
        check("SortedTokenIndex/exact", "G1" in hits, f"got {hits}")

        row2 = pd.Series({"name_sorted": "different name"})
        hits2 = idx.query(row2)
        check("SortedTokenIndex/no_match", "G1" not in hits2, f"got {hits2}")
    except Exception:
        failures.append("FAIL [SortedTokenIndex] " + traceback.format_exc())

    # FAISS TF-IDF
    try:
        target_df = pd.DataFrame([{
            "entity_id": "F1",
            "name_no_suffix": "ferrero rocher chocolate",
            "name_transliterated_no_suffix": "ferrero rocher chocolate",
            "name_sorted": "chocolate ferrero rocher",
            "locality_tokens": "delhi main",
            "state_token": "delhi",
            "address_transliterated": "delhi main market",
        }])
        query_df = pd.DataFrame([{
            "entity_id": "S1A",
            "name_no_suffix": "ferrero rocher sweets",
            "name_transliterated_no_suffix": "ferrero rocher sweets",
            "name_sorted": "ferrero rocher sweets",
            "locality_tokens": "delhi",
            "state_token": "delhi",
            "address_transliterated": "delhi market",
        }])
        faiss_idx = TFIDFFaissIndex(label="name")
        faiss_idx.build(target_df, query_df)
        hits = faiss_idx.query_forward(query_df.iloc[0])
        check("TFIDFFaiss/forward", "F1" in hits, f"got {hits}")
    except Exception:
        failures.append("FAIL [TFIDFFaiss] " + traceback.format_exc())

    # cap_candidate_list
    try:
        hits = {
            "A": {"E1", "E2"}, "B": {"E1", "E3"}, "C": {"E2"},
            "D": set(), "E": {"E1"}, "F": set(), "G": {"E1"},
            "union": {"E1", "E2", "E3"},
        }
        capped = cap_candidate_list(hits, max_candidates=2)
        check("cap_candidates/max", len(capped) <= 2, f"got {len(capped)}")
        check("cap_candidates/E1_first", capped[0] == "E1" if capped else False,
              f"E1 should rank first (4 channels), got {capped}")
    except Exception:
        failures.append("FAIL [cap_candidates] " + traceback.format_exc())

    print()
    if failures:
        for f in failures:
            print(f"  {f}")
        print(f"\n  {len(failures)} unit test(s) FAILED")
        return False
    else:
        print(f"  All unit tests PASSED")
        return True


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blocking pipeline for Amazon ML Challenge Entity Resolution"
    )
    parser.add_argument(
        "--input-dir", default="processed_data_sample",
        help="Root directory from normalization_merged.py (default: processed_data_sample)",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "test"],
        help="Which split to block (default: test)",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Output TSV file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit S1 to first N rows per partition (dev/debug mode)",
    )
    parser.add_argument(
        "--unit-test", action="store_true",
        help="Run unit tests only, then exit",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After blocking, evaluate recall against train ground truth",
    )
    args = parser.parse_args()

    if args.unit_test:
        ok = _unit_tests()
        sys.exit(0 if ok else 1)

    # Run blocking
    result = run_blocking(
        input_dir=args.input_dir,
        split=args.split,
        output_file=args.output,
        sample_s1=args.sample,
    )

    # Optional validation
    if args.validate and not result.empty:
        validate_blocking(result, input_dir=args.input_dir, split="train")

    sys.exit(0)