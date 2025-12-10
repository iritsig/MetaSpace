#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from ripser import ripser
from persim.persistent_entropy import persistent_entropy
from gudhi.bottleneck import bottleneck_distance
from gudhi.wasserstein import wasserstein_distance
from scipy.linalg import svd

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist


from pathlib import Path

Root_project = Path(__file__).parent
# -*- coding: utf-8 -*-
"""
Syntax string features (purement Python, sans dépendances externes).
Calcule un ensemble de distances/similarités entre deux libellés (noms d'attributs).
Toutes les features sont préfixées par 'syn_' pour éviter les collisions.
"""

import math
import unicodedata
import re
from collections import Counter

# --------------------- utils ---------------------

_rx_non_alnum = re.compile(r"[^0-9a-z]+", flags=re.IGNORECASE)

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = _strip_accents(str(s)).lower().strip()
    s = _rx_non_alnum.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def char_ngrams(s: str, n: int):
    s = normalize_text(s).replace(" ", "")
    if n <= 0 or len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union

def dice(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return (2 * inter) / (len(A) + len(B))

def cosine_counts(a, b):
    A, B = Counter(a), Counter(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    # dot
    dot = sum(A[k] * B.get(k, 0) for k in A)
    # norms
    na = math.sqrt(sum(v*v for v in A.values()))
    nb = math.sqrt(sum(v*v for v in B.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def common_prefix_len(a: str, b: str):
    a, b = a or "", b or ""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i

def common_suffix_len(a: str, b: str):
    a, b = a or "", b or ""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[-(i+1)] == b[-(i+1)]:
        i += 1
    return i

# --------------------- distances ---------------------

def levenshtein(a: str, b: str):
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    # DP ligne par ligne
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ai = a[i-1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j-1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j-1] + 1,    # insertion
                prev[j-1] + cost  # substitute
            )
        prev, curr = curr, prev
    return prev[lb]

def osa_damerau(a: str, b: str):
    """Optimal String Alignment (transpositions adjacentes)."""
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    d = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): d[i][0] = i
    for j in range(lb+1): d[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + 1)
    return d[la][lb]

def lcs_len(a: str, b: str):
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    # DP O(min(la,lb)) mémoire
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = [0]*(lb+1)
    for i in range(1, la+1):
        curr = [0]*(lb+1)
        ai = a[i-1]
        for j in range(1, lb+1):
            if ai == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[lb]

def jaro(a: str, b: str):
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    match_dist = max(0, (max(la, lb) // 2) - 1)
    a_flags = [False]*la
    b_flags = [False]*lb

    matches = 0
    transpositions = 0

    # count matches
    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if not b_flags[j] and a[i] == b[j]:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    # count transpositions
    k = 0
    for i in range(la):
        if a_flags[i]:
            while not b_flags[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2

    return (matches/la + matches/lb + (matches - transpositions)/matches) / 3.0

def jaro_winkler(a: str, b: str, p=0.1, max_l=4):
    ja = jaro(a, b)
    # common prefix length up to max_l
    l = 0
    na, nb = a or "", b or ""
    mx = min(max_l, len(na), len(nb))
    while l < mx and na[l] == nb[l]:
        l += 1
    return ja + l * p * (1 - ja)

# --------------------- feature pack ---------------------

def compute_syntax_string_features(name1: str, name2: str, feature_name: str = None) -> dict:
    """
    Calcule une seule feature syntaxique si `feature_name` est précisé,
    sinon calcule toutes les features syntaxiques.
    Chronomètre et enregistre le temps de la feature demandée.
    """
    start_time = time.perf_counter()

    raw_a = str(name1 or "")
    raw_b = str(name2 or "")
    a = normalize_text(raw_a)
    b = normalize_text(raw_b)

    # tokens & n-grams
    tok_a, tok_b = tokens(raw_a), tokens(raw_b)
    big_a, big_b = char_ngrams(raw_a, 2), char_ngrams(raw_b, 2)
    tri_a, tri_b = char_ngrams(raw_a, 3), char_ngrams(raw_b, 3)

    maxlen = max(len(a), len(b), 1)

    # Si on demande une seule feature, calcul ciblé :
    if feature_name:
        val = None

        try:
            if feature_name == "syn_len_a":
                val = float(len(a))
            elif feature_name == "syn_len_b":
                val = float(len(b))
            elif feature_name == "syn_equal_exact":
                val = 1.0 if raw_a == raw_b else 0.0
            elif feature_name == "syn_equal_casefold":
                val = 1.0 if raw_a.casefold() == raw_b.casefold() else 0.0
            elif feature_name == "syn_contains_a_in_b":
                val = 1.0 if a and a in b else 0.0
            elif feature_name == "syn_contains_b_in_a":
                val = 1.0 if b and b in a else 0.0

            elif feature_name == "syn_levenshtein":
                val = float(levenshtein(a, b))
            elif feature_name == "syn_levenshtein_sim":
                lev = levenshtein(a, b)
                val = 1.0 - (lev / maxlen)

            elif feature_name == "syn_damerau_osa":
                val = float(osa_damerau(a, b))
            elif feature_name == "syn_damerau_osa_sim":
                osa = osa_damerau(a, b)
                val = 1.0 - (osa / maxlen)

            elif feature_name == "syn_lcs_len":
                val = float(lcs_len(a, b))
            elif feature_name == "syn_lcs_ratio":
                lcs = lcs_len(a, b)
                val = lcs / maxlen

            elif feature_name == "syn_common_prefix_ratio":
                cp = common_prefix_len(a, b)
                val = cp / maxlen
            elif feature_name == "syn_common_suffix_ratio":
                cs = common_suffix_len(a, b)
                val = cs / maxlen

            elif feature_name == "syn_jaccard_tokens":
                val = jaccard(tok_a, tok_b)
            elif feature_name == "syn_dice_tokens":
                val = dice(tok_a, tok_b)

            elif feature_name == "syn_jaccard_bigrams":
                val = jaccard(big_a, big_b)
            elif feature_name == "syn_jaccard_trigrams":
                val = jaccard(tri_a, tri_b)
            elif feature_name == "syn_cosine_bigrams":
                val = cosine_counts(big_a, big_b)
            elif feature_name == "syn_cosine_trigrams":
                val = cosine_counts(tri_a, tri_b)

            elif feature_name == "syn_jaro":
                val = jaro(a, b)
            elif feature_name == "syn_jaro_winkler":
                val = jaro_winkler(a, b)

            else:
                raise ValueError(f"Feature syntaxique inconnue : {feature_name}")

        except Exception as e:
            print(f"[!] Échec du calcul de {feature_name}: {e}")
            val = np.nan

        elapsed = time.perf_counter() - start_time

        # Log du temps uniquement pour cette feature
        try:
            df = pd.DataFrame([{
                "Metric": "syntactic",
                "Part": "total",
                "Time_sec": elapsed,
                "Feature": feature_name,
                "Attribute1": name1,
                "Attribute2": name2,
            }])

        except Exception as e:
            print(f"[profiling log failed for syntax feature {feature_name}] {e}")

        return {feature_name: val}

    # Sinon, calcul complet de toutes les features (comportement original)
    else:
        # Reprend ton code précédent intégralement ↓
        lev = levenshtein(a, b)
        osa = osa_damerau(a, b)
        lcs = lcs_len(a, b)
        cp  = common_prefix_len(a, b)
        cs  = common_suffix_len(a, b)

        lcs_ratio = lcs / maxlen
        cp_ratio  = cp / maxlen
        cs_ratio  = cs / maxlen

        jacc_tok   = jaccard(tok_a, tok_b)
        dice_tok   = dice(tok_a, tok_b)
        jac_bi     = jaccard(big_a, big_b)
        jac_tri    = jaccard(tri_a, tri_b)
        cos_bi     = cosine_counts(big_a, big_b)
        cos_tri    = cosine_counts(tri_a, tri_b)
        jaro_s     = jaro(a, b)
        jw_s       = jaro_winkler(a, b)

        lev_sim = 1.0 - (lev / maxlen)
        osa_sim = 1.0 - (osa / maxlen)

        eq_exact    = 1.0 if raw_a == raw_b else 0.0
        eq_casefold = 1.0 if raw_a.casefold() == raw_b.casefold() else 0.0
        a_in_b      = 1.0 if a and a in b else 0.0
        b_in_a      = 1.0 if b and b in a else 0.0

        result = {
            "syn_len_a": float(len(a)),
            "syn_len_b": float(len(b)),
            "syn_equal_exact": eq_exact,
            "syn_equal_casefold": eq_casefold,
            "syn_contains_a_in_b": a_in_b,
            "syn_contains_b_in_a": b_in_a,
            "syn_levenshtein": float(lev),
            "syn_levenshtein_sim": lev_sim,
            "syn_damerau_osa": float(osa),
            "syn_damerau_osa_sim": osa_sim,
            "syn_lcs_len": float(lcs),
            "syn_lcs_ratio": lcs_ratio,
            "syn_common_prefix_ratio": cp_ratio,
            "syn_common_suffix_ratio": cs_ratio,
            "syn_jaccard_tokens": jacc_tok,
            "syn_dice_tokens": dice_tok,
            "syn_jaccard_bigrams": jac_bi,
            "syn_jaccard_trigrams": jac_tri,
            "syn_cosine_bigrams": cos_bi,
            "syn_cosine_trigrams": cos_tri,
            "syn_jaro": jaro_s,
            "syn_jaro_winkler": jw_s,
        }

        elapsed = time.perf_counter() - start_time
        try:
            df = pd.DataFrame([{
                "Metric": "syntactic",
                "Part": "total",
                "Time_sec": elapsed,
                "Feature": "syntactic_all",
                "Attribute1": name1,
                "Attribute2": name2,
            }])

        except Exception as e:
            print(f"[profiling log failed for all syntax features] {e}")

        return result




def vector_to_point_cloud(v, window_size=5):
    return np.array([v[i:i+window_size] for i in range(len(v) - window_size)])


def compute_classical_distances(v1, v2, name):
    if name == "Euclidean":
        return np.linalg.norm(v1 - v2)
    elif name == "Cosine":
        return cdist([v1], [v2], metric="cosine")[0][0]
    elif name == "Pearson":
        return pearsonr(v1, v2)[0]
    elif name == "Spearman":
        return spearmanr(v1, v2)[0]
    elif name == "Minkowski":
        return cdist([v1], [v2], metric="minkowski", p=3)[0][0]
    elif name == "Canberra":
        return cdist([v1], [v2], metric="canberra")[0][0]
    elif name == "Chebyshev":
        return cdist([v1], [v2], metric="chebyshev")[0][0]




def alpha_reQ(matrix):
    U, S, _ = svd(matrix, full_matrices=False)
    log_indices = np.log(np.arange(1, len(S) + 1))
    log_singular_values = np.log(S)
    slope, _ = np.polyfit(log_indices, log_singular_values, 1)
    return -slope

def NESum(matrix):
    C = np.cov(matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = sorted(eigenvalues, reverse=True)
    lambda_0 = eigenvalues[0] if eigenvalues[0] > 0 else 1e-6
    return np.sum(eigenvalues / lambda_0)

def rank_me(matrix):
    _, S, _ = svd(matrix, full_matrices=False)
    S_normalized = S / np.sum(S)
    return np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-12)))

def stable_rank(matrix):
    frobenius_norm = np.linalg.norm(matrix, 'fro')
    spectral_norm = np.linalg.norm(matrix, 2)
    return (frobenius_norm ** 2) / (spectral_norm ** 2)

def self_cluster(matrix):
    n, d = matrix.shape
    dot_product = matrix @ matrix.T
    frobenius_norm = np.linalg.norm(dot_product, 'fro')
    return (d * frobenius_norm - n * (d + n - 1)) / ((d - 1) * (n - 1) * n)

# ==========================================================
# === Fonction atomique de calcul (une seule méta-feature)
# ==========================================================
def compute_single_feature(name, v1, v2, mat1, mat2, a1, a2, i, j,
                           clouds=None,
                           dgms_cache_A1=None, 
                           dgms_cache_A2=None,
                           dgms_cache_i=None,
                           dgms_cache_j=None,
                           dgms_cache_ij=None):
    """Calcule une méta-feature unique"""
    if clouds is None:
        clouds = (vector_to_point_cloud(v1), vector_to_point_cloud(v2))

    start_total = time.perf_counter()
    val = np.nan
    diagram_time = 0.0

    try:
        if name in {"Euclidean", "Cosine", "Pearson", "Spearman", "Minkowski", "Canberra", "Chebyshev"}:
            val = compute_classical_distances(v1, v2, name)

        elif name.startswith(("alpha_ReQ_", "NESum_", "RankMe_", "StableRank_", "SelfCluster_")):
            key = name.split("_")[-1]
            data_map = {"a1": clouds[0], "a2": clouds[1], "A1": mat1, "A2": mat2}
            func_map = {
                "alpha_ReQ_": alpha_reQ,
                "NESum_": NESum,
                "RankMe_": rank_me,
                "StableRank_": stable_rank,
                "SelfCluster_": self_cluster,
            }
            prefix = next(p for p in func_map if name.startswith(p))
            val = func_map[prefix](data_map[key])

        elif any(k in name for k in ["h0_", "h1_", "bottleneck", "wasserstein", "entropy"]):
            available_metrics = ["euclidean", "cosine", "manhattan"]
            metric = next((m for m in available_metrics if m in name.lower()), "euclidean")

            mats = {
                "A1": mat1,
                "A2": mat2,
                "a1": clouds[0],
                "a2": clouds[1],
                "a1_a2": np.vstack([clouds[0], clouds[1]]),
            }

            def get_dgms(k):
                if k == "a1":
                    cache, key = dgms_cache_i, (metric, i)
                elif k == "A1":
                    cache, key = dgms_cache_A1, (metric, i)
                elif k == "a2":
                    cache, key = dgms_cache_j, (metric, j)
                elif k == "A2":
                    cache, key = dgms_cache_A2, (metric, j)
                elif k == "a1_a2":
                    cache, key = dgms_cache_ij, (metric, i, j)
                else:
                    raise ValueError(f"clé inattendue : {k}")

                if key in cache:
                    return cache[key]["dgms"]
                t0 = time.perf_counter()
                dgms = ripser(mats[k], metric=metric)["dgms"]
                cache[key] = {"dgms": dgms, "time": time.perf_counter() - t0}
                return dgms


            dgms_src = get_dgms("a1")
            dgms_tgt = get_dgms("a2")

            if name.startswith("h0_count"):
                val = len(dgms_src[0])
            elif name.startswith("h1_count"):
                val = len(dgms_src[1])
            elif "h0_max_lifetime" in name:
                val = max((b[1]-b[0] for b in dgms_src[0] if b[1] != np.inf), default=0)
            elif "h1_max_lifetime" in name:
                val = max((b[1]-b[0] for b in dgms_src[1] if b[1] != np.inf), default=0)
            elif "entropy_H0" in name:
                val = persistent_entropy(dgms_src[0])[0]
            elif "entropy_H1" in name:
                val = persistent_entropy(dgms_src[1])[0]
            elif "bottleneck" in name:
                val = bottleneck_distance(dgms_src[0], dgms_tgt[0])
            elif "wasserstein" in name:
                val = wasserstein_distance(dgms_src[0], dgms_tgt[0])

            for cache, key in [
                (dgms_cache_i, (metric, i)),
                (dgms_cache_j, (metric, j)),
                (dgms_cache_ij, (metric, i, j)),
                (dgms_cache_A1, (metric, i)),
                (dgms_cache_A2, (metric, j)),
            ]:
                if key in cache:
                    diagram_time += cache[key]["time"]

        elif name.startswith("syn_"):
            val = compute_syntax_string_features(a1, a2, name)[name]

    except Exception as e:
        print(f"[!] {name} failed (i={i}, j={j}): {e}")
        val = np.nan

    total_time = time.perf_counter() - start_total
    return val, total_time, diagram_time



# ==========================================================
# === Pipeline principal CPU : calcul feature par feature
# ==========================================================
def compute_distances_feat(
    embedd_ds1: pd.DataFrame,
    embedd_ds2: pd.DataFrame,
    golden_matrix: pd.DataFrame,
    Category: str,
    Relation: str,
    Dataset: str,
    Model: str,
    flush_every: int = 50,
) -> pd.DataFrame:
    path_features=Root_project / "features_list.pkl"
    with open(path_features, "rb") as f:
        meta_features = pickle.load(f)

    results = []
    attrs1 = embedd_ds1.index.astype(str).tolist()
    attrs2 = embedd_ds2.index.astype(str).tolist()
    mat1 = embedd_ds1.to_numpy(float)
    mat2 = embedd_ds2.to_numpy(float)

    has_golden = isinstance(golden_matrix, pd.DataFrame) and not golden_matrix.empty
    if has_golden:
        golden_matrix.index = golden_matrix.index.astype(str)
        golden_matrix.columns = golden_matrix.columns.astype(str)

    dgms_cache_i, dgms_cache_j, dgms_cache_ij  ,dgms_cache_A1, dgms_cache_A2  = {}, {}, {} , {}, {}


    for i, a1 in enumerate(tqdm(attrs1, desc="🔁 SOURCE", leave=True)):
        v1 = mat1[i, :]
        for j, a2 in enumerate(tqdm(attrs2, desc=f"→ TARGET ({a1})", leave=False)):
            v2 = mat2[j, :]

            true_val = 0.0
            if has_golden:
                if a1 in golden_matrix.index and a2 in golden_matrix.columns:
                    true_val = float(golden_matrix.at[a1, a2])
                else:
                    true_val = float(golden_matrix.iloc[i, j])

            clouds = (vector_to_point_cloud(v1), vector_to_point_cloud(v2))

            for feat in meta_features:
                val, t_total, t_diag = compute_single_feature(
                    feat, v1, v2, mat1, mat2, a1, a2, i, j, clouds,
                    dgms_cache_A1=dgms_cache_A1,
                    dgms_cache_A2=dgms_cache_A2,
                    dgms_cache_i=dgms_cache_i,
                    dgms_cache_j=dgms_cache_j,
                    dgms_cache_ij=dgms_cache_ij
                )


                results.append({
                    "Attribute1": a1,
                    "Attribute2": a2,
                    "true_match": true_val,
                    "Feature": feat,
                    "Value": val,
                    "ExecTime_sec": t_total,
                    "DiagTime_sec": t_diag,
                    "Category": Category,
                    "Relation": Relation,
                    "Dataset": Dataset,
                    "Model": Model,
                })

    df_out = pd.DataFrame(results)

    return df_out
