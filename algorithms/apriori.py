from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Set, Tuple


# ------------------------------------------------------------------------
# 1. Frequent-itemset mining
# ------------------------------------------------------------------------
def _count_support(
    candidates: Iterable[frozenset], transactions: List[Set[str]]
) -> Dict[frozenset, int]:
    """Return support counts for each candidate."""
    counts = defaultdict(int)
    for T in transactions:
        for cand in candidates:
            if cand <= T:  # subset test
                counts[cand] += 1
    return counts


def _join_step(prev_freq: List[frozenset], length: int) -> List[frozenset]:
    """Self-join L(k-1) to get C(k) and prune duplicates."""
    joined = set()
    for i in range(len(prev_freq)):
        for j in range(i + 1, len(prev_freq)):
            a, b = prev_freq[i], prev_freq[j]
            union = a | b
            if len(union) == length:
                joined.add(union)
    return sorted(joined)


def apriori_frequent_itemsets(
    transactions: List[List[str]], min_support: int, verbose: bool = False
) -> List[Tuple[frozenset, int]]:
    """
    Classic Apriori: returns list of (itemset, support) sorted by length then -support.
    • transactions – list of lists (strings or hashables)
    • min_support  – absolute threshold (e.g. 3 means “appears in ≥3 trx”)
    """
    # 1. Make transactions sets for fast subset checks
    trx_sets = [set(t) for t in transactions]
    n_trx = len(trx_sets)

    # 2. L1
    item_counts = _count_support(
        [frozenset([i]) for T in trx_sets for i in T], trx_sets
    )
    L1 = [frozenset([i]) for i, c in item_counts.items() if c >= min_support]
    freq_itemsets: List[Tuple[frozenset, int]] = [(i, item_counts[i]) for i in L1]

    k = 2
    L_prev = L1
    while L_prev:
        Ck = _join_step(L_prev, k)
        Ck_counts = _count_support(Ck, trx_sets)
        Lk = [i for i in Ck if Ck_counts[i] >= min_support]

        if verbose:
            print(f"k={k}  candidates={len(Ck)}  freq={len(Lk)}")

        freq_itemsets.extend((i, Ck_counts[i]) for i in Lk)
        L_prev = Lk
        k += 1

    # sort nicely
    freq_itemsets.sort(key=lambda x: (len(x[0]), -x[1], tuple(sorted(x[0]))))
    return freq_itemsets


# ------------------------------------------------------------------------
# 2. Association-rule generation
# ------------------------------------------------------------------------
def generate_rules(
    freq_itemsets: List[Tuple[frozenset, int]],
    min_conf: float,
    n_transactions: int | None = None,
    metrics: Tuple[str, ...] = ("support", "confidence", "lift"),
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with keys: antecedent, consequent, support, confidence, lift.
    min_conf – e.g. 0.4  → keep rules with conf ≥ 40 %
    n_transactions – if given, support is returned as fraction; else absolute count.
    """
    # map for quick support look-up
    supp_map = {fs: sup for fs, sup in freq_itemsets}

    rules = []
    for itemset, sup_itemset in freq_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                conf = sup_itemset / supp_map[antecedent]
                if conf >= min_conf:
                    rule = {"antecedent": antecedent, "consequent": consequent}
                    if "support" in metrics:
                        rule["support"] = (
                            sup_itemset / n_transactions
                            if n_transactions
                            else sup_itemset
                        )
                    if "confidence" in metrics:
                        rule["confidence"] = conf
                    if "lift" in metrics:
                        rule["lift"] = conf / (
                            supp_map[consequent] / (n_transactions or 1)
                        )
                    rules.append(rule)
    # sort by descending confidence then lift
    rules.sort(key=lambda r: (-r["confidence"], -r.get("lift", 0)))
    return rules


# ------------------------------------------------------------------------
# 3. Optional: quick bar-chart helper (matplotlib)
# ------------------------------------------------------------------------
def plot_itemset_supports(freq_itemsets: List[Tuple[frozenset, int]], top_n: int = 10):
    import matplotlib.pyplot as _plt

    top = sorted(freq_itemsets, key=lambda x: -x[1])[:top_n]
    labels = [
        "{" + ",".join(sorted(map(str, fs))) + "}"  # svaki element → str
        for fs, _ in top
    ]
    values = [sup for _, sup in top]
    _plt.figure(figsize=(max(6, len(top)), 3))
    _plt.bar(range(len(top)), values)
    _plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    _plt.ylabel("support (count)")
    _plt.title(f"Top {top_n} frequent itemsets")
    _plt.tight_layout()
    _plt.show()


# ------------------------------------------------------------------------
# 4. Self-test demo
# ------------------------------------------------------------------------
if __name__ == "__main__":
    trx = [
        ["milk", "bread", "butter"],
        ["beer", "bread"],
        ["milk", "bread", "butter", "beer"],
        ["bread", "butter"],
        ["milk", "bread"],
    ]
    L = apriori_frequent_itemsets(trx, min_support=2, verbose=True)
    print("\nFrequent sets:", L)
    rules = generate_rules(L, min_conf=0.6, n_transactions=len(trx))
    print("\nRules:")
    for r in rules:
        print(r)
