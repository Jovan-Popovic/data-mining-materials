from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


# ───────────────────────── core ECLAT ──────────────────────────
def _eclat(
    prefix: Tuple[str, ...],
    items: List[Tuple[str, np.ndarray]],
    minsup: int,
    out: List[Tuple[Tuple[str, ...], int]],
) -> None:
    while items:
        item, tid = items.pop()
        new_prefix = prefix + (item,)
        supp = len(tid)
        if supp >= minsup:
            out.append((new_prefix, supp))
            # extend
            suffix = []
            for item2, tid2 in items:
                inter = np.intersect1d(tid, tid2, assume_unique=True)
                if len(inter) >= minsup:
                    suffix.append((item2, inter))
            _eclat(new_prefix, suffix, minsup, out)


def eclat(
    transactions: List[List[str]], minsup: int
) -> List[Tuple[Tuple[str, ...], int]]:
    # build vertical TID-lists
    tid_lists: Dict[str, List[int]] = {}
    for tid, trx in enumerate(transactions):
        for item in trx:
            tid_lists.setdefault(item, []).append(tid)
    items = [(i, np.array(tid)) for i, tid in tid_lists.items()]
    # sort by support ascending to prune brže
    items.sort(key=lambda x: len(x[1]))
    output: List[Tuple[Tuple[str, ...], int]] = []
    _eclat(tuple(), items, minsup, output)
    # sortiraj po (dužina, -support)
    return sorted(output, key=lambda t: (len(t[0]), -t[1]))


# ───────────────────── asocijativna pravila ─────────────────────
from itertools import combinations


def gen_rules(freq_itemsets, minconf: float):
    # ❶ mapa SVEH čestih (sortirani tuple → sup)
    supp_map = {tuple(sorted(fs)): sup for fs, sup in freq_itemsets}
    rules = []
    for itemset, sup in freq_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            from itertools import combinations

            for ante in combinations(itemset, i):
                ante = tuple(sorted(ante))
                cons = tuple(sorted(set(itemset) - set(ante)))
                ante_sup = supp_map.get(ante)  # ← ❷ safe lookup
                if ante_sup is None:
                    continue  # preskoči ne-čest antecedent
                conf = sup / ante_sup
                if conf >= minconf:
                    rules.append((ante, cons, sup, conf))
    return sorted(rules, key=lambda r: -r[3])
