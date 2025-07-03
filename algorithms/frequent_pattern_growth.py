from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


# ------------------------------------------------------------------
# 1. FP-Tree struktura
# ------------------------------------------------------------------
class FPNode:
    def __init__(self, item, parent):
        self.item = item  # etiketa (None za korijen)
        self.count = 1
        self.parent = parent  # FPNode ili None
        self.children: Dict[str, "FPNode"] = {}
        self.link = None  # horizontalni link na sljedeći čvor iste stavke

    def incr(self, n=1):
        self.count += n


def build_fp_tree(
    transactions: List[List[str]], min_support: int
) -> Tuple[FPNode, Dict[str, FPNode]]:
    """
    Građenje FP-stabla.
    Vraća (korijen, header_table) gdje je header_table: {item -> prvi čvor u lancu}.
    """
    # 1. Count global frequencies
    freq = Counter(item for T in transactions for item in T)
    freq = {item: c for item, c in freq.items() if c >= min_support}
    if not freq:
        return None, None

    # 2. Redoslijed: sortiraj stavke po učestalosti ↓
    order = sorted(freq, key=lambda x: (-freq[x], x))
    order_index = {item: i for i, item in enumerate(order)}

    # 3. Build tree
    root = FPNode(None, None)
    header: Dict[str, FPNode] = {}

    for T in transactions:
        # filtriraj & sort po globalnom order-u
        sorted_items = [
            i for i in sorted(T, key=lambda x: order_index.get(x, 1e9)) if i in freq
        ]
        curr = root
        for item in sorted_items:
            if item in curr.children:
                curr.children[item].incr()
            else:
                child = FPNode(item, curr)
                curr.children[item] = child
                # horizontal link
                if item in header:
                    last = header[item]
                    while last.link:
                        last = last.link
                    last.link = child
                else:
                    header[item] = child
            curr = curr.children[item]
    return root, header


# ------------------------------------------------------------------
# 2. Rudarenje iz FP-stabla
# ------------------------------------------------------------------
def ascend(node: FPNode) -> List[str]:
    path = []
    while node.parent and node.parent.item is not None:
        node = node.parent
        path.append(node.item)
    return path


def conditional_pattern_base(
    item: str, header: Dict[str, FPNode]
) -> List[Tuple[List[str], int]]:
    """Lista (putanja, count) za dato 'item'."""
    patterns = []
    node = header[item]
    while node:
        path = ascend(node)
        if path:
            patterns.append((path, node.count))
        node = node.link
    return patterns


def mine_fp_tree(
    header: Dict[str, FPNode],
    min_support: int,
    suffix: Tuple[str] = (),
    freq_itemsets: List[Tuple[Tuple[str], int]] = None,
):
    if freq_itemsets is None:
        freq_itemsets = []
    items = sorted(header, key=lambda x: x)  # leksikografski
    for item in items:
        new_suffix = (item,) + suffix
        # support stavke = suma counts u lancu
        support = 0
        node = header[item]
        while node:
            support += node.count
            node = node.link
        freq_itemsets.append((new_suffix[::-1], support))

        # conditional base
        patterns = conditional_pattern_base(item, header)
        # expand each pattern by count
        cond_transactions = []
        for path, cnt in patterns:
            cond_transactions.extend([path] * cnt)
        # build conditional tree
        cond_root, cond_header = build_fp_tree(cond_transactions, min_support)
        if cond_header:
            mine_fp_tree(cond_header, min_support, new_suffix, freq_itemsets)
    return freq_itemsets


# ------------------------------------------------------------------
# 3. Pravila asocijacije
# ------------------------------------------------------------------
def generate_rules(
    freq_itemsets: List[Tuple[Tuple[str], int]], min_conf: float, n_transactions: int
) -> List[Tuple[Tuple[str], Tuple[str], float, float]]:
    """
    Vraća listu (antecedent, consequent, support, confidence)
    """
    # map itemset -> support
    supp_map = {" ".join(sorted(iset)): s for iset, s in freq_itemsets}
    rules = []
    for iset, sup in freq_itemsets:
        if len(iset) < 2:
            continue
        iset_key = " ".join(sorted(iset))
        for i in range(1, len(iset)):
            # sve moguće podpodskupove veličine i
            from itertools import combinations

            for antecedent in combinations(iset, i):
                consequent = tuple(sorted(set(iset) - set(antecedent)))
                ant_key = " ".join(sorted(antecedent))
                conf = sup / supp_map[ant_key]
                if conf >= min_conf:
                    rules.append((antecedent, consequent, sup, conf))
    return rules


def draw_fp_tree(
    root: FPNode,
    node_size: int = 500,
    font_size: int = 9,
    y_step: float = 1.5,
    figsize: tuple = (6, 4),
):
    """
    Crta FP-stablo samo pomoću matplotlib-a.
    • node_size – veličina kružića
    • font_size – veličina teksta pored čvora
    • y_step    – vertikalni razmak između nivoa
    """

    # --- 1) prvo izračunamo širinu (broj listova) svakog podstabla ----------
    subtree_width: Dict[int, int] = {}

    def calc_width(node: FPNode) -> int:
        if not node.children:
            subtree_width[id(node)] = 1
            return 1
        w = sum(calc_width(ch) for ch in node.children.values())
        subtree_width[id(node)] = w
        return w

    calc_width(root)

    # --- 2) potom dodjeljujemo koordinate ---------------------------
    pos: Dict[int, tuple] = {}  # id(node) -> (x, y)
    id2node: Dict[int, FPNode] = {}  # za kasnije anotacije
    x_cursor = 0  # globalni “pen” za X koordinatu

    def assign_pos(node: FPNode, depth: int):
        nonlocal x_cursor
        id2node[id(node)] = node
        if not node.children:
            pos[id(node)] = (x_cursor, -depth * y_step)
            x_cursor += 1
        else:
            # prvo pozicioniraj djecu (post-order)
            child_x = []
            for ch in node.children.values():
                assign_pos(ch, depth + 1)
                child_x.append(pos[id(ch)][0])
            # roditelj ide na sredinu raspona djece
            cx = (child_x[0] + child_x[-1]) / 2
            pos[id(node)] = (cx, -depth * y_step)

    assign_pos(root, depth=0)

    # --- 3) crtanje --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    # ivice
    for parent_id, pnode in id2node.items():
        for child in pnode.children.values():
            x1, y1 = pos[parent_id]
            x2, y2 = pos[id(child)]
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=0.8)

    # čvorovi
    for nid, (x, y) in pos.items():
        node = id2node[nid]
        ax.scatter(
            x,
            y,
            s=node_size,
            color="tab:blue" if node.item else "tab:gray",
            edgecolor="k",
            zorder=3,
        )
        if node.item is not None:  # korijen nema etiketu
            ax.text(
                x + 0.05,
                y + 0.05,
                f"{node.item}:{node.count}",
                fontsize=font_size,
                va="bottom",
                ha="left",
            )

    ax.axis("off")
    plt.tight_layout()
    plt.show()
