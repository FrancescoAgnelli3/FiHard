from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple


ASSEMBLY_HAND_GROUPS: Tuple[dict, ...] = (
    {
        "wrist_index": 5,
        "nodes": tuple(range(0, 21)),
        "links": tuple(
            [
                (4, 19),
                (3, 16),
                (2, 13),
                (1, 10),
                (19, 18),
                (16, 15),
                (13, 12),
                (10, 9),
                (18, 17),
                (15, 14),
                (12, 11),
                (9, 8),
                (17, 5),
                (14, 5),
                (11, 5),
                (8, 5),
                (0, 7),
                (7, 6),
                (6, 5),
                (20, 5),
                (17, 14),
                (14, 11),
                (11, 8),
            ]
        ),
    },
    {
        "wrist_index": 26,
        "nodes": tuple(range(21, 42)),
        "links": tuple(
            [
                (25, 40),
                (24, 37),
                (23, 34),
                (22, 31),
                (40, 39),
                (37, 36),
                (34, 33),
                (31, 30),
                (39, 38),
                (36, 35),
                (33, 32),
                (30, 29),
                (38, 26),
                (35, 26),
                (32, 26),
                (29, 26),
                (21, 28),
                (28, 27),
                (27, 26),
                (41, 26),
                (38, 35),
                (35, 32),
                (32, 29),
            ]
        ),
    },
)

BIGHAND_HAND_GROUPS: Tuple[dict, ...] = (
    {
        "wrist_index": 0,
        "nodes": tuple(range(0, 21)),
        "links": tuple(
            [
                (0, 1),
                (1, 6),
                (6, 7),
                (7, 8),
                (0, 2),
                (2, 9),
                (9, 10),
                (10, 11),
                (0, 3),
                (3, 12),
                (12, 13),
                (13, 14),
                (0, 4),
                (4, 15),
                (15, 16),
                (16, 17),
                (0, 5),
                (5, 18),
                (18, 19),
                (19, 20),
            ]
        ),
    },
)

FPHA_HAND_GROUPS: Tuple[dict, ...] = BIGHAND_HAND_GROUPS

H2O_HAND_GROUPS: Tuple[dict, ...] = (
    {
        "wrist_index": 0,
        "nodes": tuple(range(0, 21)),
        "links": tuple(
            [
                (0, 1),  (1, 2),  (2, 3),  (3, 4),
                (0, 5),  (5, 6),  (6, 7),  (7, 8),
                (0, 9),  (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
        ),
    },
    {
        "wrist_index": 21,
        "nodes": tuple(range(21, 42)),
        "links": tuple(
            [
                (21, 22), (22, 23), (23, 24), (24, 25),
                (21, 26), (26, 27), (27, 28), (28, 29),
                (21, 30), (30, 31), (31, 32), (32, 33),
                (21, 34), (34, 35), (35, 36), (36, 37),
                (21, 38), (38, 39), (39, 40), (40, 41)
            ]
        ),
    },
)


DATASET_GRAPH_METADATA: Dict[str, Dict[str, object]] = {
    "assembly": {
        "hand_groups": ASSEMBLY_HAND_GROUPS,
        "default_wrist_indices": (5, 26),
        "node_count": 21,
    },
    "h2o": {
        "hand_groups": H2O_HAND_GROUPS,
        "default_wrist_indices": (0, 21),
        "node_count": 21,
    },
    "bighands": {
        "hand_groups": BIGHAND_HAND_GROUPS,
        "default_wrist_indices": (0,),
        "node_count": 21,
    },
    "fpha": {
        "hand_groups": FPHA_HAND_GROUPS,
        "default_wrist_indices": (0,),
        "node_count": 21,
    },
}


def get_dataset_graph_metadata(name: str) -> Dict[str, object]:
    key = str(name).lower()
    if key not in DATASET_GRAPH_METADATA:
        raise ValueError(f"Unknown dataset graph '{name}'. Expected one of: {', '.join(DATASET_GRAPH_METADATA)}")
    return DATASET_GRAPH_METADATA[key]


def resolve_hand_groups(
    dataset: str,
    wrist_indices: Tuple[int, ...] = tuple(),
) -> Tuple[dict, ...]:
    meta = get_dataset_graph_metadata(dataset)
    base_groups = tuple(meta.get("hand_groups", ()))
    if not wrist_indices:
        return base_groups
    if len(wrist_indices) != len(base_groups):
        raise ValueError(
            f"Number of wrist indices ({len(wrist_indices)}) must match number of hand groups ({len(base_groups)})."
        )
    return tuple(
        {**group, "wrist_index": int(wrist_idx)}
        for group, wrist_idx in zip(base_groups, wrist_indices)
    )


def resolve_local_hand_graph_metadata(
    dataset: str,
    wrist_indices: Tuple[int, ...] = tuple(),
) -> Dict[str, object]:
    variants: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
    for group in resolve_hand_groups(dataset, wrist_indices):
        nodes_global = [int(idx) for idx in group.get("nodes", ())]
        if not nodes_global:
            continue
        g2l = {g: li for li, g in enumerate(nodes_global)}
        wrist_global = int(group.get("wrist_index", -1))
        if wrist_global not in g2l:
            raise ValueError(f"Wrist index {wrist_global} is not part of group nodes for dataset '{dataset}'.")
        wrist_local = int(g2l[wrist_global])

        local_edges = set()
        for pair in group.get("links", ()):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a_global = int(pair[0])
            b_global = int(pair[1])
            if a_global not in g2l or b_global not in g2l:
                continue
            a_local = int(g2l[a_global])
            b_local = int(g2l[b_global])
            if a_local == b_local:
                continue
            if a_local > b_local:
                a_local, b_local = b_local, a_local
            local_edges.add((a_local, b_local))
        variants.append((wrist_local, tuple(sorted(local_edges))))

    if not variants:
        raise ValueError(f"No usable hand graph metadata for dataset '{dataset}'.")

    wrist_local_ref, links_ref = variants[0]
    for wrist_local, links in variants[1:]:
        if wrist_local != wrist_local_ref or links != links_ref:
            raise ValueError(
                "Hand groups resolve to different local topologies/wrist indices; "
                "shared graph helpers expect a single shared local hand graph."
            )

    return {
        "wrist_index": int(wrist_local_ref),
        "links": tuple((int(i), int(j)) for i, j in links_ref),
    }


def build_bfs_parents_from_links(num_nodes: int, links: Iterable[Tuple[int, int]], root: int = 0) -> List[int]:
    adj = {i: [] for i in range(int(num_nodes))}
    for a, b in links:
        a = int(a)
        b = int(b)
        if 0 <= a < num_nodes and 0 <= b < num_nodes and a != b:
            adj[a].append(b)
            adj[b].append(a)
    for i in range(int(num_nodes)):
        adj[i] = sorted(set(adj[i]))

    root = int(root)
    if not (0 <= root < num_nodes):
        root = 0

    parents = [-2] * int(num_nodes)
    parents[root] = -1
    queue: deque[int] = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if parents[v] == -2:
                parents[v] = u
                queue.append(v)

    for i in range(int(num_nodes)):
        if parents[i] == -2:
            parents[i] = root if i != root else -1
    return parents


def get_root_first_single_hand_graph(
    dataset: str,
    wrist_indices: Tuple[int, ...] = tuple(),
) -> Dict[str, object]:
    local_graph = resolve_local_hand_graph_metadata(dataset, wrist_indices)
    wrist_index = int(local_graph["wrist_index"])
    links = tuple(local_graph["links"])
    node_count = int(get_dataset_graph_metadata(dataset).get("node_count", 21))

    node_order = [wrist_index] + [idx for idx in range(node_count) if idx != wrist_index]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_order)}
    root_first_links = tuple(
        sorted(
            (
                (old_to_new[int(a)], old_to_new[int(b)])
                for a, b in links
                if int(a) in old_to_new and int(b) in old_to_new and int(a) != int(b)
            )
        )
    )
    parents = build_bfs_parents_from_links(node_count, root_first_links, root=0)
    return {
        "node_order": tuple(int(idx) for idx in node_order),
        "links": tuple((int(a), int(b)) for a, b in root_first_links),
        "parents": tuple(int(p) for p in parents),
        "original_wrist_index": wrist_index,
    }
