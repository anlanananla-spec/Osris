# Osris Topology Graph Dataset

<details>
##Python
```
<details>
import argparse
import os
import re
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData


def normalize_name(name: str) -> str:
    if not name:
        return ""
    return name.upper().replace("\\", "").replace("%", "").strip()


def parse_val(val_str: str) -> float:
    if not val_str:
        return 0.0
    val_str = val_str.lower().strip()
    mults = {
        "t": 1e12,
        "g": 1e9,
        "meg": 1e6,
        "x": 1e6,
        "k": 1e3,
        "m": 1e-3,
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "f": 1e-15,
        "a": 1e-18,
    }
    try:
        return float(val_str)
    except ValueError:
        pass

    match = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-z]*)$", val_str)
    if not match:
        return 0.0

    number, suffix = match.groups()
    try:
        value = float(number)
    except ValueError:
        return 0.0

    if suffix:
        if suffix.startswith("meg"):
            return value * 1e6
        if suffix[0] in mults:
            return value * mults[suffix[0]]
    return value


def read_file_tokens(filepath: str):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    lines = []
    buffer = ""
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", "//", "$")):
            continue
        if stripped.startswith("+"):
            buffer += " " + stripped[1:].strip()
        elif buffer.endswith("\\"):
            buffer = buffer[:-1].strip() + " " + stripped
        else:
            if buffer:
                lines.append(buffer)
            buffer = stripped
    if buffer:
        lines.append(buffer)

    tokenized = []
    for line in lines:
        cleaned = line.replace("(", " ").replace(")", " ")
        tokens = cleaned.split()
        if tokens:
            tokenized.append(tokens)
    return tokenized


def get_param(tokens, key: str, default: float = 0.0) -> float:
    prefix = key.lower() + "="
    for token in tokens:
        if token.lower().startswith(prefix):
            return parse_val(token.split("=", 1)[1])
    return default


def parse_spice_subckt(sp_path: str, target_subckt: str | None = None):
    lines = read_file_tokens(sp_path)

    subckts = []
    current = None
    for tokens in lines:
        head = tokens[0].upper()
        if head in [".SUBCKT", "SUBCKT"]:
            if len(tokens) < 2:
                continue
            current = {
                "name": normalize_name(tokens[1]),
                "ports": [normalize_name(t) for t in tokens[2:] if "=" not in t],
                "lines": [],
            }
        elif head in [".ENDS", "ENDS"]:
            if current is not None:
                subckts.append(current)
                current = None
        elif current is not None:
            current["lines"].append(tokens)

    if not subckts:
        raise ValueError(f"No .subckt found in {sp_path}")

    if target_subckt is None:
        return subckts[0]

    normalized_target = normalize_name(target_subckt)
    for subckt in subckts:
        if subckt["name"] == normalized_target:
            return subckt
    raise ValueError(f"Subckt '{target_subckt}' not found in {sp_path}")


def build_topology_graph(sp_path: str, target_subckt: str | None = None):
    subckt = parse_spice_subckt(sp_path, target_subckt)

    device_rows = []
    device_types = []
    device_names = []
    device_connections = []
    node_to_devices = defaultdict(list)
    all_nodes = set(subckt["ports"])
    internal_first_seen = []
    internal_seen = set()

    for tokens in subckt["lines"]:
        name = normalize_name(tokens[0])
        if not name:
            continue

        if name.startswith("M") and len(tokens) >= 6:
            nodes = [normalize_name(token) for token in tokens[1:5]]
            model_name = normalize_name(tokens[5])
            is_pmos = 1.0 if "PFET" in model_name or "__P" in model_name else 0.0
            features = [
                1.0,  # is_mos (0 means cap)
                is_pmos,  # for MOS: 1=PMOS, 0=NMOS
                get_param(tokens, "w"),
                get_param(tokens, "l"),
                get_param(tokens, "nf", 1.0) or 1.0,
            ]
            device_rows.append(features)
            device_types.append("mos")
            device_names.append(name)
            device_connections.append(nodes)
            for pin_idx, node in enumerate(nodes):
                all_nodes.add(node)
                node_to_devices[node].append((len(device_rows) - 1, pin_idx))
                if node not in subckt["ports"] and node not in internal_seen:
                    internal_seen.add(node)
                    internal_first_seen.append(node)
        elif (name.startswith("C") or name.startswith("MIM")) and len(tokens) >= 4:
            nodes = [normalize_name(tokens[1]), normalize_name(tokens[2])]
            features = [
                0.0,  # is_mos (0 means cap)
                0.0,  # not applicable for cap
                get_param(tokens, "w"),
                get_param(tokens, "l"),
                0.0,  # nf is not applicable for capacitor
            ]
            device_rows.append(features)
            device_types.append("cap")
            device_names.append(name)
            device_connections.append(nodes)
            for pin_idx, node in enumerate(nodes):
                all_nodes.add(node)
                node_to_devices[node].append((len(device_rows) - 1, pin_idx))
                if node not in subckt["ports"] and node not in internal_seen:
                    internal_seen.add(node)
                    internal_first_seen.append(node)
        else:
            # Frontend topology currently models only MOS and CAP devices.
            continue

    preferred_port_order = ["INM", "INP", "OUT", "VDD", "GND"]
    port_set = set(subckt["ports"])
    ordered_ports = [p for p in preferred_port_order if p in port_set]
    ordered_ports.extend([p for p in subckt["ports"] if p not in set(ordered_ports)])

    internal_nodes = [n for n in all_nodes if n not in port_set]
    net_style_nodes = []
    other_internal_nodes = []
    for node in internal_nodes:
        match = re.match(r"^NET(\d+)$", node)
        if match:
            net_style_nodes.append((int(match.group(1)), node))
        else:
            other_internal_nodes.append(node)
    net_style_nodes.sort(key=lambda x: x[0])
    net_style_nodes = [name for _, name in net_style_nodes]

    # Keep non-NET internal nodes deterministic by first appearance.
    first_seen_rank = {n: i for i, n in enumerate(internal_first_seen)}
    other_internal_nodes.sort(key=lambda n: first_seen_rank.get(n, 10**9))

    ordered_nodes = ordered_ports + net_style_nodes + other_internal_nodes
    node_index = {node: idx for idx, node in enumerate(ordered_nodes)}

    node_rows = []
    for node in ordered_nodes:
        is_external_port = 1.0 if node in port_set else 0.0

        # Node feature (8-d):
        # [is_external_port, mos_count, src_count, drain_count, gate_count, bulk_count, total_dev_w, total_dev_l]
        connected = node_to_devices.get(node, [])
        mos_dev_set = set()
        dev_set = set()
        src_count = 0.0
        drain_count = 0.0
        gate_count = 0.0
        bulk_count = 0.0

        for dev_idx, pin_idx in connected:
            dev_set.add(dev_idx)
            if device_types[dev_idx] == "mos":
                mos_dev_set.add(dev_idx)
                if pin_idx == 0:
                    drain_count += 1.0
                elif pin_idx == 1:
                    gate_count += 1.0
                elif pin_idx == 2:
                    src_count += 1.0
                elif pin_idx == 3:
                    bulk_count += 1.0

        total_dev_w = 0.0
        total_dev_l = 0.0
        for dev_idx in dev_set:
            total_dev_w += float(device_rows[dev_idx][2])
            total_dev_l += float(device_rows[dev_idx][3])

        node_rows.append([
            is_external_port,
            float(len(mos_dev_set)),
            src_count,
            drain_count,
            gate_count,
            bulk_count,
            total_dev_w,
            total_dev_l,
        ])

    dev_to_node_src = []
    dev_to_node_dst = []
    dev_to_node_attr = []
    for dev_idx, nodes in enumerate(device_connections):
        d_type = device_types[dev_idx]
        for pin_idx, node in enumerate(nodes):
            dev_to_node_src.append(dev_idx)
            dev_to_node_dst.append(node_index[node])
            if d_type == "mos":
                # 5-d edge feature: [D, G, S, B, is_cap_pin]
                edge_feature = [0.0, 0.0, 0.0, 0.0, 0.0]
                if 0 <= pin_idx <= 3:
                    edge_feature[pin_idx] = 1.0
            else:
                edge_feature = [0.0, 0.0, 0.0, 0.0, 1.0]
            dev_to_node_attr.append(edge_feature)

    # Node-node prediction edges (unordered pairs, i < j).
    pred_src = []
    pred_dst = []
    pred_pair_names = []
    num_nodes = len(ordered_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pred_src.append(i)
            pred_dst.append(j)
            pred_pair_names.append((ordered_nodes[i], ordered_nodes[j]))

    data = HeteroData()
    data["device"].x = torch.tensor(device_rows, dtype=torch.float) if device_rows else torch.empty((0, 5), dtype=torch.float)
    data["node"].x = torch.tensor(node_rows, dtype=torch.float) if node_rows else torch.empty((0, 8), dtype=torch.float)

    if dev_to_node_src:
        data["device", "connects", "node"].edge_index = torch.tensor([dev_to_node_src, dev_to_node_dst], dtype=torch.long)
        data["device", "connects", "node"].edge_attr = torch.tensor(dev_to_node_attr, dtype=torch.float)
    else:
        data["device", "connects", "node"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["device", "connects", "node"].edge_attr = torch.empty((0, 5), dtype=torch.float)

    if pred_src:
        data["node", "predicts_cap_to", "node"].edge_index = torch.tensor([pred_src, pred_dst], dtype=torch.long)
        data["node", "predicts_cap_to", "node"].edge_label = torch.full((len(pred_src), 4), float("nan"), dtype=torch.float)
    else:
        data["node", "predicts_cap_to", "node"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["node", "predicts_cap_to", "node"].edge_label = torch.empty((0, 4), dtype=torch.float)

    data.subckt_name = subckt["name"]
    data.port_names = ordered_ports
    data.device_names = device_names
    data.node_names = ordered_nodes
    data.node_encoding = {name: idx for idx, name in enumerate(ordered_nodes)}
    data.node_pair_names = pred_pair_names
    data.node_pair_label_desc = ["mean_F", "var_F2", "max_F", "min_F"]
    # Compatibility alias for downstream scripts that still use net naming.
    data.net_names = ordered_nodes
    return data


def main():
    parser = argparse.ArgumentParser(description="Convert a frontend SPICE subckt topology into a graph.")
    parser.add_argument("--sp", default=os.path.join("work", "ahuja_ota_4.sp"), help="Path to the frontend SPICE file.")
    parser.add_argument("--subckt", default=None, help="Target subckt name. Defaults to the first subckt in the file.")
    parser.add_argument(
        "--out",
        default=os.path.join("work", "ahuja_ota_4_topology.pt"),
        help="Output path for the serialized PyG HeteroData.",
    )
    args = parser.parse_args()

    graph = build_topology_graph(args.sp, args.subckt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(graph, args.out)

    print(f"Saved graph to: {args.out}")
    print(f"Subckt: {graph.subckt_name}")
    print(f"Ports ({len(graph.port_names)}): {graph.port_names}")
    print(f"Devices: {graph['device'].x.shape[0]}")
    print(f"Nodes: {graph['node'].x.shape[0]}")
    print(f"Device-Node edges: {graph['device', 'connects', 'node'].edge_index.shape[1]}")
    print(f"Node-Node prediction edges: {graph['node', 'predicts_cap_to', 'node'].edge_index.shape[1]}")


if __name__ == "__main__":
    main()

</details>
```


## How To Run

Generate the frontend topology graph:

```bash
python work/sp_topology_to_graph.py --sp work/ahuja_ota_4.sp --out work/ahuja_ota_4_topology.pt
```

Aggregate PEX capacitance labels and write the final labeled graph:

```bash
python pex_port_cap_stats_to_frontend.py --frontend-graph work/ahuja_ota_4_topology.pt --out-graph work/ahuja_ota_4_topology_labeled.pt
```

## Graph Overview

The output graph is a PyTorch Geometric `HeteroData` object.

| Graph Item | Shape | Meaning |
| --- | ---: | --- |
| `data["device"].x` | `(15, 5)` | Device node features. |
| `data["node"].x` | `(13, 8)` | Circuit node features. |
| `data["device", "connects", "node"].edge_index` | `(2, 58)` | Device-to-node topology edges. |
| `data["device", "connects", "node"].edge_attr` | `(58, 5)` | Pin-role feature for each device-node edge. |
| `data["node", "predicts_cap_to", "node"].edge_index` | `(2, 78)` | All unordered frontend node pairs. |
| `data["node", "predicts_cap_to", "node"].edge_label` | `(78, 4)` | Node-node capacitance labels. |
| `data["node"].ground_cap_label` | `(13, 4)` | Node-to-ground capacitance labels. |



## Device Nodes

Devices are modeled as `device` nodes. The frontend graph currently includes
MOS devices and capacitors.

`data["device"].x` has 5 dimensions:

| Index | Feature | Meaning |
| ---: | --- | --- |
| 0 | `is_mos` | `1` for MOS, `0` for capacitor. |
| 1 | `is_pmos` | `1` for PMOS, `0` for NMOS. For capacitors, this is `0`. |
| 2 | `w` | Device width parsed from the frontend SPICE netlist. |
| 3 | `l` | Device length parsed from the frontend SPICE netlist. |
| 4 | `nf` | Number of fingers for MOS. For capacitors, this is `0` because `nf` is not applicable. |

Example:

| Device | Feature Vector |
| --- | --- |
| `M7` | `[1, 0, w, l, nf]` |
| `M8` | `[1, 1, w, l, nf]` |
| `C0` | `[0, 0, w, l, 0]` |

## Circuit Nodes

Frontend nets are modeled as `node` nodes.

`data["node"].x` has 8 dimensions:

| Index | Feature | Meaning |
| ---: | --- | --- |
| 0 | `is_port` | `1` if this node is an external subckt port, otherwise `0`. |
| 1 | `mos_cnt` | Number of unique MOS devices connected to this node. |
| 2 | `src_cnt` | Number of MOS source pins connected to this node. |
| 3 | `drain_cnt` | Number of MOS drain pins connected to this node. |
| 4 | `gate_cnt` | Number of MOS gate pins connected to this node. |
| 5 | `bulk_cnt` | Number of MOS bulk/substrate pins connected to this node. |
| 6 | `total_w` | Sum of `w` over unique devices connected to this node. Includes MOS and capacitors. |
| 7 | `total_l` | Sum of `l` over unique devices connected to this node. Includes MOS and capacitors. |

Notes:

| Point | Explanation |
| --- | --- |
| `mos_cnt` is local to each node | It counts how many unique frontend MOS devices touch that node. Summing `mos_cnt` over all nodes is not equal to the global MOS count. |
| Pin counts are role-specific | A diode-connected MOS can contribute to multiple pin-role counts on the same node. |
| `total_w` and `total_l` include capacitors | Capacitors are included because they are frontend devices connected to the node. |

## Device-Node Edges

Each device terminal connection becomes one `device -> node` edge.

`data["device", "connects", "node"].edge_attr` has 5 dimensions:

| Index | Feature | Meaning |
| ---: | --- | --- |
| 0 | `is_D` | This edge is a MOS drain connection. |
| 1 | `is_G` | This edge is a MOS gate connection. |
| 2 | `is_S` | This edge is a MOS source connection. |
| 3 | `is_B` | This edge is a MOS bulk/substrate connection. |
| 4 | `is_cap_pin` | This edge belongs to a capacitor terminal. |

Examples:

| Edge Type | Edge Feature |
| --- | --- |
| MOS drain edge | `[1, 0, 0, 0, 0]` |
| MOS gate edge | `[0, 1, 0, 0, 0]` |
| MOS source edge | `[0, 0, 1, 0, 0]` |
| MOS bulk edge | `[0, 0, 0, 1, 0]` |
| Capacitor terminal edge | `[0, 0, 0, 0, 1]` |

## Node-Node Labels

All unordered frontend node pairs are represented as prediction edges:

```python
data["node", "predicts_cap_to", "node"].edge_index
data["node", "predicts_cap_to", "node"].edge_label
```

There are 13 frontend nodes, so there are:

```text
13 * 12 / 2 = 78 node-node prediction edges
```

`edge_label` has 4 dimensions:

| Index | Label | Meaning |
| ---: | --- | --- |
| 0 | `mean_F` | Mean capacitance across 101 PEX variants. |
| 1 | `var_F2` | Variance across 101 PEX variants. |
| 2 | `max_F` | Maximum capacitance across 101 PEX variants. |
| 3 | `min_F` | Minimum capacitance across 101 PEX variants. |

These labels are extracted from mapped PEX capacitance lines:

```spice
Cxxx node_a node_b value
```

PEX hash-coordinate nodes are first mapped back to frontend nodes. Then
capacitances that land on the same frontend node pair are summed within each
variant. The final 4-dimensional label is computed across all 101 variants.

Coverage:

| Label Set | Coverage |
| --- | ---: |
| Node-node capacitance labels | `78 / 78` |

## Node-To-Ground Labels

PEX lines of the form:

```spice
Cxxx node 0 value
```

are treated as node-to-ground parasitic capacitance. These are stored on the
corresponding node:

```python
data["node"].ground_cap_label
```

`ground_cap_label` has 4 dimensions:

| Index | Label | Meaning |
| ---: | --- | --- |
| 0 | `mean_F` | Mean node-to-ground capacitance across 101 PEX variants. |
| 1 | `var_F2` | Variance across 101 PEX variants. |
| 2 | `max_F` | Maximum node-to-ground capacitance across 101 PEX variants. |
| 3 | `min_F` | Minimum node-to-ground capacitance across 101 PEX variants. |

Coverage:

| Label Set | Coverage |
| --- | ---: |
| Node-to-ground capacitance labels | `13 / 13` |

## PEX Hash Node Mapping

PEX netlists use layout-coordinate names such as:

```text
M1_4386_10724#
LI_3578_10634#
```

These names are mapped back to frontend nodes by matching MOS connection
patterns between the frontend topology and each PEX netlist.

Example mapping from one variant:

| PEX Node | Frontend Node |
| --- | --- |
| `M1_4386_10724#` | `NET11` |
| `M1_5332_8204#` | `NET7` |
| `LI_3578_10634#` | `NET16` |
| `M1_3408_7364#` | `VCAS` |

The full mapping for all variants is stored in:

```text
work/ahuja_ota_4_hash_node_mapping.json
```

## Important Distinctions

| Concept | Meaning |
| --- | --- |
| Frontend MOS count | The frontend netlist has 14 MOS devices. |
| PEX expanded MOS count | A PEX netlist can contain more extracted MOS instances, for example 36 in `variant_0`. |
| `mos_cnt` node feature | Counts unique frontend MOS devices connected to one node. It is not the global MOS count. |
| `GND` / `SUB` | `SUB` in PEX is mapped to frontend `GND`. |
| `0` in PEX capacitance | `node-0` capacitance is treated as node-to-ground label, not as a node-node edge. |

## Current Result Summary

| Item | Value |
| --- | ---: |
| Frontend devices | 15 |
| Frontend MOS devices | 14 |
| Frontend capacitors | 1 |
| Frontend nodes | 13 |
| Device-node edges | 58 |
| Node-node prediction edges | 78 |
| PEX variants used | 101 |
| Node-node labels | `78 / 78` |
| Node-ground labels | `13 / 13` |

