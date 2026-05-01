"""
run_cluster_meta.py — meta-autointerp: name each A-excl cluster from its
member features' Gemma-generated explanations.

Inputs:
  --autointerp     dir of feat_*.json from run_autointerp_local.py
  --clusters-csv   aexcl_assignments.csv from run_umap.py
  --umap-meta      umap_meta.json (used for cluster sizes + silhouette echo)
  --out            cluster_meta.json output path
  --figure-out     optional re-rendered Fig 2 with cluster names overlaid

For each cluster id != -1:
  - gather member feature explanations (skipping dead / errored features)
  - prompt Gemma: name (2–5 words) + one-sentence summary
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from autointerp_local import LocalGemmaClient
from run_umap import figure_aexcl_clusters

CLUSTER_PROMPT_SYSTEM = """\
You are an interpretability researcher. You will be shown N descriptions of
sparse-autoencoder features that were grouped together based on
decoder-direction similarity. Your job is to identify the common theme.

Reply on EXACTLY two lines, no other text:
[CLUSTER_NAME]: <2 to 5 word label>
[CLUSTER_SUMMARY]: <one sentence summarising the shared theme>"""


CLUSTER_PROMPT_USER = """\
Cluster member feature descriptions (truncated to 200 chars):

{members}

What is the shared theme?"""


def _truncate(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) > n:
        s = s[:n].rstrip() + "…"
    return s


def parse_meta(raw: str) -> tuple[str, str]:
    name, summary = "", ""
    for line in raw.splitlines():
        if line.startswith("[CLUSTER_NAME]:"):
            name = line[len("[CLUSTER_NAME]:"):].strip()
        elif line.startswith("[CLUSTER_SUMMARY]:"):
            summary = line[len("[CLUSTER_SUMMARY]:"):].strip()
    if not name:
        name = (raw.strip().split("\n", 1)[0])[:60] or "(unnamed)"
    if not summary:
        summary = "(no summary parsed)"
    return name, summary


def load_assignments(csv_path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["feature_idx"])] = int(row["cluster_id"])
    return out


def load_explanations(ai_dir: Path) -> dict[int, str]:
    """Return {feature_idx: explanation} for non-dead features."""
    out: dict[int, str] = {}
    for sub in ai_dir.iterdir():
        if not sub.is_dir():
            continue
        for fp in sub.glob("feat_*.json"):
            try:
                data = json.loads(fp.read_text())
            except json.JSONDecodeError:
                continue
            idx = int(data.get("feat_idx", -1))
            expl = data.get("explanation", "") or ""
            if idx >= 0 and not data.get("is_dead", False) and expl.strip():
                out[idx] = expl
    return out


async def name_clusters(
    members_per_cluster: dict[int, list[str]],
    gemma_model: str,
    device: str,
    max_new_tokens: int,
) -> dict[int, dict]:
    client = LocalGemmaClient(
        model_id=gemma_model, device=device,
        max_new_tokens=max_new_tokens, max_concurrent=1,
    )
    out: dict[int, dict] = {}
    for cid in sorted(members_per_cluster):
        descs = members_per_cluster[cid]
        body = "\n".join(f"  {i+1}. {_truncate(d)}" for i, d in enumerate(descs))
        raw = await client.call(
            CLUSTER_PROMPT_SYSTEM,
            CLUSTER_PROMPT_USER.format(members=body),
        )
        name, summary = parse_meta(raw)
        out[cid] = {
            "name": name,
            "summary": summary,
            "n_members": len(descs),
            "raw_response": raw,
        }
        print(f"[cluster {cid}] n={len(descs)}  {name}  —  {summary}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--autointerp", required=True, type=Path)
    p.add_argument("--clusters-csv", required=True, type=Path)
    p.add_argument("--umap-meta", required=False, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--figure-out", type=Path, default=None,
                   help="optional path to re-render Fig 2 with named clusters")
    p.add_argument("--gemma-model", default="google/gemma-2-9b-it")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args()

    assignments = load_assignments(args.clusters_csv)
    explanations = load_explanations(args.autointerp)
    print(f"[load] {len(assignments)} assignments  /  "
          f"{len(explanations)} non-dead explanations")

    members_per_cluster: dict[int, list[str]] = {}
    member_indices_per_cluster: dict[int, list[int]] = {}
    for fidx, cid in assignments.items():
        if cid == -1:
            continue
        if fidx not in explanations:
            continue
        members_per_cluster.setdefault(cid, []).append(explanations[fidx])
        member_indices_per_cluster.setdefault(cid, []).append(fidx)

    if not members_per_cluster:
        raise SystemExit("no clusters with named members; did autointerp run?")

    cluster_meta = asyncio.run(name_clusters(
        members_per_cluster, args.gemma_model, args.device, args.max_new_tokens,
    ))
    for cid, data in cluster_meta.items():
        data["members"] = member_indices_per_cluster[cid]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cluster_meta, indent=2))
    print(f"[save] {args.out}")

    # Re-render figure with names if requested
    if args.figure_out:
        import numpy as np
        coords = np.zeros((len(assignments), 2), dtype=np.float32)
        labels = np.full(len(assignments), -1, dtype=np.int64)
        with args.clusters_csv.open() as f:
            for row in csv.DictReader(f):
                i = int(row["feature_idx"])
                coords[i] = (float(row["umap_x"]), float(row["umap_y"]))
                labels[i] = int(row["cluster_id"])
        names = {int(cid): m["name"] for cid, m in cluster_meta.items()}
        figure_aexcl_clusters(
            coords, labels, args.figure_out,
            title="A-excl decoder UMAP — labelled clusters",
            cluster_names=names,
        )


if __name__ == "__main__":
    main()
