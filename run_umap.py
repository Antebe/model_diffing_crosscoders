"""
run_umap.py — decoder-direction UMAP figures for one crosscoder.

Outputs:
  - ``<out>/umap_partitions.png``        (Fig 1: 3-partition view)
  - ``<out>/umap_aexcl_clusters.png``    (Fig 2: A-excl HDBSCAN clusters)
  - ``<clusters_dir>/aexcl_assignments.csv``  (feature_idx, cluster_id, x, y)
  - ``<clusters_dir>/umap_meta.json``    (silhouette + cluster sizes)

Inputs:
  --crosscoder    HF repo id; we read W_dec from the loaded model.
  --out           Figures directory.
  --clusters-dir  CSV / metadata directory.
  --models-jsonl  Default ``models.jsonl``.

Dependencies (uv add):
  umap-learn  scikit-learn  matplotlib  numpy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sweep_eval import load_crosscoder


def _load_cc(crosscoder: str, models_jsonl: str, device: str):
    entries = [json.loads(l) for l in Path(models_jsonl).read_text().splitlines() if l.strip()]
    matches = [e for e in entries if e["name"] == crosscoder]
    if not matches:
        raise SystemExit(f"crosscoder {crosscoder!r} not in {models_jsonl}")
    cc = load_crosscoder(matches[0], device=device)
    if cc is None:
        raise SystemExit("crosscoder load failed")
    return cc


def _umap_2d(
    vectors: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
) -> np.ndarray:
    """Project ``vectors`` (N, D) to (N, 2) via UMAP."""
    import umap  # imported lazily so the script works for non-UMAP subcommands
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=seed,
    )
    return reducer.fit_transform(vectors)


def figure_partitions(
    coords: np.ndarray,
    n_a: int,
    n_b: int,
    dict_size: int,
    out_path: Path,
) -> None:
    """Fig 1: scatter colored by A-excl / B-excl / shared."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    def _scatter(start: int, end: int, color: str, label: str, alpha: float):
        c = coords[start:end]
        ax.scatter(c[:, 0], c[:, 1], s=6, c=color, alpha=alpha, label=label, linewidths=0)

    _scatter(n_a + n_b, dict_size, "#888888", f"shared (n={dict_size - n_a - n_b})", 0.20)
    _scatter(0,           n_a,        "#d62728", f"A-excl (n={n_a})", 0.85)
    _scatter(n_a,         n_a + n_b,  "#1f77b4", f"B-excl (n={n_b})", 0.85)

    ax.set_title("Decoder UMAP — partition view")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig1] {out_path}")


def figure_aexcl_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = "A-excl decoder UMAP — HDBSCAN clusters",
    cluster_names: dict[int, str] | None = None,
) -> None:
    """Fig 2: A-excl projection colored by cluster id; -1 = noise (grey)."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    uniq = sorted(set(int(x) for x in labels))
    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")

    color_idx = 0
    for c in uniq:
        mask = labels == c
        if c == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, c="#bbbbbb",
                       alpha=0.5, label="noise", linewidths=0)
        else:
            color = cmap(color_idx % cmap.N)
            color_idx += 1
            label = (cluster_names.get(c) if cluster_names else None) or f"cluster {c}"
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, c=[color],
                       alpha=0.9, label=f"{label} (n={int(mask.sum())})", linewidths=0)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig2] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crosscoder", required=True)
    p.add_argument("--out", required=True, type=Path,
                   help="figures dir (PNGs go here)")
    p.add_argument("--clusters-dir", required=True, type=Path,
                   help="cluster CSV / metadata dir")
    p.add_argument("--models-jsonl", default="models.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--min-cluster-size", type=int, default=20)
    p.add_argument("--min-samples", type=int, default=5)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.clusters_dir.mkdir(parents=True, exist_ok=True)

    cc = _load_cc(args.crosscoder, args.models_jsonl, args.device)
    n_a = int(getattr(cc, "n_a", 0))
    n_b = int(getattr(cc, "n_b", 0))
    dict_size = int(cc.dict_size)
    print(f"[load] dict={dict_size} n_a={n_a} n_b={n_b} shared={dict_size - n_a - n_b}")

    W = cc.W_dec.detach().cpu().numpy()              # (dict_size, 2, d)
    # Concatenate A-side + B-side decoder rows for each feature
    full = np.concatenate([W[:, 0, :], W[:, 1, :]], axis=1)   # (dict_size, 2d)

    # — Fig 1: 3-partition UMAP on full decoder vectors —
    print("[umap] full decoder ...")
    coords_full = _umap_2d(
        full, n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
    )
    figure_partitions(coords_full, n_a, n_b, dict_size, args.out / "umap_partitions.png")

    # — Fig 2: A-excl-only on Model A decoder rows + HDBSCAN —
    if n_a == 0:
        print("[umap] crosscoder has no A-excl partition — skipping Fig 2")
        return

    a_vectors = W[:n_a, 0, :]                         # (n_a, d)
    print(f"[umap] A-excl ({n_a} features) ...")
    coords_a = _umap_2d(
        a_vectors, n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed,
    )

    from sklearn.cluster import HDBSCAN
    print("[hdbscan] clustering A-excl UMAP ...")
    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    labels = clusterer.fit_predict(coords_a)

    n_clusters = len([c for c in set(labels) if c != -1])
    n_noise = int((labels == -1).sum())
    print(f"[hdbscan] {n_clusters} clusters; {n_noise} noise points")

    sil = None
    if n_clusters >= 2:
        try:
            from sklearn.metrics import silhouette_score
            mask = labels != -1
            if int(mask.sum()) > 2:
                sil = float(silhouette_score(coords_a[mask], labels[mask]))
                print(f"[hdbscan] silhouette={sil:.3f}")
        except Exception as e:                       # noqa: BLE001
            print(f"[hdbscan] silhouette failed: {e}")

    figure_aexcl_clusters(coords_a, labels, args.out / "umap_aexcl_clusters.png")

    # Persist cluster assignments + metadata
    csv_path = args.clusters_dir / "aexcl_assignments.csv"
    with csv_path.open("w") as f:
        f.write("feature_idx,cluster_id,umap_x,umap_y\n")
        for i in range(n_a):
            f.write(f"{i},{int(labels[i])},{coords_a[i, 0]:.4f},{coords_a[i, 1]:.4f}\n")
    print(f"[save] {csv_path}")

    cluster_sizes: dict[int, int] = {}
    for c in set(labels):
        cluster_sizes[int(c)] = int((labels == c).sum())
    meta = {
        "crosscoder": args.crosscoder,
        "n_a": n_a,
        "umap": {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "seed": args.seed},
        "hdbscan": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil,
            "cluster_sizes": cluster_sizes,
        },
    }
    (args.clusters_dir / "umap_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] {args.clusters_dir / 'umap_meta.json'}")


if __name__ == "__main__":
    main()
