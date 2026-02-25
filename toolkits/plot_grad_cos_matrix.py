#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plot gradient cosine similarity matrix from PCGrad .npz output.

Usage:
  python toolkits/plot_grad_cos_matrix.py <path_to.npz> [--out <figure_path>]
  python toolkits/plot_grad_cos_matrix.py <path_to_grad_cos_dir> [--out <dir>]

If input is a directory, all .npz files inside are plotted; figures are saved
alongside each .npz (or under --out if given).
"""

import argparse
import os

import numpy as np


def _triu_to_symmetric_matrix(cos_values: np.ndarray) -> np.ndarray:
    """Reconstruct n×n symmetric matrix from upper-triangle (offset=1) values."""
    # n*(n-1)/2 = len(cos_values) => n = (1 + sqrt(1+8*L)) / 2
    n_pairs = cos_values.size
    n = int(round((1 + (1 + 8 * n_pairs) ** 0.5) / 2))
    assert n * (n - 1) // 2 == n_pairs, (
        f"cos_values length {n_pairs} is not n*(n-1)/2 for any integer n"
    )
    M = np.eye(n, dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = cos_values[idx]
            M[j, i] = cos_values[idx]
            idx += 1
    return M


def plot_one(npz_path: str, out_path: str | None = None) -> str:
    """Load one .npz, plot grad cos matrix heatmap, save figure. Returns saved path."""
    data = np.load(npz_path)
    cos_values = data["cos_values"]

    if cos_values.size == 0:
        raise ValueError(f"No cos_values in {npz_path}")

    M = _triu_to_symmetric_matrix(cos_values)
    n = M.shape[0]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib") from None

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.45)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_yticklabels(np.arange(n))
    ax.set_xlabel("Task j")
    ax.set_ylabel("Task i")
    ax.set_title("cos(g_i, g_j)")

    for i in range(n):
        for j in range(n):
            text = ax.text(
                j, i, f"{M[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=min(8, 200 // n)
            )

    plt.colorbar(im, ax=ax, label="cosine")
    plt.tight_layout()

    if out_path is None:
        base, _ = os.path.splitext(npz_path)
        out_path = base + "_matrix.png"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot gradient cosine matrix from PCGrad .npz file(s)."
    )
    parser.add_argument(
        "input",
        help="Path to a .npz file or to a directory containing .npz files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path: for a single .npz, path to the figure; for a dir, directory for all figures",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        if not args.input.endswith(".npz"):
            raise SystemExit("Input must be a .npz file or a directory.")
        out_path = args.out
        if out_path and os.path.isdir(out_path):
            base = os.path.basename(args.input)
            out_path = os.path.join(out_path, os.path.splitext(base)[0] + "_matrix.png")
        saved = plot_one(args.input, out_path)
        print(f"Saved: {saved}")
        return

    if not os.path.isdir(args.input):
        raise SystemExit(f"Not a file or directory: {args.input}")

    out_dir = args.out or args.input
    if args.out:
        os.makedirs(out_dir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(args.input) if f.endswith(".npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files in {args.input}")

    for f in npz_files:
        npz_path = os.path.join(args.input, f)
        base = os.path.splitext(f)[0]
        out_path = os.path.join(out_dir, base + "_matrix.png") if args.out else None
        try:
            saved = plot_one(npz_path, out_path)
            print(f"Saved: {saved}")
        except Exception as e:
            print(f"Skip {npz_path}: {e}")


if __name__ == "__main__":
    main()
