#!/usr/bin/env python3
"""Generate SVG figures used by README.rst."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _add_repo_root_to_path() -> None:
    """Allow imports from the local repository without installation."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


def make_poincare1(output_path: Path) -> None:
    """Recreate poincare1.svg from the 05-Jones-Examples notebook."""
    from pypolar import jones
    from pypolar import visualization as vis

    b = jones.op_linear_polarizer(0)
    c = jones.op_quarter_wave_plate(np.pi / 4)
    d = jones.op_mirror()
    e = jones.op_quarter_wave_plate(-np.pi / 4)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    vis.draw_empty_sphere(ax)

    j1 = jones.field_elliptical(np.pi / 6, np.pi / 6)
    j2 = b @ j1
    j3 = c @ j2
    j4 = d @ j3
    j5 = e @ j4

    vis.draw_jones_poincare(j1, ax, label="  start", color="red", va="center")
    vis.draw_jones_poincare(j2, ax, label="  after Polarizer", color="blue", va="center")
    vis.draw_jones_poincare(j3, ax, label="  after QWP", color="blue", va="center")
    vis.draw_jones_poincare(j4, ax, label="  after mirror", color="blue", va="center")
    vis.draw_jones_poincare(j5, ax, label="  final", color="red", va="center")

    vis.join_jones_poincare(j1, j2, ax, color="blue", lw=2, linestyle=":")
    vis.join_jones_poincare(j2, j3, ax, color="blue", lw=2, linestyle=":")
    vis.join_jones_poincare(j3, j4, ax, color="blue", lw=2, linestyle=":")
    vis.join_jones_poincare(j4, j5, ax, color="blue", lw=2, linestyle=":")

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def make_isolator_svg(input_png: Path, output_svg: Path) -> None:
    """
    Wrap isolator.png into an SVG file.

    README currently uses a PNG source for the isolator diagram, so this keeps
    the same visual while producing an SVG artifact beside it.
    """
    image = mpimg.imread(input_png)
    height, width = image.shape[:2]
    dpi = 150
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_svg, format="svg", dpi=dpi)
    plt.close(fig)


def main() -> None:
    """Create README image artifacts in this directory."""
    image_dir = Path(__file__).resolve().parent

    poincare_path = image_dir / "poincare1.svg"
    make_poincare1(poincare_path)
    print(f"wrote {poincare_path}")

    isolator_png = image_dir / "isolator.png"
    isolator_svg = image_dir / "isolator.svg"
    if isolator_png.exists():
        make_isolator_svg(isolator_png, isolator_svg)
        print(f"wrote {isolator_svg}")
    else:
        print(f"skipped isolator.svg (missing {isolator_png})")


if __name__ == "__main__":
    _add_repo_root_to_path()
    main()
