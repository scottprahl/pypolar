#!/usr/bin/env python3
"""
Generate figures used by README.rst.

poincare1.svg
circular.gif

"""

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


def make_circular_gif(output_path: Path) -> None:
    """Create an animated GIF for left circularly polarized light."""
    from pypolar import jones
    from pypolar import visualization as vis

    v = jones.field_left_circular()
    print("Jones vector for left circularly polarized light")
    ani = vis.draw_jones_animated(v, nframes=32)
    ani.save(output_path, writer="pillow")
    plt.close(ani._fig)


def main() -> None:
    """Create README image artifacts in this directory."""
    image_dir = Path(__file__).resolve().parent

    poincare_path = image_dir / "poincare1.svg"
    make_poincare1(poincare_path)
    print(f"wrote {poincare_path}")

    circular_gif = image_dir / "circular.gif"
    make_circular_gif(circular_gif)
    print(f"wrote {circular_gif}")


if __name__ == "__main__":
    _add_repo_root_to_path()
    main()
