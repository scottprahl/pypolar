"""Update CITATION.cff and README.rst citation block from latest GitHub release.

- Fetches the latest release date from GitHub for USERNAME/REPO.
- Reads the version from REPO/__init__.py (__version__ variable).
- Updates:
    * CITATION.cff:
        - date-released  <- from GitHub latest release
        - version        <- from __init__.__version__
        - preferred-citation.year    (if present)
        - preferred-citation.version (if present)
    * README.rst:
        - year in the prose citation line ("Prahl, S. (YYYY).")
        - Version in the prose citation "(Version X.Y.Z)"
        - year in the BibTeX key "@software{<repo>_prahl_YYYY,"
        - year field "year      = {YYYY},"
        - version field "version   = {X.Y.Z},"

The Zenodo DOI and URL are assumed to be stable and are not modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests
import yaml

# --------------------------------------------------------------------
# Configuration: change REPO per project
# --------------------------------------------------------------------
USERNAME = "scottprahl"
REPO = "pypolar"  # <-- only change this per repo

GITHUB_API_URL = f"https://api.github.com/repos/{USERNAME}/{REPO}/releases/latest"

HEADERS = {
    "User-Agent": f"{REPO}-citation-updater",
    "Accept": "application/vnd.github+json",
}


def get_release_date() -> tuple[str, str]:
    """Return (release_date, year) from the latest GitHub release."""
    response = requests.get(GITHUB_API_URL, timeout=10, headers=HEADERS)
    response.raise_for_status()
    release_info = response.json()

    release_date = release_info["published_at"].split("T")[0]  # e.g. "2025-11-17"
    year = release_date.split("-")[0]
    tag_version = release_info.get("tag_name", "").lstrip("v")

    print(
        f"GitHub latest release → tag: {release_info.get('tag_name')}, "
        f"version (from tag): {tag_version}, date: {release_date}"
    )
    return release_date, year


def get_code_version() -> str:
    """Extract __version__ from REPO/__init__.py."""
    init_path = Path(REPO) / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"{init_path} not found; cannot read __version__")

    text = init_path.read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        raise RuntimeError(
            f"Could not find __version__ = 'x.y.z' in {init_path}"
        )
    version = m.group(1).strip()
    print(f"Version from {init_path} → {version}")
    return version


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
release_date, year = get_release_date()
version = get_code_version()

# --------------------------------------------------------------------
# Update CITATION.cff
# --------------------------------------------------------------------
citation_path = Path("CITATION.cff")
if citation_path.exists():
    with citation_path.open("r", encoding="utf-8") as f:
        cff_data = yaml.safe_load(f)

    if not isinstance(cff_data, dict):
        raise RuntimeError("CITATION.cff does not contain a YAML mapping at top level.")

    cff_changed = False

    # Top-level date-released from GitHub
    if cff_data.get("date-released") != release_date:
        cff_data["date-released"] = release_date
        cff_changed = True

    # Top-level version from __init__.__version__
    if cff_data.get("version") != version:
        cff_data["version"] = version
        cff_changed = True

    # Optional: preferred-citation block (if present)
    preferred = cff_data.get("preferred-citation")
    if isinstance(preferred, dict):
        # year
        if str(preferred.get("year")) != str(year):
            preferred["year"] = int(year) if year.isdigit() else year
            cff_changed = True

        # version
        if preferred.get("version") != version:
            preferred["version"] = version
            cff_changed = True

        cff_data["preferred-citation"] = preferred

    if cff_changed:
        with citation_path.open("w", encoding="utf-8") as f:
            yaml.dump(cff_data, f, sort_keys=False)
        print(f"CITATION.cff updated → version: {version}, date: {release_date}")
    else:
        print("CITATION.cff: no change in release date, version, or preferred-citation.")
else:
    print("CITATION.cff not found; skipping CITATION.cff update.")

# --------------------------------------------------------------------
# Update citation block in README.rst
# --------------------------------------------------------------------
readme_path = Path("README.rst")
if not readme_path.exists():
    print("README.rst not found; skipping README citation update.")
else:
    text = readme_path.read_text(encoding="utf-8")
    original_text = text

    # 1. Prose citation year:
    #    Prahl, S. (2025). *ofiber: ...* (Version 0.9.0) [Computer software]. ...
    text = re.sub(
        r"(Prahl,\s*S\.\s*\()(\d{4})(\)\.)",
        lambda m: f"{m.group(1)}{year}{m.group(3)}",
        text,
    )

    # 2. Prose citation version: "(Version X.Y.Z)"
    text = re.sub(
        r"\(Version [^)]+\)",
        f"(Version {version})",
        text,
    )

    # 3. BibTeX key:
    #    @software{ofiber_prahl_2025,
    # Make this generic over the repo name: "<whatever>_prahl_YYYY"
    text = re.sub(
        r"(@software\{[A-Za-z0-9_]+_prahl_)(\d{4})(\s*,)",
        lambda m: f"{m.group(1)}{year}{m.group(3)}",
        text,
    )

    # 4. BibTeX year field:
    #    year      = {2025},
    text = re.sub(
        r"(year\s*=\s*\{)(\d{4})(\s*\},)",
        lambda m: f"{m.group(1)}{year}{m.group(3)}",
        text,
    )

    # 5. BibTeX version field:
    #    version   = {0.9.0},
    text = re.sub(
        r"(version\s*=\s*\{)([^}]+)(\s*\},)",
        lambda m: f"{m.group(1)}{version}{m.group(3)}",
        text,
    )

    if text != original_text:
        readme_path.write_text(text, encoding="utf-8")
        print(f"README.rst citation block updated → version: {version}, year: {year}")
    else:
        print("README.rst citation block already up to date.")
