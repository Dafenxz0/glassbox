"""Assemble the static Pages and Sites distribution from its runtime assets."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> None:
    assets = (
        "index.html", "favicon.svg", "share.jpg", "assets/glassbox.css",
        "assets/home.js", "assets/drone-surface.svg", "assets/drone-trace.json", "assets/fonts",
    )
    for name in assets:
        src = ROOT / "site" / name
        dest = DIST / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
    shutil.copytree(ROOT / "studio", DIST / "studio", dirs_exist_ok=True)
    (DIST / ".nojekyll").touch()


if __name__ == "__main__":
    main()
