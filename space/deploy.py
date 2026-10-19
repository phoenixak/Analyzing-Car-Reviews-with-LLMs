"""Copy the Space files into a local folder and upload them to the Hugging Face Space.

python space/deploy.py ../hf-car-reviews-analyzer
hf upload Phoenixak99/car-reviews-analyzer ../hf-car-reviews-analyzer . --repo-type space --delete "*"
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["app.py", "style.css", "requirements.txt", "README.md", "Dockerfile"]


def main(target: str) -> None:
    dst = Path(target).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for item in dst.iterdir():
        if item.name not in {".git", ".gitattributes"}:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    for name in FILES:
        shutil.copy2(ROOT / "space" / name, dst / name)
    shutil.copytree(
        ROOT / "src" / "review_intel", dst / "src" / "review_intel", ignore=shutil.ignore_patterns("__pycache__")
    )
    print(f"copied the Space into {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "hf-car-reviews-analyzer"))
