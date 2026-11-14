"""Step 1: download each raw source into data/raw and record its hash.

The Kaggle file needs a free Kaggle account: either run
`kaggle datasets download heeraldedhia/groceries-dataset -p data/raw --unzip`
or download it by hand from the URL in config.yaml. Every other source is fetched here.
"""
import urllib.request
from datetime import date
import yaml
from common import CFG, P, ROOT, md5

manifest = {}
for name, src in CFG["sources"].items():
    dest = P["raw"] / src["file"]
    if not dest.exists():
        if name == "kaggle":
            raise SystemExit(f"Missing {dest}. Download it from {src['url']} (see data/README.md).")
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(src["url"], dest)
    digest = md5(dest)
    if src.get("md5") and src["md5"] != digest:
        raise SystemExit(f"{name}: hash {digest} does not match config ({src['md5']}). The source changed.")
    manifest[name] = {"file": src["file"], "url": src["url"], "md5": digest,
                      "bytes": dest.stat().st_size, "checked_on": str(date.today())}
yaml.safe_dump(manifest, open(ROOT / "data" / "manifest.yaml", "w"), sort_keys=False)
print("manifest written:", ", ".join(manifest))
