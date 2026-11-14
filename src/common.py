"""Shared helpers: config, paths, plotting style."""
import hashlib, json
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
P = {k: ROOT / v for k, v in CFG["paths"].items()}
for p in P.values():
    p.mkdir(parents=True, exist_ok=True)

# colours: one accent for observed data, grey for chance, orange as second series
BLUE, ORANGE, GREEN, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 13.5, "legend.fontsize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 170, "savefig.bbox": "tight",
})


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_numbers(name, d):
    """Merge a dict of headline numbers into reports/numbers.json."""
    f = ROOT / "reports" / "numbers.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[name] = d
    f.write_text(json.dumps(cur, indent=2, default=float))
