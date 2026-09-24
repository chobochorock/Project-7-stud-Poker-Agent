import json
from pathlib import Path

import matplotlib.pyplot as plt


def final_json(path: Path) -> dict:
    data = path.read_bytes()
    text = data.decode(
        "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8",
        errors="replace",
    )
    agent = text.rfind('"agent_a"')
    start = text.rfind("{", 0, agent)
    if agent < 0 or start < 0:
        raise RuntimeError(f"final evaluation JSON not found in {path}")
    return json.loads(text[start:])


output_dir = Path("agents/cpp_mccfr/data/results/seventh_resolver_240p")
experiments = [
    ("30M baseline", "baseline.log"),
    ("iter=100\nprior=100", "iter100_prior100.log"),
    ("iter=100\nprior=1000", "iter100_prior1000.log"),
]
results = [(label, final_json(output_dir / name)) for label, name in experiments]
means = [result["average_profit_ante_for_lbr"] for _, result in results]
errors = [1.96 * result["paired_standard_error_ante"] for _, result in results]

fig, ax = plt.subplots(figsize=(8.4, 5.2))
bars = ax.bar(
    [label for label, _ in results],
    means,
    yerr=errors,
    capsize=6,
    color=["#5B6573", "#277DA1", "#43AA8B"],
)
ax.set_ylabel("LBR profit (ante/hand, lower is better)")
ax.set_title("7th-street resolver: 10,000 hands, 240 particles")
ax.grid(axis="y", alpha=0.25)
for bar, mean in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{mean:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
fig.tight_layout()
fig.savefig(output_dir / "seventh_resolver_240p.png", dpi=180)
plt.close(fig)
