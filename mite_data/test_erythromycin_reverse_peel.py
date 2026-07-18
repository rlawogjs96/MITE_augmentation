"""Test BGC0000054 (Erythromycin A) reverse peel against mite_rdchiral_rules.tsv.

Literature (Established): 6-dEB → EryF → EryBV → EryCIII → EryK → EryG → Ery A
Reverse peel: Ery A → −EryG → −EryK → −EryCIII → −EryBV → −EryF → 6-dEB

Writes:
  - mite_data/erythromycin_reverse_peel_test.json  (match report)
  - erythromycin_reverse_peel_mite.png            (figure)
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "mite_rdchiral_rules.tsv"
OUT_JSON = Path(__file__).resolve().parent / "erythromycin_reverse_peel_test.json"
OUT_PNG = ROOT / "erythromycin_reverse_peel_mite.png"

# Intermediate SMILES (canonical literature / PubChem stereo)
INTERMEDIATES = {
    "Erythromycin A": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "Erythromycin C": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(O)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "Erythromycin D": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(O)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@H]1C",
    "3-O-mycarosyl-EB": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(O)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@H]1C",
    "Erythronolide B": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@H]1C",
    "6-Deoxyerythronolide B": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@@H](C)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@H]1C",
}

# Reverse sequence: enzyme undone, expected substrate→product, MITE term filter
REVERSE_STEPS = [
    {
        "step": 1,
        "enzyme": "EryG",
        "label": "− O-methylation (EryG)",
        "term_needles": ["Methylation"],
        "substrate": "Erythromycin A",
        "expected": "Erythromycin C",
        "color": "#c62828",
    },
    {
        "step": 2,
        "enzyme": "EryK",
        "label": "− C12 hydroxylation (EryK)",
        "term_needles": ["Hydroxylation", "Monooxygenation"],
        "substrate": "Erythromycin C",
        "expected": "Erythromycin D",
        "color": "#1565c0",
    },
    {
        "step": 3,
        "enzyme": "EryCIII",
        "label": "− desosamine glycosylation (EryCIII)",
        "term_needles": ["Glycosylation"],
        "substrate": "Erythromycin D",
        "expected": "3-O-mycarosyl-EB",
        "color": "#6a1b9a",
    },
    {
        "step": 4,
        "enzyme": "EryBV",
        "label": "− mycarose glycosylation (EryBV)",
        "term_needles": ["Glycosylation"],
        "substrate": "3-O-mycarosyl-EB",
        "expected": "Erythronolide B",
        "color": "#ef6c00",
    },
    {
        "step": 5,
        "enzyme": "EryF",
        "label": "− C6 hydroxylation (EryF)",
        "term_needles": ["Hydroxylation", "Monooxygenation"],
        "substrate": "Erythronolide B",
        "expected": "6-Deoxyerythronolide B",
        "color": "#2e7d32",
    },
]


@dataclass
class Hit:
    rule_name: str
    terms: str
    n_examples: int
    streptomycetaceae: str
    retro_smarts: str
    n_products: int
    exact_expected: bool


def inchikey(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol)


def load_rules(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rule_matches_terms(rule: dict, needles: list[str]) -> bool:
    terms = rule.get("tailoring_terms") or ""
    return any(n in terms for n in needles)


def apply_retro(rule: dict, mol: Chem.Mol) -> tuple[list[Chem.Mol], int]:
    """Return (sanitized products, raw_hit_count including unsanitized)."""
    smarts = rule["retro_SMARTS"]
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
    except Exception:
        return [], 0
    if rxn is None:
        return [], 0
    # multi-component retro templates cannot run on a single mol
    if rxn.GetNumReactantTemplates() != 1:
        return [], 0
    try:
        outs = rxn.RunReactants((mol,))
    except Exception:
        return [], 0
    products: list[Chem.Mol] = []
    raw = 0
    for tup in outs:
        if not tup:
            continue
        raw += 1
        p = Chem.Mol(tup[0])
        try:
            Chem.SanitizeMol(p)
            products.append(p)
        except Exception:
            # still a reaction hit; valence often breaks on sugar leave-groups
            continue
    return products, raw


def test_step(step: dict, rules: list[dict]) -> dict:
    sub = Chem.MolFromSmiles(INTERMEDIATES[step["substrate"]])
    exp = Chem.MolFromSmiles(INTERMEDIATES[step["expected"]])
    assert sub and exp
    exp_key = inchikey(exp)

    candidates = [r for r in rules if rule_matches_terms(r, step["term_needles"])]
    hits: list[Hit] = []
    exact_rules: list[str] = []
    n_raw_firing = 0

    for r in candidates:
        products, raw = apply_retro(r, sub)
        if raw == 0:
            continue
        n_raw_firing += 1
        exact = any(inchikey(p) == exp_key for p in products) if products else False
        if exact:
            exact_rules.append(r["rule_name"])
        hits.append(
            Hit(
                rule_name=r["rule_name"],
                terms=r["tailoring_terms"],
                n_examples=int(r["n_examples"]),
                streptomycetaceae=r["streptomycetaceae"],
                retro_smarts=r["retro_SMARTS"],
                n_products=len(products),
                exact_expected=exact,
            )
        )

    exact_hits = [h for h in hits if h.exact_expected]
    exact_hits.sort(key=lambda h: -h.n_examples)
    sanitized_hits = [h for h in hits if h.n_products > 0]
    any_hits = sorted(hits, key=lambda h: (-h.exact_expected, -h.n_products, -h.n_examples))

    return {
        "step": step["step"],
        "enzyme": step["enzyme"],
        "label": step["label"],
        "substrate": step["substrate"],
        "expected": step["expected"],
        "term_needles": step["term_needles"],
        "n_candidate_rules": len(candidates),
        "n_firing_rules": len(sanitized_hits),
        "n_raw_firing_rules": n_raw_firing,
        "n_exact_rules": len(exact_hits),
        "fires": len(sanitized_hits) > 0,
        "fires_raw": n_raw_firing > 0,
        "exact_match": len(exact_hits) > 0,
        "best_rule": asdict(exact_hits[0]) if exact_hits else (asdict(any_hits[0]) if any_hits else None),
        "exact_rule_names": exact_rules[:12],
    }


def mol_to_png_bytes(mol: Chem.Mol, size=(320, 240)) -> bytes:
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = d.drawOptions()
    opts.clearBackground = True
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def render_figure(results: list[dict], path: Path) -> None:
    """Sequential reverse-peel panel inspired by the augmentation figure."""
    names = [
        "Erythromycin A",
        "Erythromycin C",
        "Erythromycin D",
        "3-O-mycarosyl-EB",
        "Erythronolide B",
        "6-Deoxyerythronolide B",
    ]
    mols = [Chem.MolFromSmiles(INTERMEDIATES[n]) for n in names]
    imgs = [Draw.MolToImage(m, size=(380, 280)) for m in mols]

    fig = plt.figure(figsize=(18, 7.2), facecolor="white")
    fig.suptitle(
        "BGC0000054 Erythromycin A — reverse peel with mite_rdchiral_rules",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        "Literature Established (flip): Ery A  →  −EryG  →  −EryK  →  −EryCIII  →  −EryBV  →  −EryF  →  6-dEB (PKS core)",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    n = len(mols)
    # leave room for arrows between panels
    left = 0.02
    right = 0.98
    width = (right - left) / n
    y0 = 0.12
    h = 0.72

    for i, (name, img) in enumerate(zip(names, imgs)):
        x = left + i * width
        ax = fig.add_axes([x + 0.01, y0, width - 0.02, h])
        ax.imshow(img)
        ax.axis("off")
        # status for transition into this node (i>0)
        if i == 0:
            status = "final product"
            status_color = "#333333"
        else:
            r = results[i - 1]
            if r["exact_match"]:
                status = f"EXACT  ({r['n_exact_rules']} rules)"
                status_color = "#2e7d32"
            elif r["fires"]:
                status = f"fires, not exact  ({r['n_firing_rules']})"
                status_color = "#ef6c00"
            elif r.get("fires_raw"):
                status = f"hits, unsanitized  ({r['n_raw_firing_rules']})"
                status_color = "#ef6c00"
            else:
                status = "NO FIRE"
                status_color = "#c62828"
        ax.set_title(name, fontsize=9, pad=4)
        # label under molecule
        if i == 0:
            fig.text(
                x + width / 2,
                0.08,
                "start",
                ha="center",
                fontsize=8,
                color="#666666",
            )
        else:
            step = REVERSE_STEPS[i - 1]
            fig.text(
                x + width / 2,
                0.095,
                step["label"],
                ha="center",
                fontsize=8,
                color=step["color"],
                fontweight="bold",
            )
            fig.text(
                x + width / 2,
                0.055,
                status,
                ha="center",
                fontsize=7.5,
                color=status_color,
            )

        # draw arrow to next
        if i < n - 1:
            fig.patches.append(
                FancyBboxPatch(
                    (0, 0),
                    0,
                    0,
                    transform=fig.transFigure,
                    boxstyle="square,pad=0",
                    mutation_aspect=1,
                    visible=False,
                )
            )
            ax_arrow = fig.add_axes([x + width - 0.012, y0 + h * 0.45, 0.024, 0.08])
            ax_arrow.set_xlim(0, 1)
            ax_arrow.set_ylim(0, 1)
            ax_arrow.axis("off")
            ax_arrow.annotate(
                "",
                xy=(1, 0.5),
                xytext=(0, 0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )

    # footer summary
    exact = sum(1 for r in results if r["exact_match"])
    fires = sum(1 for r in results if r["fires"] or r.get("fires_raw"))
    fig.text(
        0.5,
        0.015,
        f"mite_rdchiral_rules.tsv · {exact}/5 exact · {fires}/5 fire/hit · "
        "source: post-pks-tailoring-literature BGC0000054 Established",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rules = load_rules(RULES)
    results = [test_step(step, rules) for step in REVERSE_STEPS]

    report = {
        "bgc": "BGC0000054",
        "product": "Erythromycin A",
        "order": "Established",
        "forward_sequence": "6-dEB → EryF → EryBV → EryCIII → EryK → EryG → Erythromycin A",
        "reverse_sequence": "Ery A → −EryG → −EryK → −EryCIII → −EryBV → −EryF → 6-dEB",
        "rules_file": str(RULES),
        "n_rules": len(rules),
        "steps": results,
        "summary": {
            "exact_steps": sum(1 for r in results if r["exact_match"]),
            "firing_steps": sum(1 for r in results if r["fires"]),
            "raw_hit_steps": sum(1 for r in results if r.get("fires_raw")),
            "total_steps": len(results),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    render_figure(results, OUT_PNG)

    print(json.dumps(report["summary"], indent=2))
    for r in results:
        if r["exact_match"]:
            flag = "EXACT"
        elif r["fires"]:
            flag = "FIRE"
        elif r.get("fires_raw"):
            flag = "RAW"
        else:
            flag = "MISS"
        best = r["best_rule"]["rule_name"] if r["best_rule"] else "-"
        print(
            f"step{r['step']} {r['enzyme']:8s} {flag:5s} candidates={r['n_candidate_rules']:3d} "
            f"fire={r['n_firing_rules']:3d} raw={r['n_raw_firing_rules']:3d} "
            f"exact={r['n_exact_rules']:3d} best={best}"
        )
    print("wrote", OUT_JSON)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
