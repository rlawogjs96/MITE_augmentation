"""Grammar-guided forward augmentation of Library A/B with mite_rdchiral.tsv.

See AUGMENTATION_LOGIC.md for the policy.
Uses forward_SMARTS only; preserves PKS core subgraph; decorate-class allow-list.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
DEFAULT_RULES = ROOT / "mite_rdchiral.tsv"
DEFAULT_OUT = ROOT / "augmentations" / "mite_forward_grammar"
DEFAULT_LIB_A = ROOT / "data" / "library_a_SMILES.txt"
DEFAULT_LIB_B = ROOT / "data" / "library_b_SMILES.txt"

# Decorate families (MITE tailoring_class primary token)
DECORATE_ALLOW = {
    "Methylation",
    "Acetylation",
    "Acylation",
    "Glycosylation",
    "Halogenation",
    "Hydroxylation",
    "Monooxygenation",
    "Phosphorylation",
    "Prenylation",
    "Amination",
}

DECORATE_BLOCK = {
    "Biaryl bond formation",
    "Cyclization",
    "Heterocyclization",
    "Ring contraction",
    "Macrolactam formation",
    "Decarboxylation",
    "Reduction",
    "Dehydration",
    "Dehydrogenation",
    "Hydrolysis",
    "Carboxylation",
    "Dioxygenation",
    "Other",
    "Oxidation",  # often backbone redox; allow only if subgraph+heavy-atom gates pass via exception path — blocked by default
}

# Handle detectors
HANDLE_SMARTS = {
    "aliphatic_oh": "[CX4][OX2H]",
    "olefin": "[CX3]=[CX3]",
    "ketone": "[#6](=[#8])[#6]",
    "aryl": "a",
}

CLASS_HANDLES: dict[str, tuple[str, ...]] = {
    "Glycosylation": ("aliphatic_oh",),
    "Methylation": ("aliphatic_oh", "aryl"),
    "Acetylation": ("aliphatic_oh",),
    "Acylation": ("aliphatic_oh", "aryl"),
    "Phosphorylation": ("aliphatic_oh",),
    "Halogenation": ("aryl", "olefin"),
    "Hydroxylation": ("aryl", "olefin", "aliphatic_oh", "ketone"),
    "Monooxygenation": ("olefin", "aryl", "ketone"),
    "Prenylation": ("aryl", "aliphatic_oh"),
    "Amination": ("ketone", "olefin"),
}


@dataclass
class CompiledRule:
    rule_name: str
    terms: str
    primary_class: str
    forward_smarts: str
    projected: str
    rxn: object
    reactant_template: object


def primary_class(terms: str) -> str:
    return (terms or "").split("|")[0].strip()


def project_forward(smarts: str) -> str | None:
    if ">>" not in smarts:
        return None
    left, right = smarts.split(">>", maxsplit=1)
    # Prefer largest reactant fragment as scaffold when multi-component
    left_parts = [p for p in left.split(".") if p]
    right_parts = [p for p in right.split(".") if p]
    if not left_parts or not right_parts:
        return None
    left0 = max(left_parts, key=len)
    right0 = max(right_parts, key=len)
    return f"{left0}>>{right0}"


def load_and_compile_rules(path: Path) -> list[CompiledRule]:
    compiled: list[CompiledRule] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("source") and row["source"] != "MITE_rdchiral":
                continue
            pc = primary_class(row.get("tailoring_class") or "")
            if pc in DECORATE_BLOCK or pc not in DECORATE_ALLOW:
                continue
            fwd = (row.get("forward_SMARTS") or "").strip()
            projected = project_forward(fwd)
            if not projected:
                continue
            try:
                rxn = AllChem.ReactionFromSmarts(projected)
            except Exception:
                continue
            if rxn is None or rxn.GetNumReactantTemplates() != 1:
                continue
            compiled.append(
                CompiledRule(
                    rule_name=row["rule_name"],
                    terms=row.get("tailoring_class") or "",
                    primary_class=pc,
                    forward_smarts=fwd,
                    projected=projected,
                    rxn=rxn,
                    reactant_template=rxn.GetReactantTemplate(0),
                )
            )
    return compiled


def detect_handles(mol: Chem.Mol) -> set[str]:
    found: set[str] = set()
    for name, sma in HANDLE_SMARTS.items():
        patt = Chem.MolFromSmarts(sma)
        if patt is not None and mol.HasSubstructMatch(patt):
            found.add(name)
    return found


def class_eligible(pc: str, handles: set[str]) -> bool:
    needed = CLASS_HANDLES.get(pc)
    if not needed:
        return True
    return any(h in handles for h in needed)


def core_preserved(core: Chem.Mol, product: Chem.Mol, min_frac: float = 0.98) -> bool:
    """Require nearly all core heavy atoms to appear in an MCS with the product."""
    if product.GetNumHeavyAtoms() < core.GetNumHeavyAtoms():
        return False
    try:
        res = rdFMCS.FindMCS(
            [core, product],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            matchValences=False,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
            timeout=2,
        )
    except Exception:
        return False
    if res.canceled or res.numAtoms < 1:
        return False
    return res.numAtoms >= int(min_frac * core.GetNumHeavyAtoms())


def apply_rule(rule: CompiledRule, mol: Chem.Mol, max_products: int = 8) -> list[Chem.Mol]:
    if not mol.HasSubstructMatch(rule.reactant_template):
        return []
    try:
        outs = rule.rxn.RunReactants((mol,))
    except Exception:
        return []
    products: list[Chem.Mol] = []
    seen: set[str] = set()
    for tup in outs:
        if not tup:
            continue
        p = Chem.Mol(tup[0])
        try:
            Chem.SanitizeMol(p)
        except Exception:
            continue
        smi = Chem.MolToSmiles(p, isomericSmiles=False)
        if smi in seen:
            continue
        seen.add(smi)
        products.append(p)
        if len(products) >= max_products:
            break
    return products


def eligible_edits(
    mol: Chem.Mol,
    core: Chem.Mol,
    rules: list[CompiledRule],
    visited: set[str],
) -> list[tuple[CompiledRule, str]]:
    handles = detect_handles(mol)
    options: list[tuple[CompiledRule, str]] = []
    mol_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    for rule in rules:
        if not class_eligible(rule.primary_class, handles):
            continue
        for product in apply_rule(rule, mol):
            if not core_preserved(core, product):
                continue
            # Prefer additive or equal size (local O insert can be equal C count with +O)
            if product.GetNumHeavyAtoms() < mol.GetNumHeavyAtoms():
                continue
            smi = Chem.MolToSmiles(product, isomericSmiles=False)
            if smi == mol_smi or smi in visited:
                continue
            options.append((rule, smi))
    return options


def choose_edit(
    opts: list[tuple[CompiledRule, str]],
    rng: random.Random,
    balance: str,
) -> tuple[CompiledRule, str]:
    """Sample one edit. `class` = uniform over eligible classes, then over that class's opts.
    `rule` = uniform over distinct rules with ≥1 opt, then over that rule's opts.
    `uniform` = flat over all (rule, product) pairs (OH-heavy).
    """
    if balance == "uniform" or len(opts) == 1:
        return rng.choice(opts)
    if balance == "rule":
        by_rule: dict[str, list[tuple[CompiledRule, str]]] = {}
        for rule, smi in opts:
            by_rule.setdefault(rule.rule_name, []).append((rule, smi))
        chosen_rule = rng.choice(list(by_rule.keys()))
        return rng.choice(by_rule[chosen_rule])
    # default: class-balanced
    by_class: dict[str, list[tuple[CompiledRule, str]]] = {}
    for rule, smi in opts:
        by_class.setdefault(rule.primary_class, []).append((rule, smi))
    chosen_class = rng.choice(list(by_class.keys()))
    return rng.choice(by_class[chosen_class])


def one_walk(
    core_smi: str,
    rules: list[CompiledRule],
    rng: random.Random,
    min_steps: int,
    max_steps: int,
    balance: str = "class",
) -> dict | None:
    core = Chem.MolFromSmiles(core_smi)
    if core is None:
        return None
    core_canon = Chem.MolToSmiles(core, isomericSmiles=False)
    core_heavy = core.GetNumHeavyAtoms()
    n_target = rng.randint(min_steps, max_steps)

    current = core_canon
    step_smiles = [core_canon]
    step_rules: list[str] = []
    step_classes: list[str] = []
    visited = {core_canon}

    for _ in range(n_target):
        mol = Chem.MolFromSmiles(current)
        if mol is None:
            break
        opts = eligible_edits(mol, core, rules, visited)
        if not opts:
            break
        rule, nxt = choose_edit(opts, rng, balance)
        step_rules.append(rule.rule_name)
        step_classes.append(rule.primary_class)
        current = nxt
        visited.add(current)
        step_smiles.append(current)

    if not step_rules:
        return None
    final = Chem.MolFromSmiles(current)
    if final is None:
        return None
    return {
        "pks_smiles": core_canon,
        "aug_smiles": current,
        "source": "mite_forward_grammar",
        "n_added_atoms": final.GetNumHeavyAtoms() - core_heavy,
        "core_heavy_atoms": core_heavy,
        "tailoring_class": "+".join(step_classes),
        "n_tailoring_steps": len(step_rules),
        "step_smiles": json.dumps(step_smiles),
        "step_rules": json.dumps(step_rules),
    }


def iter_smiles(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            smi = line.strip()
            if smi:
                yield smi


def already_done(out_parquet: Path) -> set[str]:
    if not out_parquet.exists():
        return set()
    df = pd.read_parquet(out_parquet, columns=["pks_smiles"])
    return set(df["pks_smiles"].astype(str).tolist())


def augment_library(
    library_path: Path,
    rules: list[CompiledRule],
    out_parquet: Path,
    min_steps: int,
    max_steps: int,
    walks_per_core: int,
    seed: int,
    limit: int | None,
    resume: bool,
    balance: str = "class",
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + shard_index)
    done = already_done(out_parquet) if resume else set()
    rows: list[dict] = []
    n_cores = n_walks = n_skip = 0
    flush_every = 500

    for i, smi in enumerate(iter_smiles(library_path)):
        if shard_count > 1 and (i % shard_count) != shard_index:
            continue
        if limit is not None and n_cores >= limit:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skip += 1
            continue
        core_canon = Chem.MolToSmiles(mol, isomericSmiles=False)
        n_cores += 1
        if core_canon in done:
            continue
        produced = 0
        attempts = 0
        while produced < walks_per_core and attempts < walks_per_core * 20:
            attempts += 1
            walked = one_walk(core_canon, rules, rng, min_steps, max_steps, balance)
            if walked is None:
                continue
            rows.append(walked)
            produced += 1
            n_walks += 1
        done.add(core_canon)

        if len(rows) >= flush_every:
            _flush(rows, out_parquet)
            rows.clear()
            print(
                f"{library_path.name} shard={shard_index}/{shard_count}: "
                f"cores={n_cores} walks={n_walks} skip_parse={n_skip}",
                flush=True,
            )

    if rows:
        _flush(rows, out_parquet)
    return {
        "library": str(library_path),
        "n_cores_seen": n_cores,
        "n_walks": n_walks,
        "n_skip_parse": n_skip,
        "output": str(out_parquet),
        "n_rules_compiled": len(rules),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "balance": balance,
    }


def _flush(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--library-a", type=Path, default=DEFAULT_LIB_A)
    ap.add_argument("--library-b", type=Path, default=DEFAULT_LIB_B)
    ap.add_argument("--which", choices=("a", "b", "both"), default="both")
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--walks-per-core", type=int, default=1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=None, help="Max cores per library (debug)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument(
        "--balance",
        choices=("class", "rule", "uniform"),
        default="class",
        help="Edit sampling: class (default) = equal weight among eligible classes; "
        "rule = equal among eligible rules; uniform = flat over products (OH-heavy)",
    )
    args = ap.parse_args()

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index must be in [0, --shard-count)")

    rules = load_and_compile_rules(args.rules)
    print(
        f"compiled decorate rules: {len(rules)} from {args.rules}; balance={args.balance}",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logic = ROOT / "AUGMENTATION_LOGIC.md"
    if logic.exists():
        (args.out_dir / "AUGMENTATION_LOGIC.md").write_text(
            logic.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    summaries = []
    targets = []
    if args.which in ("a", "both"):
        targets.append(("library_a", args.library_a))
    if args.which in ("b", "both"):
        targets.append(("library_b", args.library_b))

    for name, path in targets:
        shard_suffix = f"_shard{args.shard_index}" if args.shard_count > 1 else ""
        out = args.out_dir / f"{name}_mite_forward_grammar{shard_suffix}.parquet"
        print(f"=== {name} → {out} ===", flush=True)
        summary = augment_library(
            path,
            rules,
            out,
            args.min_steps,
            args.max_steps,
            args.walks_per_core,
            args.seed,
            args.limit,
            args.resume,
            args.balance,
            args.shard_index,
            args.shard_count,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    manifest = args.out_dir / f"manifest_shard{args.shard_index}.json"
    if args.shard_count == 1:
        manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("wrote", manifest, flush=True)


if __name__ == "__main__":
    main()
