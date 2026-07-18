"""Extract rdchiral generalized reaction templates from MITE entries.

Transfers atom maps from each entry's reactionSMARTS onto substrate/product
example SMILES, then runs rdchiral.template_extractor.extract_from_reaction
(radius-1 changed-atom neighborhoods) to produce generalizable retro SMARTS.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rdchiral"))
from rdchiral.template_extractor import extract_from_reaction

DATA = Path(__file__).resolve().parent / "mite_data" / "data"
OUT = Path(__file__).resolve().parent / "mite_rdchiral_templates.jsonl"
OUT_SUMMARY = Path(__file__).resolve().parent / "mite_rdchiral_templates_by_term.json"
OUT_FAIL = Path(__file__).resolve().parent / "mite_rdchiral_failures.jsonl"


def mol_with_maps_from_match(mol, template):
    """Copy atom map numbers from SMARTS template onto mol via substructure match."""
    match = mol.GetSubstructMatch(template)
    if not match or len(match) != template.GetNumAtoms():
        return None
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    for t_idx, m_idx in enumerate(match):
        mapnum = template.GetAtomWithIdx(t_idx).GetAtomMapNum()
        if mapnum:
            m.GetAtomWithIdx(m_idx).SetAtomMapNum(mapnum)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def mapped_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True)


def process_entry(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("status") != "active":
        return []
    acc = d["accession"]
    enzyme = d.get("enzyme", {}).get("name", "")
    results = []
    for ri, rxn in enumerate(d.get("reactions") or [], 1):
        terms = rxn.get("tailoring") or []
        smarts = rxn.get("reactionSMARTS") or ""
        if not smarts or ">>" not in smarts:
            continue
        try:
            chem_rxn = rdChemReactions.ReactionFromSmarts(smarts)
        except Exception as e:
            results.append(
                {
                    "accession": acc,
                    "reaction": ri,
                    "status": "smarts_parse_fail",
                    "error": str(e),
                    "terms": terms,
                }
            )
            continue
        if chem_rxn.GetNumReactantTemplates() < 1 or chem_rxn.GetNumProductTemplates() < 1:
            results.append(
                {
                    "accession": acc,
                    "reaction": ri,
                    "status": "bad_rxn_stoich",
                    "terms": terms,
                }
            )
            continue
        r_tmpl = chem_rxn.GetReactantTemplate(0)
        p_tmpl = chem_rxn.GetProductTemplate(0)
        examples = rxn.get("reactions") or []
        if not examples:
            results.append(
                {
                    "accession": acc,
                    "reaction": ri,
                    "status": "no_example",
                    "terms": terms,
                }
            )
            continue
        for ei, ex in enumerate(examples, 1):
            sub = ex.get("substrate")
            prods = ex.get("products") or []
            if not sub or not prods:
                continue
            prod = prods[0]
            sm = Chem.MolFromSmiles(sub)
            pm = Chem.MolFromSmiles(prod)
            if sm is None or pm is None:
                results.append(
                    {
                        "accession": acc,
                        "reaction": ri,
                        "example": ei,
                        "status": "smiles_parse_fail",
                        "terms": terms,
                        "enzyme": enzyme,
                    }
                )
                continue
            sm_m = mol_with_maps_from_match(sm, r_tmpl)
            pm_m = mol_with_maps_from_match(pm, p_tmpl)
            if sm_m is None or pm_m is None:
                results.append(
                    {
                        "accession": acc,
                        "reaction": ri,
                        "example": ei,
                        "status": "map_transfer_fail",
                        "terms": terms,
                        "enzyme": enzyme,
                        "sub_atoms": sm.GetNumAtoms(),
                        "r_tmpl_atoms": r_tmpl.GetNumAtoms(),
                        "prod_atoms": pm.GetNumAtoms(),
                        "p_tmpl_atoms": p_tmpl.GetNumAtoms(),
                    }
                )
                continue
            r_maps = {a.GetAtomMapNum() for a in sm_m.GetAtoms() if a.GetAtomMapNum()}
            p_maps = {a.GetAtomMapNum() for a in pm_m.GetAtoms() if a.GetAtomMapNum()}
            if not (r_maps & p_maps):
                results.append(
                    {
                        "accession": acc,
                        "reaction": ri,
                        "example": ei,
                        "status": "no_shared_maps",
                        "terms": terms,
                    }
                )
                continue
            reaction = {
                "_id": f"{acc}.r{ri}.e{ei}",
                "reactants": mapped_smiles(sm_m),
                "products": mapped_smiles(pm_m),
            }
            try:
                tmpl = extract_from_reaction(reaction)
            except Exception as e:
                results.append(
                    {
                        "accession": acc,
                        "reaction": ri,
                        "example": ei,
                        "status": "extract_exception",
                        "error": str(e),
                        "terms": terms,
                        "enzyme": enzyme,
                    }
                )
                continue
            if not tmpl or "reaction_smarts" not in tmpl:
                results.append(
                    {
                        "accession": acc,
                        "reaction": ri,
                        "example": ei,
                        "status": "extract_empty",
                        "terms": terms,
                        "enzyme": enzyme,
                    }
                )
                continue
            results.append(
                {
                    "accession": acc,
                    "reaction": ri,
                    "example": ei,
                    "status": "ok",
                    "terms": terms,
                    "enzyme": enzyme,
                    "forward_smarts": f"{tmpl['reactants']}>>{tmpl['products']}",
                    "retro_smarts": tmpl["reaction_smarts"],
                    "intra_only": tmpl.get("intra_only"),
                    "necessary_reagent": tmpl.get("necessary_reagent") or "",
                    "original_smarts_len": len(smarts),
                    "general_smarts_len": len(tmpl["reaction_smarts"]),
                }
            )
    return results


def main() -> None:
    all_rows: list[dict] = []
    for fp in sorted(DATA.glob("MITE*.json")):
        all_rows.extend(process_entry(fp))

    status_c = Counter(r["status"] for r in all_rows)
    ok = [r for r in all_rows if r["status"] == "ok"]
    fails = [r for r in all_rows if r["status"] != "ok"]

    print("STATUS", dict(status_c))
    print("ok", len(ok), "total attempts", len(all_rows))

    tmpl_c = Counter(r["retro_smarts"] for r in ok)
    print("unique retro templates", len(tmpl_c))
    print("top 15 by frequency:")
    for s, c in tmpl_c.most_common(15):
        print(f"  n={c:3d}  len={len(s):4d}  {s[:140]}")

    by_term: dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        for t in r["terms"]:
            by_term[t][r["retro_smarts"]] += 1

    if ok:
        orig = sorted(r["original_smarts_len"] for r in ok)
        gen = sorted(r["general_smarts_len"] for r in ok)
        print(
            "orig SMARTS len median",
            orig[len(orig) // 2],
            "gen median",
            gen[len(gen) // 2],
        )
        print(
            "compression median ratio",
            round(orig[len(orig) // 2] / max(1, gen[len(gen) // 2]), 2),
        )

    with OUT.open("w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_FAIL.open("w", encoding="utf-8") as f:
        for r in fails:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {}
    for term, ctr in sorted(by_term.items(), key=lambda x: -sum(x[1].values())):
        summary[term] = {
            "n_extractions": sum(ctr.values()),
            "n_unique_templates": len(ctr),
            "top_templates": [
                {"retro_smarts": s, "count": c, "len": len(s)}
                for s, c in ctr.most_common(8)
            ],
        }
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "status_counts": dict(status_c),
                "n_ok": len(ok),
                "n_unique_retro": len(tmpl_c),
                "by_term": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", OUT)
    print("wrote", OUT_SUMMARY)
    print("wrote", OUT_FAIL)

    for term in ["Halogenation", "Methylation", "Hydroxylation", "Glycosylation"]:
        print(f"\n=== {term} top templates ===")
        for item in summary.get(term, {}).get("top_templates", [])[:5]:
            print(f"  n={item['count']} len={item['len']}")
            print(f"    {item['retro_smarts'][:220]}")


if __name__ == "__main__":
    main()
