"""Extract rdchiral templates directly from MITE reactionSMARTS (not examples)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

from rdkit import Chem
from rdkit.Chem import rdChemReactions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rdchiral"))
from rdchiral.template_extractor import extract_from_reaction

DATA = Path(__file__).resolve().parent / "mite_data" / "data"
OUT = Path(__file__).resolve().parent / "mite_rdchiral_from_smarts.jsonl"
OUT_SUMMARY = Path(__file__).resolve().parent / "mite_rdchiral_from_smarts_summary.json"


def smarts_side_to_mapped_mol(tmpl):
    """Turn a reaction SMARTS template into a concrete mapped mol when possible."""
    try:
        smi = Chem.MolToSmiles(tmpl)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        match = m.GetSubstructMatch(tmpl)
        if match and len(match) == tmpl.GetNumAtoms():
            for t_idx, m_idx in enumerate(match):
                m.GetAtomWithIdx(m_idx).SetAtomMapNum(
                    tmpl.GetAtomWithIdx(t_idx).GetAtomMapNum()
                )
            return m
        if m.GetNumAtoms() != tmpl.GetNumAtoms():
            return None
        for i, a in enumerate(m.GetAtoms()):
            a.SetAtomMapNum(tmpl.GetAtomWithIdx(i).GetAtomMapNum())
        return m
    except Exception:
        return None


def main() -> None:
    files = sorted(DATA.glob("MITE*.json"))
    file_status = Counter()
    by_status = Counter()
    direct_ok = []
    n_rxn_smarts = 0
    n_active_files = 0

    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        file_status[d.get("status", "?")] += 1
        if d.get("status") != "active":
            continue
        n_active_files += 1
        for ri, rxn in enumerate(d.get("reactions") or [], 1):
            smarts = rxn.get("reactionSMARTS") or ""
            if not smarts or ">>" not in smarts:
                continue
            n_rxn_smarts += 1
            try:
                chem = rdChemReactions.ReactionFromSmarts(smarts)
            except Exception:
                by_status["smarts_parse_fail"] += 1
                continue
            if chem.GetNumReactantTemplates() < 1 or chem.GetNumProductTemplates() < 1:
                by_status["bad_stoich"] += 1
                continue
            r_mols, p_mols = [], []
            good = True
            for i in range(chem.GetNumReactantTemplates()):
                m = smarts_side_to_mapped_mol(chem.GetReactantTemplate(i))
                if m is None:
                    good = False
                    break
                r_mols.append(m)
            if good:
                for i in range(chem.GetNumProductTemplates()):
                    m = smarts_side_to_mapped_mol(chem.GetProductTemplate(i))
                    if m is None:
                        good = False
                        break
                    p_mols.append(m)
            if not good:
                by_status["template_to_mol_fail"] += 1
                continue
            reaction = {
                "_id": f"{d['accession']}.r{ri}",
                "reactants": ".".join(Chem.MolToSmiles(m) for m in r_mols),
                "products": ".".join(Chem.MolToSmiles(m) for m in p_mols),
            }
            try:
                tmpl = extract_from_reaction(reaction)
            except Exception:
                by_status["extract_exception"] += 1
                continue
            if not tmpl or "reaction_smarts" not in tmpl:
                by_status["extract_empty"] += 1
                continue
            by_status["ok"] += 1
            direct_ok.append(
                {
                    "accession": d["accession"],
                    "reaction": ri,
                    "terms": rxn.get("tailoring") or [],
                    "enzyme": d.get("enzyme", {}).get("name", ""),
                    "original_smarts_len": len(smarts),
                    "general_smarts_len": len(tmpl["reaction_smarts"]),
                    "retro_smarts": tmpl["reaction_smarts"],
                    "forward_smarts": f"{tmpl['reactants']}>>{tmpl['products']}",
                }
            )

    print("json files", len(files), dict(file_status))
    print("active files", n_active_files)
    print("reactionSMARTS (active)", n_rxn_smarts)
    print("direct status", dict(by_status))
    print("ok", by_status["ok"], "unique", len({r["retro_smarts"] for r in direct_ok}))
    if direct_ok:
        print(
            "median orig",
            median(r["original_smarts_len"] for r in direct_ok),
            "gen",
            median(r["general_smarts_len"] for r in direct_ok),
        )
        for r in direct_ok:
            if r["accession"] == "MITE0000001":
                print("AviG2", r["retro_smarts"])
                break

    with OUT.open("w", encoding="utf-8") as f:
        for r in direct_ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "n_json_files": len(files),
                "file_status": dict(file_status),
                "n_active_files": n_active_files,
                "n_reaction_smarts": n_rxn_smarts,
                "extract_status": dict(by_status),
                "n_ok": by_status["ok"],
                "n_unique_retro": len({r["retro_smarts"] for r in direct_ok}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
