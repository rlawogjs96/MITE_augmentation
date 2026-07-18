# Grammar-guided post-PKS augmentation (Library A / B)

**Status:** proposed operator policy for training-data generation  
**Operators:** `mite_rdchiral.tsv` column `forward_SMARTS` (MITE-derived; `tailoring_class` from MITE)  
**Not used:** DORAnet JN tables, deleted `JN_rules.tsv`, MCS-to-target / DORA-XGB feasibility  
**Peel-only (separate):** `retro_SMARTS` in the same TSV may be used later for reverse proposals; this document is about **forward** walks only.

## Goal

Libraries A and B are de novo PKS cores that obey Type I PKS grammar (starter / extenders / KR–DH–ER / TE). They are not MIBiG natural products. Augmentation must **re-enact plausible post-PKS decoration** on those cores—not blind rule firing and not target-chasing expansion.

Forward walks produce tailored molecules; inverting the walks yields PATRO-style reverse trajectories for the peel LSTM.

## Operator set

| Source | Role |
|--------|------|
| `mite_rdchiral.tsv` `forward_SMARTS` | Legal forward edits |
| `tailoring_class` | MITE family label (not inferred from SMARTS) |
| Multi-reactant SMARTS | Project to scaffold component (drop cofactors / unpaired fragments) when applying to a single core |

rdchiral generalization exists for **peeling** coverage. Augmentation uses the stored forward templates in this TSV; eligibility is further constrained by grammar checks below.

## Concrete grammar rules

### 1. Preserve the reduction fingerprint

KR β-OH, DH alkenes, ER methylenes, and unreduced ketones are **assembly outcomes**, not post-PKS accidents.

- Prefer edits that **use** those motifs as acceptors (e.g. O-glycosylation / O-methylation / O-acylation on aliphatic OH; monooxygenation / epoxidation-like chemistry on olefins where the template matches).
- Reject edits that **rewrite** the fingerprint as if it were tailoring (e.g. stripping a KR-like OH, saturating a DH olefin without a justified post-PKS template, NAD-style redox that flips ketone ↔ alcohol outside a decorate step).

### 2. Decoration over backbone rewrite

Allow **additive** tailoring on grammar-legal sites:

- glycosylation, methylation, acetylation / acylation, halogenation, phosphorylation, prenylation (when a handle exists)
- local oxygen insertion consistent with hydroxylation / monooxygenation templates that leave the carbon skeleton intact

Down-rank / **block** classes that typically change PKS carbon-skeleton grammar:

- biaryl bond formation, cyclization / heterocyclization, ring contraction, macrolactam formation
- decarboxylation, reductions / dehydrations / dehydrogenations that alter the assembly redox pattern
- other / catch-all templates that fail the subgraph check below

### 3. Site inventory before sampling

Before choosing a rule, detect handles on the **current** molecule:

| Handle | Typical detection | Eligible decorate families (examples) |
|--------|-------------------|----------------------------------------|
| Aliphatic OH (KR-like) | `[CX4][OX2H]` | Glycosylation, O-methylation, Acetylation, Acylation, Phosphorylation |
| Olefin / enone (DH-like) | carbon–carbon double bond | Monooxygenation / epoxidation-like templates that require an alkene |
| Unreduced ketone | `[#6]=[#8]` ketone pattern | Only templates that decorate without destroying the ketone unless class is explicitly hydroxylation-compatible |
| Aryl | aromatic atoms | Halogenation, aryl hydroxylation / methylation templates |

A rule is eligible only if:

1. its `tailoring_class` is in the decorate allow-list (or passes a conservative exception),
2. at least one required handle for that class is present (when the class implies a handle),
3. `forward_SMARTS` matches and yields a sanitizable product.

### 4. Sequential walks on a fixed grammar core

- Walk length free within bounds (default 1–4 steps).
- Each step stacks decoration on the same assembly product.
- **Core subgraph conserved:** the original core must remain a substructure of every intermediate (MCS / substructure coverage of core heavy atoms). Additive steps increase heavy-atom count; steps that dissolve or rewrite the core fail.
- **Class-balanced sampling (default `--balance class`):** among eligible edits at a step, pick a `tailoring_class` uniformly, then an edit within that class. Avoids hydroxylation drowning rarer decorate families. Alternatives: `--balance rule` or `--balance uniform` (flat over products; OH-heavy).

Single-step and multi-step walks are both emitted so the peel model sees empty and non-empty history.

## What this is not

- Not fixed DORA-style class weights (“glyco = 5”); class balance only equalizes among classes that already match the molecule.
- Not DORA forward expansion ranked by MCS / fingerprint similarity to a target NP.
- Not DORA-XGB reaction-feasibility scoring.
- Not using UniProt IDs on JN rules as proof of Type I PKS post-tailoring.

## Outputs (intended)

Per successful walk:

- `pks_smiles` — library core
- `aug_smiles` — final decorated molecule
- `step_smiles` — core → … → final
- `step_rules` — `rule_name` per step
- `tailoring_class` — `class1+class2+...`
- `source` — `mite_forward_grammar`
- `n_tailoring_steps`, `n_added_atoms`, `core_heavy_atoms`

Compatible with inversion via `patro_data/convert_jn_aug_to_patro.py` (or a thin MITE-named wrapper with the same schema).

## Libraries

- `pks_core_libraries/library_a_SMILES.txt` (~1M cores)
- `pks_core_libraries/library_b_SMILES.txt` (~65k cores)

Full libraries; shard / resume for runtime. No random “pilot sample” as the training definition—sharding is only an engineering batching device.
