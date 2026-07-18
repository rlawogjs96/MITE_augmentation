# MITE_augmentation

Grammar-guided forward decoration of PKS core libraries (A/B) using `mite_rdchiral.tsv` `forward_SMARTS`.

Policy: see [`AUGMENTATION_LOGIC.md`](AUGMENTATION_LOGIC.md).

## Contents

| Path | Role |
|------|------|
| `data/library_a_SMILES.txt` | ~1M PKS cores |
| `data/library_b_SMILES.txt` | ~65k PKS cores |
| `mite_rdchiral.tsv` | MITE rdchiral rules (forward + retro) |
| `augment_mite_grammar.py` | Augmenter (CPU; RDKit MCS / RunReactants) |

## Setup (server)

```bash
git clone https://github.com/rlawogjs96/MITE_augmentation.git
cd MITE_augmentation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Full A+B (single process, class-balanced sampling by default):

```bash
python augment_mite_grammar.py --which both --resume --balance class
```

Lib B only (faster smoke):

```bash
python augment_mite_grammar.py --which b --balance class
```

Multi-CPU shards (example: 8 workers on Lib A):

```bash
for i in $(seq 0 7); do
  python augment_mite_grammar.py --which a --balance class --shard-index $i --shard-count 8 \
    --out-dir augmentations/mite_forward_grammar &
done
wait
```

`--balance class` (default) picks uniformly among eligible decorate classes at each step, then among products in that class. Use `--balance uniform` only if you want the old OH-heavy behavior.

Outputs land in `augmentations/mite_forward_grammar/*.parquet`.

**Note:** GPU does not speed this up (RDKit MCS / reaction application are CPU-bound). Prefer more CPU cores + shards.
