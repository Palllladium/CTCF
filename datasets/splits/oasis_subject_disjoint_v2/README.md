# OASIS subject-disjoint split v2

The frozen development split of the 394 OASIS `All` subjects: 294 training subjects and 100
development subjects paired into 50 disjoint pairs. Labels of the development subjects were
never read for training or model selection in Stage 5. The 20 subjects of the Learn2Reg OASIS
validation set (`Test`, the OASIS-19 pairs) are not part of this split.

Copied verbatim from the Stage 5 run `S5_DEVELOPMENT_20260913T203740Z_cf7183ee11d2`
(`data_attestations/`), protocol `CTCF-STAGE5-OASIS-DATA-V2`.

| File | SHA-256 |
|---|---|
| `split_manifest.json` | `ff1672a90b8aa2b70b6b7e696882582324308e015ac83661c9c1b1ca308683d4` |
| `pair_manifest.json` | `6e69b37fedc9f6341a32b655523e2a656f58ded7cbd0023f436fa31bf8cf647a` |

The subject inventory the manifests were derived from stays with that run
(`source_inventory.json`, SHA-256 `e71695aa…2b4d82946`; `data_contract.json`, `85b9da38…de0184ccd`).
The loader that enforced the label barrier lives at the `stage5-final` tag (`datasets/OASIS100.py`).
