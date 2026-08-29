#!/usr/bin/env bash
# Certify the DEPLOYED sampler map Psi directly (align_corners=False + normalize-by-(N-1)), not only the
# canonical Phi. Materializes u_Psi = Psi - x from certified Phi flows, then runs the artifact-bound verifier
# on Psi. This makes the certified object literally the map grid_sample applies, so the reader need not follow
# the affine corollary. Psi has an AFFINE boundary, so --require-zero-boundary is NOT used here.
#
# Cheap: field op + cert only, no model, no training. Point FLOWS at any certified Phi set (e.g. the collar
# OASIS flows). Runs locally (RTX 5070) or on --3.
set -e

PYBIN="${PYBIN:-python}"
FLOWS="${FLOWS:-results/collar/OAS_COLLAR_w4/flows}"   # certified canonical Phi flows
OUT="${OUT:-results/psi_cert/oas_w4}"
EPS="${EPS:-0.001}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

echo "########## Certify deployed Psi | flows=$FLOWS | eps=$EPS ##########"
"${PYBIN}" tools/analysis/materialize_psi.py --flows "$FLOWS" --out "$OUT/flows"
"${PYBIN}" -m utils.cert_exact --flow "$OUT/flows" --eps "$EPS" --report "$OUT/exact_psi.json"
echo ">>> deployed-Psi certificate: $OUT/exact_psi.json"
