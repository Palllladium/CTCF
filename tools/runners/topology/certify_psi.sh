#!/usr/bin/env bash
# Cellwise certificate of the materialized sampler map Psi (normalize-by-(N-1) + align_corners=False).
# Materializes u_Psi = Psi - x from certified Phi flows in float32, then runs the artifact-bound verifier on it.
# The verdict covers the trilinear interpolant of these stored values cell by cell. It does not by itself prove a
# global homeomorphism (the boundary equals the affine A only up to rounding) and does not verify the grid_sample
# kernel; the global statement for the deployed map comes from the zero-boundary Phi certificate and the exact
# affine A. Psi has an approximately affine boundary, so --require-zero-boundary is NOT used here.
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
