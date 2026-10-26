#!/usr/bin/env bash
# Certificate of the materialized sampler map Psi (normalize-by-(N-1) + align_corners=False).
# Materializes u_Psi = Psi - x from certified Phi flows in float32, then checks every saved field on its exact bytes:
# the cellwise Bernstein predicate (utils.cert_exact) and the separable monotone boundary (utils.cert_boundary).
# GLOBAL_HOMEOMORPHISM means both passed: the trilinear interpolant of the stored Psi array is a homeomorphism onto a
# box close to A(Omega). NOT_ESTABLISHED is not a fold. The grid_sample kernel arithmetic is not verified.
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
"${PYBIN}" tools/analysis/materialize_psi.py --flows "$FLOWS" --out "$OUT/flows" --eps "$EPS" --report "$OUT/global_psi.json"
echo ">>> deployed-Psi certificate: $OUT/global_psi.json"
