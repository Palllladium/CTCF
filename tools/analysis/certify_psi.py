"""Check real OASIS/IXI materialized fields against their existing cell certificates and the new boundary rule."""

import argparse
import json
from pathlib import Path

from tools.analysis.materialize_psi import certify_stored, materialize_psi_displacement
from utils.cert_boundary import boundary_report, global_verdict
from utils.field_io import load_flow_npz, save_flow_npz_atomic


def check_collection(root: Path, output: Path, name: str, canonical: str, expected: int) -> dict:
    source = root / "psi_cert" / name
    cells = json.loads((source / "exact_psi.json").read_text(encoding="utf-8"))
    if len(cells) != expected or len({Path(cell["file"]).name for cell in cells}) != expected:
        raise ValueError(f"{name}: expected {expected} distinct certified fields, got {len(cells)}")
    rows = []
    for cell in cells:
        filename = Path(cell["file"]).name
        path = source / "flows" / filename
        try:
            if not path.is_file():
                phi = root / "collar" / canonical / "flows" / filename
                path = output / "fields" / name / filename
                save_flow_npz_atomic(path, materialize_psi_displacement(load_flow_npz(phi)))
            boundary = boundary_report(load_flow_npz(path))
            row = {"file": str(path), **global_verdict(cell, boundary), "cell": cell, "boundary": boundary}
        except (OSError, ValueError, TypeError, KeyError) as error:
            row = {"file": filename, "status": "ERROR", "global_homeomorphism": False, "error": str(error)}
        rows.append(row)
        print(f"[PSI CERT] {name}/{filename} {row['status']}", flush=True)
    passed = [row for row in rows if row["global_homeomorphism"]]
    fresh = None
    if passed:
        worst = min(passed, key=lambda row: row["cell"]["interval_lo_min"])
        fresh = certify_stored(Path(worst["file"]), worst["cell"]["epsilon_fraction"], 100_000)
        if fresh["sha256"] != worst["sha256"]:
            raise ValueError("fresh certification read different field bytes")
        print(f"[PSI CERT FRESH] {name} {fresh['status']}", flush=True)
    return {
        "name": name,
        "expected": expected,
        "rows": rows,
        "fresh": fresh,
        "passed": len(passed) == expected and bool(fresh and fresh["global_homeomorphism"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    collections = []
    for name, canonical, expected in (("oas_w4", "OAS_COLLAR_w4", 19), ("ixi_w4", "IXI_COLLAR_w4", 115)):
        try:
            collections.append(check_collection(args.root, args.out, name, canonical, expected))
        except (OSError, ValueError, TypeError, KeyError) as error:
            collections.append({"name": name, "passed": False, "error": str(error)})
    report = {
        "root": str(args.root.resolve()),
        "passed": all(item["passed"] for item in collections),
        "collections": collections,
        "scope": "stored trilinear interpolants, not GPU kernel arithmetic",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
