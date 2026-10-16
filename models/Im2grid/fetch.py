"""Download the pinned GPL-3.0 im2grid source (github.com/yihao6/vfa) into the git-ignored vfa_port/ and adapt it.

    python -m models.Im2grid.fetch

Kept unchanged: the encoder, the decoder's matching, positional embeddings and field composition, and the sampling
helpers. Changed: imports; skipped levels become a set; the decoder returns the native local grids (normalized, before
upsampling) instead of full-resolution copies; the attention-weighted coordinates are computed in FP32 under autocast.
Checksums pin the downloaded files and every edit must apply, so a changed upstream fails instead of passing silently.
"""

from __future__ import annotations

import ast
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "07d933d8396d5f9f8d7e3dac74d8c95b3f8e8314"
PORT_VERSION = 2
PORT = Path(__file__).parent / "vfa_port"
SOURCES = {
    "vfa/models/im2grid.py": "dbf538b239476d13ab563968c40f511567e133b67fc90dd73669faa8d0a0b8a1",
    "vfa/utils/utils.py": "41debe04b7c84816e752766ea5dac63fe54588d3c4ea0b79d1050d371bd665bf",
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}
MODEL_EDITS = (
    ("self.skip = skip", "self.skip = set(skip)"),
    ("local_grids = []", "native_grids = []"),
    ("local_grids.append(tensor_grid)", "native_grids.append(tensor_grid)"),
    (
        "                for j in range(len(local_grids)):\n"
        "                    local_grids[j] = nn.functional.interpolate(local_grids[j], size=None,\n"
        "                        scale_factor=2, mode='trilinear', align_corners=True)",
        "",
    ),
    (
        "        for i in range(len(local_grids)):\n            local_grids[i] = (local_grids[i] + 1) / 2 * (dims - 1)",
        "",
    ),
    ("return composed_grids, local_grids, w_keypoints", "return composed_grids, native_grids, w_keypoints"),
    (
        "out = torch.matmul(attention, value)",
        "with torch.autocast(device_type=value.device.type, enabled=False):\n"
        "            out = torch.matmul(attention.float(), value.float())",
    ),
)
SAMPLING_FUNCTIONS = {"identity_grid_like", "keypoints_sampler", "normalize_grid", "grid_sampler"}


def _definitions(source: str, kind, keep) -> str:
    lines = source.splitlines(keepends=True)
    nodes = [n for n in ast.parse(source).body if isinstance(n, kind) and keep(n.name)]
    return "\n\n".join("".join(lines[n.lineno - 1 : n.end_lineno]) for n in nodes)


def prepare_port(model_source: str, utils_source: str) -> dict[str, str]:
    model = "import math\nimport torch\nimport torch.nn as nn\n"
    model += "from .sampling import grid_sampler, identity_grid_like, keypoints_sampler\n\n"
    model += _definitions(model_source, ast.ClassDef, lambda name: name != "Im2grid")
    for old, new in MODEL_EDITS:
        if old not in model:
            raise RuntimeError(f"the pinned im2grid source no longer contains {old[:60]!r}")
        model = model.replace(old, new)
    sampling = "import torch\nimport torch.nn.functional as nnf\n\n"
    sampling += _definitions(utils_source, ast.FunctionDef, SAMPLING_FUNCTIONS.__contains__)
    return {"model.py": model, "sampling.py": sampling, "__init__.py": ""}


def is_ready() -> bool:
    manifest = PORT / "SOURCE.json"
    if not manifest.exists():
        return False
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        saved.get("revision") == REVISION
        and saved.get("port_version") == PORT_VERSION
        and all(
            (PORT / name).is_file() and hashlib.sha256((PORT / name).read_bytes()).hexdigest() == digest
            for name, digest in saved["files"].items()
        )
    )


def main() -> None:
    if is_ready():
        print(f"im2grid source ready: {REVISION}")
        return
    source = {}
    for name, digest in SOURCES.items():
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/yihao6/vfa/{REVISION}/{name}", timeout=60) as r:
            content = r.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"upstream checksum mismatch: {name}")
        source[name] = content.decode("utf-8")
    files = prepare_port(source["vfa/models/im2grid.py"], source["vfa/utils/utils.py"])
    files["LICENSE"] = source["LICENSE"]
    PORT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (PORT / name).write_bytes(content.encode("utf-8"))
    digests = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    manifest = {"revision": REVISION, "port_version": PORT_VERSION, "files": digests}
    (PORT / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"im2grid source downloaded: {REVISION}")


if __name__ == "__main__":
    main()
