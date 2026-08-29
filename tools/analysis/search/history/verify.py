from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

REGISTRY_SCHEMA = "ctcf-search-history-registry-v1"
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.v1.json"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]

VERIFICATION_SCOPE = "COMPACT_PRODUCT_BYTES_AND_RECORDED_PROVENANCE"
SCOPE_EXCLUSIONS = (
    "does not recompute the GPU numerics that produced the run",
    "does not inspect heavy roots that were never packaged into outputs.tsv",
    "does not read checkpoint bytes",
    "does not judge the scientific correctness of the formulas or of the conclusion",
)

EXIT_OK = 0
EXIT_INVALID_PRODUCT = 2
EXIT_MISSING_PRODUCT = 3
EXIT_MISSING_GIT_OBJECT = 4
EXIT_INVALID_REGISTRY = 5
EXIT_INTERNAL_ERROR = 6

OUTPUTS_HEADER = "relative_path\tbytes\tsha256"
EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
GIT_STATUS_FILE = "git_status.txt"
READ_CHUNK_BYTES = 1 << 20

SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_SHA_PATTERN = re.compile(r"\A[0-9a-f]{40}\Z")
PATH_SEGMENT_PATTERN = re.compile(r"\A[A-Za-z0-9._-]+\Z")
CANONICAL_SIZE_PATTERN = re.compile(r"\A(?:0|[1-9][0-9]*)\Z")
WINDOWS_DRIVE_PATTERN = re.compile(r"\A[A-Za-z]:")
RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{index}" for index in range(1, 10)} | {f"LPT{index}" for index in range(1, 10)}
)

ASSERTION_KINDS = frozenset({"file_sha256", "json_equals", "json_equals_file_sha256"})
PRODUCT_ROLES = frozenset({"CANONICAL", "SUPPORTING"})
EVIDENCE_GROUPS = ("assertions", "test_isolation", "source_links")
# Only these may be empty, and only when the entry declares the gap; the rest are always mandatory.
GAPPABLE_GROUPS = frozenset({"test_isolation", "source_links"})
# A registry entry may later carry a diagnostic failed attempt; the schema fixes the vocabulary now.
PRODUCT_STATUSES = frozenset({"COMPLETE", "FAILED"})

# Which entries may lack which evidence, frozen so a gap cannot be opened by editing the registry.
# C0 carries no test-115 field and no upstream product; C1 exploration is the head of its own chain.
FROZEN_EVIDENCE_GAPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "SG-C0": ("test_isolation", "source_links"),
        "SG-C1-EXPLORATION": ("source_links",),
    }
)

# The one payload equivalence the registry carries, frozen down to the file list and its direction.
FROZEN_RELATION_TYPE = "EQUIVALENT_SCIENTIFIC_PAYLOAD_TO"
FROZEN_RELATION_FROM_ID = "SG-C2-PARITY"
FROZEN_RELATION_TO_ID = "SG-C2-SCIENCE"
FROZEN_RELATION_FILES = (
    "per_step.csv",
    "per_branch.csv",
    "trajectory_summary.csv",
    "summary.json",
    "c1_reference.csv",
    "datasets.csv",
    "datasets.tsv",
    "c2_selfcheck.json",
    "transactional_selfcheck.json",
)


class VerifierError(Exception):
    """Base failure carrying the named check that rejected the input."""

    category = "INTERNAL_ERROR"
    exit_code = EXIT_INTERNAL_ERROR

    def __init__(self, check: str, detail: str) -> None:
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail


class InvalidRegistryError(VerifierError):
    category = "INVALID_REGISTRY"
    exit_code = EXIT_INVALID_REGISTRY


class MissingProductError(VerifierError):
    category = "MISSING_PRODUCT"
    exit_code = EXIT_MISSING_PRODUCT


class MissingGitObjectError(VerifierError):
    category = "MISSING_GIT_OBJECT"
    exit_code = EXIT_MISSING_GIT_OBJECT


class InvalidProductError(VerifierError):
    category = "INVALID_PRODUCT"
    exit_code = EXIT_INVALID_PRODUCT


def _duplicate_key_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate object key {key!r}")
        seen.add(key)
    return dict(pairs)


def _constant_guard(name: str) -> Any:
    raise ValueError(f"non-JSON constant {name!r}")


def parse_strict_json(raw: bytes, *, check: str, error: type[VerifierError]) -> Any:
    """Decode UTF-8 and parse JSON, rejecting duplicate keys and NaN/Infinity."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise error(check, f"not valid UTF-8: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_duplicate_key_guard, parse_constant=_constant_guard)
    except ValueError as exc:
        raise error(check, f"strict JSON rejected the document: {exc}") from exc


def resolve_pointer(document: Any, pointer: str, *, check: str, error: type[VerifierError]) -> Any:
    """Resolve an RFC 6901 JSON pointer, failing closed on any missing step."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise InvalidRegistryError(check, f"pointer {pointer!r} does not start with '/'")
    node = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping):
            if token not in node:
                raise error(check, f"pointer {pointer!r} has no member {token!r}")
            node = node[token]
        elif isinstance(node, list):
            if not CANONICAL_SIZE_PATTERN.match(token) or int(token) >= len(node):
                raise error(check, f"pointer {pointer!r} has no index {token!r}")
            node = node[int(token)]
        else:
            raise error(check, f"pointer {pointer!r} descends into a scalar at {token!r}")
    return node


def validate_relative_path(raw: str, *, check: str, error: type[VerifierError]) -> tuple[str, ...]:
    """Accept only a normalised, forward-slash, in-tree relative path."""
    if raw == "":
        raise error(check, "empty relative path")
    if "\x00" in raw:
        raise error(check, f"relative path {raw!r} contains NUL")
    if "\\" in raw:
        raise error(check, f"relative path {raw!r} contains a backslash or is a UNC path")
    if raw.startswith("/"):
        raise error(check, f"relative path {raw!r} is a POSIX absolute path")
    if WINDOWS_DRIVE_PATTERN.match(raw):
        raise error(check, f"relative path {raw!r} carries a Windows drive letter")
    segments = tuple(raw.split("/"))
    for segment in segments:
        if segment in ("", ".", ".."):
            raise error(check, f"relative path {raw!r} is not normalised at segment {segment!r}")
        if not PATH_SEGMENT_PATTERN.match(segment):
            raise error(check, f"relative path {raw!r} has an unsupported segment {segment!r}")
        # Windows silently strips a trailing dot or space, so `foo.` would alias `foo`.
        if segment[-1] in ". ":
            raise error(check, f"relative path {raw!r} ends a segment with a dot or space: {segment!r}")
        if segment.split(".")[0].upper() in RESERVED_DEVICE_NAMES:
            raise error(check, f"relative path {raw!r} uses the reserved device name {segment!r}")
    return segments


def _is_link_like(entry: os.stat_result) -> bool:
    if stat.S_ISLNK(entry.st_mode):
        return True
    attributes = getattr(entry, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def assert_plain_directory(path: Path, *, check: str, error: type[VerifierError]) -> Path:
    """Reject a product root that is itself a symlink, junction or any other reparse point."""
    try:
        entry = os.lstat(path)
    except OSError as exc:
        raise error(check, f"{path} is unreadable: {exc}") from exc
    if _is_link_like(entry):
        raise error(check, f"{path} is a symlink, junction or reparse point")
    if not stat.S_ISDIR(entry.st_mode):
        raise error(check, f"{path} is not a directory")
    return path


def resolve_inside_root(
    root: Path,
    relative: str,
    *,
    check: str,
    error: type[VerifierError],
    expect_directory: bool = False,
) -> Path:
    """Map a validated relative path to a real file or directory, rejecting links and escapes."""
    segments = validate_relative_path(relative, check=check, error=error)
    current = root
    for index, segment in enumerate(segments):
        current = current / segment
        try:
            entry = os.lstat(current)
        except OSError as exc:
            raise error(check, f"{relative!r} is unreadable at {segment!r}: {exc}") from exc
        if _is_link_like(entry):
            raise error(check, f"{relative!r} traverses a symlink or reparse point at {segment!r}")
        is_last = index == len(segments) - 1
        if is_last and expect_directory and not stat.S_ISDIR(entry.st_mode):
            raise error(check, f"{relative!r} is not a directory")
        if is_last and not expect_directory and not stat.S_ISREG(entry.st_mode):
            raise error(check, f"{relative!r} is not a regular file")
        if not is_last and not stat.S_ISDIR(entry.st_mode):
            raise error(check, f"{relative!r} descends into a non-directory at {segment!r}")
    real_root = os.path.realpath(root)
    real_target = os.path.realpath(current)
    if os.path.commonpath([real_root, real_target]) != real_root or real_target == real_root:
        raise error(check, f"{relative!r} resolves outside the product root")
    return current


def hash_regular_file(path: Path, *, check: str, error: type[VerifierError]) -> tuple[int, str]:
    """Stream a file to SHA-256, rejecting a size or mtime change during the read."""
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise error(check, f"{path.name!r} is not a regular file")
            while chunk := handle.read(READ_CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise error(check, f"cannot read {path.name!r}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise error(check, f"{path.name!r} changed size or mtime while it was being read")
    if size != after.st_size:
        raise error(check, f"{path.name!r} yielded {size} bytes but reports {after.st_size}")
    return size, digest.hexdigest()


def parse_outputs_index(raw: bytes, *, check: str) -> dict[str, tuple[int, str]]:
    """Parse outputs.tsv under the exact header, rejecting every malformed row."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidProductError(check, f"outputs.tsv is not valid UTF-8: {exc}") from exc
    if "\r" in text:
        raise InvalidProductError(check, "outputs.tsv contains a carriage return")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise InvalidProductError(check, "outputs.tsv is empty")
    if lines[0] != OUTPUTS_HEADER:
        raise InvalidProductError(check, f"outputs.tsv header is {lines[0]!r}")
    entries: dict[str, tuple[int, str]] = {}
    folded: dict[str, str] = {}
    for number, line in enumerate(lines[1:], start=2):
        if line == OUTPUTS_HEADER:
            raise InvalidProductError(check, f"outputs.tsv repeats the header on line {number}")
        fields = line.split("\t")
        if len(fields) != 3:
            raise InvalidProductError(check, f"outputs.tsv line {number} has {len(fields)} fields")
        relative, raw_size, digest = fields
        validate_relative_path(relative, check=check, error=InvalidProductError)
        if relative in entries:
            raise InvalidProductError(check, f"outputs.tsv repeats path {relative!r} on line {number}")
        key = relative.casefold()
        if key in folded:
            raise InvalidProductError(
                check, f"outputs.tsv path {relative!r} collides with {folded[key]!r} after casefold"
            )
        if not CANONICAL_SIZE_PATTERN.match(raw_size):
            raise InvalidProductError(check, f"outputs.tsv line {number} has a non-canonical size {raw_size!r}")
        if not SHA256_PATTERN.match(digest):
            raise InvalidProductError(check, f"outputs.tsv line {number} has a malformed sha256 {digest!r}")
        entries[relative] = (int(raw_size), digest)
        folded[key] = relative
    return entries


def enumerate_tree(root: Path, *, check: str) -> dict[str, Path]:
    """List every regular file under the product root, rejecting links and odd objects."""
    found: dict[str, Path] = {}
    for directory, subdirectories, filenames in os.walk(root):
        current = Path(directory)
        for name in list(subdirectories):
            if _is_link_like(os.lstat(current / name)):
                raise InvalidProductError(check, f"{current / name} is a symlink or reparse point")
        for name in filenames:
            path = current / name
            entry = os.lstat(path)
            if _is_link_like(entry):
                raise InvalidProductError(check, f"{path} is a symlink or reparse point")
            if not stat.S_ISREG(entry.st_mode):
                raise InvalidProductError(check, f"{path} is not a regular file")
            relative = path.relative_to(root).as_posix()
            validate_relative_path(relative, check=check, error=InvalidProductError)
            found[relative] = path
    return found


def _require(condition: bool, check: str, detail: str, error: type[VerifierError]) -> None:
    if not condition:
        raise error(check, detail)


def _registry_string(node: Mapping[str, Any], key: str, *, where: str) -> str:
    value = node.get(key)
    if not isinstance(value, str) or value == "":
        raise InvalidRegistryError("registry_schema", f"{where}.{key} must be a non-empty string")
    return value


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Read and structurally validate the declarative registry."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InvalidRegistryError("registry_present", f"cannot read {path}: {exc}") from exc
    registry = parse_strict_json(raw, check="registry_json", error=InvalidRegistryError)
    if not isinstance(registry, Mapping):
        raise InvalidRegistryError("registry_schema", "registry root is not an object")
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise InvalidRegistryError("registry_schema", f"schema is {registry.get('schema')!r}")
    freeze = _registry_string(registry, "freeze_commit", where="registry")
    if not GIT_SHA_PATTERN.match(freeze):
        raise InvalidRegistryError("registry_schema", f"freeze_commit {freeze!r} is not a 40-hex lowercase sha")
    products = registry.get("products")
    if not isinstance(products, list) or not products:
        raise InvalidRegistryError("registry_schema", "products must be a non-empty list")
    _validate_products(products)
    declared = {entry["id"] for entry in products}
    _validate_source_link_targets(products, declared)
    _validate_frozen_evidence_gaps(products)
    if "relations" not in registry:
        raise InvalidRegistryError("registry_required_key", "registry is missing 'relations'")
    _validate_relations(registry["relations"], declared)
    _validate_frozen_relations(registry["relations"])
    return dict(registry)


def _validate_frozen_evidence_gaps(products: Sequence[Mapping[str, Any]]) -> None:
    """Pin which entries may lack which evidence, so a gap cannot be opened by editing the registry."""
    unknown = sorted(set(FROZEN_EVIDENCE_GAPS) - {entry["id"] for entry in products})
    if unknown:
        raise InvalidRegistryError("registry_frozen_evidence_gaps", f"the frozen map names absent products {unknown}")
    for entry in products:
        expected = FROZEN_EVIDENCE_GAPS.get(entry["id"], ())
        actual = tuple(entry["evidence_gaps"])
        if actual != expected:
            raise InvalidRegistryError(
                "registry_frozen_evidence_gaps",
                f"{entry['id']} declares evidence_gaps {list(actual)}, the frozen contract allows {list(expected)}",
            )


def _validate_frozen_relations(relations: Sequence[Mapping[str, Any]]) -> None:
    """The payload equivalence is a frozen contract: one relation, one direction, nine named files."""
    if len(relations) != 1:
        raise InvalidRegistryError(
            "registry_frozen_relations",
            f"the registry carries {len(relations)} relations, the contract fixes exactly 1",
        )
    relation = relations[0]
    for field, expected in (
        ("type", FROZEN_RELATION_TYPE),
        ("from_id", FROZEN_RELATION_FROM_ID),
        ("to_id", FROZEN_RELATION_TO_ID),
    ):
        if relation[field] != expected:
            raise InvalidRegistryError(
                "registry_frozen_relations", f"relation.{field} is {relation[field]!r}, the contract fixes {expected!r}"
            )
    files = tuple(relation["files"])
    if files != FROZEN_RELATION_FILES:
        raise InvalidRegistryError(
            "registry_frozen_relations",
            f"relation.files is {list(files)}, the contract fixes {list(FROZEN_RELATION_FILES)}",
        )


def _validate_source_link_targets(products: Sequence[Mapping[str, Any]], declared: set[str]) -> None:
    """Resolve every source_product_id once all ids are known, so a typo cannot pass as an absent source."""
    for entry in products:
        for item in entry["source_links"]:
            source = item["source_product_id"]
            if source not in declared:
                raise InvalidRegistryError(
                    "registry_source_link_id",
                    f"{entry['id']} links to unknown product {source!r}",
                )
            if source == entry["id"]:
                raise InvalidRegistryError("registry_source_link_id", f"{entry['id']} links to itself")


def _validate_products(products: Sequence[Mapping[str, Any]]) -> None:
    ids: set[str] = set()
    run_ids: set[str] = set()
    hints: dict[str, str] = {}
    for entry in products:
        if not isinstance(entry, Mapping):
            raise InvalidRegistryError("registry_schema", "product entry is not an object")
        product_id = _registry_string(entry, "id", where="product")
        if product_id in ids:
            raise InvalidRegistryError("registry_unique_id", f"duplicate product id {product_id!r}")
        ids.add(product_id)
        where = f"product[{product_id}]"
        role = _registry_string(entry, "role", where=where)
        if role not in PRODUCT_ROLES:
            raise InvalidRegistryError("registry_schema", f"{where}.role is {role!r}")
        run_id = _registry_string(entry, "run_id", where=where)
        if run_id in run_ids:
            raise InvalidRegistryError("registry_unique_run_id", f"duplicate run_id {run_id!r}")
        run_ids.add(run_id)
        declared_hints = entry.get("relative_hints")
        if not isinstance(declared_hints, list) or not declared_hints:
            raise InvalidRegistryError("registry_schema", f"{where}.relative_hints must be a non-empty list")
        for hint in declared_hints:
            if not isinstance(hint, str):
                raise InvalidRegistryError("registry_schema", f"{where}.relative_hints holds a non-string")
            validate_relative_path(hint, check="registry_hint_path", error=InvalidRegistryError)
            if hint in hints:
                raise InvalidRegistryError("registry_unique_hint", f"duplicate relative hint {hint!r}")
            hints[hint] = product_id
        gaps = _validate_evidence_gaps(entry, where=where)
        _validate_manifest_block(_required_list_or_object(entry, "manifest", where), where=where)
        _validate_code_heads(_required_group(entry, "code_heads", where, gaps), where=where)
        _validate_entrypoints(_required_group(entry, "entrypoints", where, gaps), where=where)
        _validate_unindexed(_required_group(entry, "unindexed_files", where, gaps), where=where)
        checks: set[str] = set()
        for group in EVIDENCE_GROUPS:
            _validate_assertions(
                _required_group(entry, group, where, gaps),
                where=f"{where}.{group}",
                group=group,
                checks=checks,
            )


def _required_list_or_object(entry: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in entry:
        raise InvalidRegistryError("registry_required_key", f"{where} is missing {key!r}")
    return entry[key]


def _required_group(entry: Mapping[str, Any], key: str, where: str, gaps: frozenset[str]) -> list[Any]:
    """A registry group must be present, and may be empty only where the entry says so."""
    value = _required_list_or_object(entry, key, where)
    if not isinstance(value, list):
        raise InvalidRegistryError("registry_schema", f"{where}.{key} must be a list")
    if not value and key not in gaps:
        raise InvalidRegistryError("registry_required_key", f"{where}.{key} is empty but not declared in evidence_gaps")
    if value and key in gaps:
        raise InvalidRegistryError(
            "registry_required_key", f"{where}.{key} is declared an evidence gap yet is populated"
        )
    return value


def _validate_evidence_gaps(entry: Mapping[str, Any], *, where: str) -> frozenset[str]:
    declared = _required_list_or_object(entry, "evidence_gaps", where)
    if not isinstance(declared, list):
        raise InvalidRegistryError("registry_schema", f"{where}.evidence_gaps must be a list")
    gaps = set()
    for name in declared:
        if name not in GAPPABLE_GROUPS:
            raise InvalidRegistryError("registry_schema", f"{where}.evidence_gaps names {name!r}")
        if name in gaps:
            raise InvalidRegistryError("registry_schema", f"{where}.evidence_gaps repeats {name!r}")
        gaps.add(name)
    return frozenset(gaps)


def _validate_manifest_block(manifest: Any, *, where: str) -> None:
    if not isinstance(manifest, Mapping):
        raise InvalidRegistryError("registry_schema", f"{where}.manifest must be an object")
    digest = _registry_string(manifest, "sha256", where=f"{where}.manifest")
    if not SHA256_PATTERN.match(digest):
        raise InvalidRegistryError("registry_schema", f"{where}.manifest.sha256 is malformed")
    _registry_string(manifest, "schema", where=f"{where}.manifest")
    status = _registry_string(manifest, "status", where=f"{where}.manifest")
    if status not in PRODUCT_STATUSES:
        raise InvalidRegistryError("registry_schema", f"{where}.manifest.status is {status!r}")
    head = _registry_string(manifest, "code_git_head", where=f"{where}.manifest")
    if not GIT_SHA_PATTERN.match(head):
        raise InvalidRegistryError("registry_schema", f"{where}.manifest.code_git_head is malformed")
    if not isinstance(manifest.get("exit_code"), int) or isinstance(manifest.get("exit_code"), bool):
        raise InvalidRegistryError("registry_schema", f"{where}.manifest.exit_code must be an int")
    if manifest.get("tracked_tree_clean_at_start") is not True:
        raise InvalidRegistryError("registry_schema", f"{where}.manifest.tracked_tree_clean_at_start must be true")


def _validate_code_heads(heads: Any, *, where: str) -> None:
    roles: set[str] = set()
    for head in heads:
        if not isinstance(head, Mapping):
            raise InvalidRegistryError("registry_schema", f"{where}.code_heads holds a non-object")
        role = _registry_string(head, "role", where=f"{where}.code_heads")
        if role in roles:
            raise InvalidRegistryError("registry_unique_role", f"{where}.code_heads repeats the role {role!r}")
        roles.add(role)
        value = _registry_string(head, "git_head", where=f"{where}.code_heads")
        if not GIT_SHA_PATTERN.match(value):
            raise InvalidRegistryError("registry_schema", f"{where}.code_heads.git_head {value!r} is malformed")
        path = _registry_string(head, "file", where=f"{where}.code_heads")
        validate_relative_path(path, check="registry_code_head_path", error=InvalidRegistryError)
        _registry_string(head, "pointer", where=f"{where}.code_heads")


def _validate_entrypoints(entrypoints: Any, *, where: str) -> None:
    seen: set[tuple[str, str]] = set()
    for item in entrypoints:
        if not isinstance(item, Mapping):
            raise InvalidRegistryError("registry_schema", f"{where}.entrypoints holds a non-object")
        value = _registry_string(item, "git_head", where=f"{where}.entrypoints")
        if not GIT_SHA_PATTERN.match(value):
            raise InvalidRegistryError("registry_schema", f"{where}.entrypoints.git_head {value!r} is malformed")
        _registry_string(item, "head_role", where=f"{where}.entrypoints")
        path = _registry_string(item, "path", where=f"{where}.entrypoints")
        validate_relative_path(path, check="registry_entrypoint_path", error=InvalidRegistryError)
        if (value, path) in seen:
            raise InvalidRegistryError("registry_unique_entrypoint", f"{where}.entrypoints repeats {value}:{path}")
        seen.add((value, path))


def _validate_unindexed(files: Any, *, where: str) -> None:
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise InvalidRegistryError("registry_schema", f"{where}.unindexed_files holds a non-object")
        path = _registry_string(item, "path", where=f"{where}.unindexed_files")
        validate_relative_path(path, check="registry_unindexed_path", error=InvalidRegistryError)
        if path in seen:
            raise InvalidRegistryError("registry_schema", f"{where}.unindexed_files repeats {path!r}")
        seen.add(path)
        digest = _registry_string(item, "sha256", where=f"{where}.unindexed_files")
        if not SHA256_PATTERN.match(digest):
            raise InvalidRegistryError("registry_schema", f"{where}.unindexed_files.sha256 is malformed")


def _validate_assertions(assertions: Any, *, where: str, group: str, checks: set[str]) -> None:
    for item in assertions:
        if not isinstance(item, Mapping):
            raise InvalidRegistryError("registry_schema", f"{where} holds a non-object")
        name = _registry_string(item, "check", where=where)
        if name in checks:
            raise InvalidRegistryError("registry_unique_check", f"{where} repeats the check name {name!r}")
        checks.add(name)
        kind = _registry_string(item, "kind", where=where)
        if kind not in ASSERTION_KINDS:
            raise InvalidRegistryError("registry_schema", f"{where}.kind is {kind!r}")
        path = _registry_string(item, "file", where=where)
        validate_relative_path(path, check="registry_assertion_path", error=InvalidRegistryError)
        if kind == "file_sha256":
            digest = _registry_string(item, "sha256", where=where)
            if not SHA256_PATTERN.match(digest):
                raise InvalidRegistryError("registry_schema", f"{where}.sha256 is malformed")
        else:
            _registry_string(item, "pointer", where=where)
            if kind == "json_equals" and "value" not in item:
                raise InvalidRegistryError("registry_schema", f"{where} json_equals needs a value")
            if kind == "json_equals_file_sha256":
                target = _registry_string(item, "target_file", where=where)
                validate_relative_path(target, check="registry_assertion_path", error=InvalidRegistryError)
        if group == "source_links":
            _registry_string(item, "source_product_id", where=where)
            artifact = _registry_string(item, "source_artifact", where=where)
            validate_relative_path(artifact, check="registry_assertion_path", error=InvalidRegistryError)


def _validate_relations(relations: Any, ids: set[str]) -> None:
    if not isinstance(relations, list):
        raise InvalidRegistryError("registry_schema", "relations must be a list")
    for item in relations:
        if not isinstance(item, Mapping):
            raise InvalidRegistryError("registry_schema", "relations holds a non-object")
        _registry_string(item, "type", where="relation")
        for side in ("from_id", "to_id"):
            value = _registry_string(item, side, where="relation")
            if value not in ids:
                raise InvalidRegistryError("registry_relation_id", f"relation names unknown product {value!r}")
        if item["from_id"] == item["to_id"]:
            raise InvalidRegistryError("registry_relation_id", f"relation relates {item['from_id']!r} to itself")
        files = item.get("files")
        if not isinstance(files, list) or not files:
            raise InvalidRegistryError("registry_schema", "relation.files must be a non-empty list")
        seen: set[str] = set()
        for name in files:
            if not isinstance(name, str):
                raise InvalidRegistryError("registry_schema", "relation.files holds a non-string")
            validate_relative_path(name, check="registry_relation_path", error=InvalidRegistryError)
            if name in seen:
                raise InvalidRegistryError("registry_unique_relation_file", f"relation.files repeats {name!r}")
            seen.add(name)


def product_by_id(registry: Mapping[str, Any], product_id: str) -> Mapping[str, Any]:
    for entry in registry["products"]:
        if entry["id"] == product_id:
            return entry
    raise InvalidRegistryError("registry_lookup", f"unknown product id {product_id!r}")


class _GitRunner:
    """Read-only git access pinned to explicit revisions, never to the current HEAD."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        environment = dict(os.environ)
        for inherited in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
            environment.pop(inherited, None)
        environment["GIT_NO_REPLACE_OBJECTS"] = "1"
        environment["GIT_TERMINAL_PROMPT"] = "0"
        # No promisor fetch may reach the network, and no read may take a lock in the user's repository.
        environment["GIT_NO_LAZY_FETCH"] = "1"
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        self.environment = environment

    def run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = ["git", "-C", str(self.repo_root), *arguments]
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=self.environment,
                shell=False,
                check=False,
            )
        except OSError as exc:
            raise MissingGitObjectError("git_available", f"cannot run git: {exc}") from exc

    def object_type(self, revision: str) -> str | None:
        completed = self.run(["cat-file", "-t", revision])
        return completed.stdout.strip() if completed.returncode == 0 else None

    def is_ancestor(self, candidate: str, descendant: str) -> bool:
        return self.run(["merge-base", "--is-ancestor", candidate, descendant]).returncode == 0


def _read_product_bytes(root: Path, relative: str, *, check: str, error: type[VerifierError]) -> bytes:
    path = resolve_inside_root(root, relative, check=check, error=error)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise error(check, f"cannot read {relative!r}: {exc}") from exc


def _referenced_json_files(entry: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for head in entry["code_heads"]:
        names.append(head["file"])
    for group in ("assertions", "test_isolation", "source_links"):
        for item in entry[group]:
            if item["kind"] != "file_sha256":
                names.append(item["file"])
    return sorted({name for name in names if name.endswith(".json")})


def _evaluate_assertion(
    root: Path,
    item: Mapping[str, Any],
    documents: Mapping[str, Any],
) -> None:
    check = item["check"]
    if item["kind"] == "file_sha256":
        path = resolve_inside_root(root, item["file"], check=check, error=InvalidProductError)
        _, digest = hash_regular_file(path, check=check, error=InvalidProductError)
        _require(digest == item["sha256"], check, f"{item['file']} hashes to {digest}", InvalidProductError)
        return
    document = documents[item["file"]]
    actual = resolve_pointer(document, item["pointer"], check=check, error=InvalidProductError)
    if item["kind"] == "json_equals":
        expected = item["value"]
        matches = actual == expected and isinstance(actual, type(expected)) and isinstance(expected, type(actual))
        _require(
            matches, check, f"{item['file']}{item['pointer']} is {actual!r}, expected {expected!r}", InvalidProductError
        )
        return
    target = resolve_inside_root(root, item["target_file"], check=check, error=InvalidProductError)
    _, digest = hash_regular_file(target, check=check, error=InvalidProductError)
    _require(
        actual == digest,
        check,
        f"{item['file']}{item['pointer']} is {actual!r} but {item['target_file']} hashes to {digest}",
        InvalidProductError,
    )


def _verify_run_manifest(root: Path, entry: Mapping[str, Any]) -> tuple[Mapping[str, Any], str, Mapping[str, Any]]:
    """Check run_manifest.json against its registry declaration; return it, its digest and its files block."""
    declaration = entry["manifest"]
    if not (root / "run_manifest.json").is_file():
        raise MissingProductError("run_manifest_present", f"{root / 'run_manifest.json'} is absent")
    manifest_bytes = _read_product_bytes(
        root, "run_manifest.json", check="run_manifest_read", error=MissingProductError
    )
    manifest = parse_strict_json(manifest_bytes, check="run_manifest_json", error=InvalidProductError)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    _require(
        manifest_digest == declaration["sha256"],
        "run_manifest_sha256",
        f"run_manifest.json hashes to {manifest_digest}",
        InvalidProductError,
    )
    _require(
        isinstance(manifest, Mapping), "run_manifest_object", "run_manifest.json is not an object", InvalidProductError
    )
    _require(
        manifest.get("schema") == declaration["schema"],
        "manifest_schema",
        f"schema is {manifest.get('schema')!r}",
        InvalidProductError,
    )
    _require(
        manifest.get("run_id") == entry["run_id"],
        "manifest_run_id",
        f"run_id is {manifest.get('run_id')!r}",
        InvalidProductError,
    )
    _require(
        manifest.get("status") == declaration["status"],
        "manifest_status",
        f"status is {manifest.get('status')!r}",
        InvalidProductError,
    )
    _require(
        # `0.0 == 0` and `True == 1` in Python, so the type must be checked before the value.
        type(manifest.get("exit_code")) is int and manifest["exit_code"] == declaration["exit_code"],
        "manifest_exit_code",
        f"exit_code is {manifest.get('exit_code')!r}",
        InvalidProductError,
    )
    code = manifest.get("code")
    _require(isinstance(code, Mapping), "manifest_code_block", "manifest has no code object", InvalidProductError)
    _require(
        code.get("git_head") == declaration["code_git_head"],
        "manifest_code_git_head",
        f"code.git_head is {code.get('git_head')!r}",
        InvalidProductError,
    )
    _require(
        code.get("tracked_tree_clean_at_start") is True,
        "manifest_tracked_tree_clean",
        f"tracked_tree_clean_at_start is {code.get('tracked_tree_clean_at_start')!r}",
        InvalidProductError,
    )
    files_block = manifest.get("files")
    _require(
        isinstance(files_block, Mapping), "manifest_files_block", "manifest has no files object", InvalidProductError
    )
    return manifest, manifest_digest, files_block


def _verify_outputs_index(root: Path, files_block: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    """Check outputs.tsv against the manifest and demand an empty, indexed git_status.txt."""
    if not (root / "outputs.tsv").is_file():
        raise MissingProductError("outputs_index_present", f"{root / 'outputs.tsv'} is absent")
    index_bytes = _read_product_bytes(root, "outputs.tsv", check="outputs_index_read", error=MissingProductError)
    index = parse_outputs_index(index_bytes, check="outputs_index_format")
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    _require(
        index_digest == files_block.get("outputs_sha256"),
        "outputs_index_sha256",
        f"outputs.tsv hashes to {index_digest} but the manifest records {files_block.get('outputs_sha256')!r}",
        InvalidProductError,
    )
    _require(GIT_STATUS_FILE in index, "git_status_indexed", "git_status.txt is not indexed", InvalidProductError)
    status_size, status_digest = index[GIT_STATUS_FILE]
    _require(
        status_size == 0 and status_digest == EMPTY_FILE_SHA256,
        "git_status_empty",
        f"git_status.txt is indexed as {status_size} bytes / {status_digest}",
        InvalidProductError,
    )
    _require(
        files_block.get("git_status_sha256") == EMPTY_FILE_SHA256,
        "manifest_git_status_sha256",
        f"manifest records git_status_sha256 {files_block.get('git_status_sha256')!r}",
        InvalidProductError,
    )
    return index


def _verify_tree_closure(
    root: Path,
    index: Mapping[str, tuple[int, str]],
    unindexed: Mapping[str, str],
) -> dict[str, Path]:
    """Demand the tree hold exactly the declared files and no two that collide when casefolded."""
    present = enumerate_tree(root, check="tree_closure")
    expected_names = set(index) | set(unindexed)
    missing = sorted(expected_names - set(present))
    _require(not missing, "tree_closure_missing", f"absent files: {missing[:8]}", InvalidProductError)
    extra = sorted(set(present) - expected_names)
    _require(not extra, "tree_closure_extra", f"unregistered files: {extra[:8]}", InvalidProductError)
    folded: dict[str, str] = {}
    # On a case-insensitive filesystem two declared names would silently become one file.
    for name in sorted(present):
        key = name.casefold()
        if key in folded:
            raise InvalidProductError(
                "tree_casefold_collision", f"{name!r} collides with {folded[key]!r} after casefold"
            )
        folded[key] = name
    return present


def verify_product(
    registry: Mapping[str, Any],
    product_id: str,
    product_root: Path,
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Verify one product tree against its registry entry; raise on the first failing named check.

    The git provenance pass is unconditional: there is no bytes-only mode, so a returned
    report always carries result=PASS together with git.checked=true.
    """
    entry = product_by_id(registry, product_id)
    root = Path(product_root)
    if not root.is_dir():
        raise MissingProductError("product_root", f"{root} is not a directory")
    assert_plain_directory(root, check="product_root_is_plain", error=InvalidProductError)

    _manifest, manifest_digest, files_block = _verify_run_manifest(root, entry)
    index = _verify_outputs_index(root, files_block)
    unindexed = {item["path"]: item["sha256"] for item in entry["unindexed_files"]}
    present = _verify_tree_closure(root, index, unindexed)

    unindexed_bytes = 0
    for name in sorted(unindexed):
        path = resolve_inside_root(root, name, check="unindexed_file_sha256", error=InvalidProductError)
        size, digest = hash_regular_file(path, check="unindexed_file_sha256", error=InvalidProductError)
        _require(digest == unindexed[name], "unindexed_file_sha256", f"{name} hashes to {digest}", InvalidProductError)
        unindexed_bytes += size

    documents: dict[str, Any] = {}
    for name in _referenced_json_files(entry):
        raw = _read_product_bytes(root, name, check="referenced_json_strict", error=InvalidProductError)
        documents[name] = parse_strict_json(raw, check="referenced_json_strict", error=InvalidProductError)

    for head in entry["code_heads"]:
        actual = resolve_pointer(documents[head["file"]], head["pointer"], check="code_head", error=InvalidProductError)
        _require(
            actual == head["git_head"],
            f"code_head_{head['role'].lower()}",
            f"{head['file']}{head['pointer']} is {actual!r}, expected {head['git_head']}",
            InvalidProductError,
        )

    for group in ("assertions", "test_isolation", "source_links"):
        for item in entry[group]:
            _evaluate_assertion(root, item, documents)

    scanned_bytes = 0
    for name in sorted(index):
        expected_size, expected_digest = index[name]
        path = resolve_inside_root(root, name, check="indexed_file_readable", error=InvalidProductError)
        size, digest = hash_regular_file(path, check="indexed_file_sha256", error=InvalidProductError)
        _require(
            size == expected_size,
            "indexed_file_bytes",
            f"{name} is {size} bytes, indexed as {expected_size}",
            InvalidProductError,
        )
        _require(digest == expected_digest, "indexed_file_sha256", f"{name} hashes to {digest}", InvalidProductError)
        scanned_bytes += size

    git_report = _verify_git_provenance(registry, entry, repo_root)

    return {
        "id": entry["id"],
        "role": entry["role"],
        "gate": entry["gate"],
        "run_id": entry["run_id"],
        "result": "PASS",
        "product_root_hint": entry["relative_hints"][0],
        "manifest_sha256": manifest_digest,
        "code_git_head": entry["manifest"]["code_git_head"],
        "indexed_files": len(index),
        "unindexed_files": len(unindexed),
        "total_files": len(present),
        "indexed_bytes": scanned_bytes,
        "total_bytes": scanned_bytes + unindexed_bytes,
        "code_heads": [{"role": head["role"], "git_head": head["git_head"]} for head in entry["code_heads"]],
        "assertions": sum(len(entry[group]) for group in EVIDENCE_GROUPS),
        "git": git_report,
        "verification_scope": VERIFICATION_SCOPE,
        "scope_exclusions": list(SCOPE_EXCLUSIONS),
    }


def _verify_git_provenance(
    registry: Mapping[str, Any],
    entry: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    runner = _GitRunner(repo_root)
    freeze = registry["freeze_commit"]
    if runner.object_type(freeze) != "commit":
        raise MissingGitObjectError("git_freeze_commit", f"{freeze} is not a commit in {repo_root}")
    commits = {entry["manifest"]["code_git_head"]}
    commits.update(head["git_head"] for head in entry["code_heads"])
    commits.update(item["git_head"] for item in entry["entrypoints"])
    for commit in sorted(commits):
        if runner.object_type(commit) != "commit":
            raise MissingGitObjectError("git_commit_present", f"{commit} is not a commit in {repo_root}")
        if not runner.is_ancestor(commit, freeze):
            raise MissingGitObjectError("git_commit_ancestor", f"{commit} is not an ancestor of {freeze}")
    for item in entry["entrypoints"]:
        revision = f"{item['git_head']}:{item['path']}"
        if runner.object_type(revision) != "blob":
            raise MissingGitObjectError("git_entrypoint_blob", f"{revision} is not a blob")
    return {
        "checked": True,
        "repo_root_is_default": Path(repo_root) == DEFAULT_REPO_ROOT,
        "freeze_commit": freeze,
        "commits": sorted(commits),
        "entrypoints": [f"{item['git_head']}:{item['path']}" for item in entry["entrypoints"]],
    }


def verify_relation(
    registry: Mapping[str, Any], relation: Mapping[str, Any], roots: Mapping[str, Path]
) -> dict[str, Any]:
    """Confirm a declared cross-product payload equivalence over both real trees."""
    check = f"relation_{relation['type'].lower()}"
    left, right = relation["from_id"], relation["to_id"]
    for side in (left, right):
        if side not in roots:
            raise MissingProductError(check, f"{relation['type']} needs {side}, which was not located")
    compared = []
    for name in relation["files"]:
        digests = []
        for side in (left, right):
            path = resolve_inside_root(roots[side], name, check=check, error=InvalidProductError)
            digests.append(hash_regular_file(path, check=check, error=InvalidProductError)[1])
        _require(
            digests[0] == digests[1],
            check,
            f"{name} differs: {left}={digests[0]} {right}={digests[1]}",
            InvalidProductError,
        )
        compared.append({"file": name, "sha256": digests[0]})
    return {
        "type": relation["type"],
        "from_id": left,
        "to_id": right,
        "result": "PASS",
        "files": compared,
        "note": relation.get("note", ""),
    }


def verify_source_link_targets(
    registry: Mapping[str, Any],
    entry: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Re-derive each declared source hash from the source tree itself, when it is available."""
    resolved: list[dict[str, Any]] = []
    for item in entry["source_links"]:
        source_id = item["source_product_id"]
        if source_id not in roots:
            resolved.append({"check": item["check"], "source_product_id": source_id, "result": "SOURCE_ABSENT"})
            continue
        artifact = item["source_artifact"]
        path = resolve_inside_root(roots[source_id], artifact, check=item["check"], error=InvalidProductError)
        _, digest = hash_regular_file(path, check=item["check"], error=InvalidProductError)
        expected = item.get("value") if item["kind"] == "json_equals" else None
        _require(
            expected == digest,
            item["check"],
            f"{source_id}:{artifact} hashes to {digest} but the link pins {expected!r}",
            InvalidProductError,
        )
        resolved.append(
            {"check": item["check"], "source_product_id": source_id, "source_artifact": artifact, "result": "PASS"}
        )
    return resolved
