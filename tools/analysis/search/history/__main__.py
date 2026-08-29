from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.analysis.search.history.verify import (
    DEFAULT_REPO_ROOT,
    EVIDENCE_GROUPS,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID_PRODUCT,
    EXIT_INVALID_REGISTRY,
    EXIT_MISSING_GIT_OBJECT,
    EXIT_MISSING_PRODUCT,
    EXIT_OK,
    REGISTRY_PATH,
    SCOPE_EXCLUSIONS,
    VERIFICATION_SCOPE,
    InvalidProductError,
    MissingProductError,
    VerifierError,
    assert_plain_directory,
    load_registry,
    parse_strict_json,
    product_by_id,
    resolve_inside_root,
    verify_product,
    verify_relation,
    verify_source_link_targets,
)

# Reported when several distinct registry hints for one product all resolve on disk.
AMBIGUOUS = "AMBIGUOUS"
NOT_FOUND = "NOT_FOUND"
CATEGORY_PRECEDENCE = (
    ("INVALID_REGISTRY", EXIT_INVALID_REGISTRY),
    ("INVALID_PRODUCT", EXIT_INVALID_PRODUCT),
    ("MISSING_GIT_OBJECT", EXIT_MISSING_GIT_OBJECT),
    ("MISSING_PRODUCT", EXIT_MISSING_PRODUCT),
)


def _emit(document: Mapping[str, Any]) -> None:
    json.dump(document, sys.stdout, indent=2, sort_keys=True, ensure_ascii=True)
    sys.stdout.write("\n")


def _failure(error: VerifierError, product_id: str | None = None) -> dict[str, Any]:
    record = {"result": "FAIL", "category": error.category, "check": error.check, "detail": error.detail}
    if product_id is not None:
        record["id"] = product_id
    return record


def _exit_code(categories: Sequence[str]) -> int:
    for name, code in CATEGORY_PRECEDENCE:
        if name in categories:
            return code
    return EXIT_OK


def _resolve_hint(results_root: Path, hint: str) -> Path | None:
    """Resolve one exact hint through the link-rejecting resolver, or report it absent."""
    try:
        directory = resolve_inside_root(
            results_root,
            hint,
            check="verify_known_hint_path",
            error=InvalidProductError,
            expect_directory=True,
        )
    except InvalidProductError as error:
        if "is unreadable at" in error.detail:
            return None
        raise
    resolve_inside_root(directory, "run_manifest.json", check="verify_known_hint_path", error=InvalidProductError)
    return directory


def _locate(registry: Mapping[str, Any], entry: Mapping[str, Any], results_root: Path) -> dict[str, Any]:
    """Resolve a product by its exact registry hints; never search recursively."""
    assert_plain_directory(results_root, check="verify_known_results_root", error=InvalidProductError)
    resolved = [(hint, root) for hint in entry["relative_hints"] if (root := _resolve_hint(results_root, hint))]
    if not resolved:
        return {"status": NOT_FOUND, "candidates": []}
    if len(resolved) > 1:
        return {"status": AMBIGUOUS, "candidates": [hint for hint, _ in resolved]}
    hint, root = resolved[0]
    raw = (root / "run_manifest.json").read_bytes()
    manifest = parse_strict_json(raw, check="verify_known_manifest", error=InvalidProductError)
    declared = manifest.get("run_id") if isinstance(manifest, Mapping) else None
    if declared != entry["run_id"]:
        raise MissingProductError(
            "verify_known_run_id",
            f"{hint} declares run_id {declared!r}, registry expects {entry['run_id']!r}",
        )
    return {"status": "MATCHED", "candidates": [hint], "root": root}


def command_list(args: argparse.Namespace) -> int:
    registry = load_registry(Path(args.registry))
    products = [
        {
            "id": entry["id"],
            "role": entry["role"],
            "gate": entry["gate"],
            "run_id": entry["run_id"],
            "summary": entry["summary"],
            "relative_hints": list(entry["relative_hints"]),
            "manifest_sha256": entry["manifest"]["sha256"],
            "code_git_head": entry["manifest"]["code_git_head"],
            "code_heads": [{"role": head["role"], "git_head": head["git_head"]} for head in entry["code_heads"]],
            "entrypoints": [f"{item['git_head']}:{item['path']}" for item in entry["entrypoints"]],
            "assertion_count": sum(len(entry[group]) for group in EVIDENCE_GROUPS),
        }
        for entry in registry["products"]
    ]
    _emit(
        {
            "command": "list",
            "schema": registry["schema"],
            "freeze_commit": registry["freeze_commit"],
            "canonical_count": sum(1 for entry in products if entry["role"] == "CANONICAL"),
            "supporting_count": sum(1 for entry in products if entry["role"] == "SUPPORTING"),
            "products": products,
            "relations": registry["relations"],
            "external_reference_commits": registry["external_reference_commits"],
            "verification_scope": VERIFICATION_SCOPE,
            "scope_exclusions": list(SCOPE_EXCLUSIONS),
        }
    )
    return EXIT_OK


def command_verify(args: argparse.Namespace) -> int:
    registry = load_registry(Path(args.registry))
    entry = product_by_id(registry, args.id)
    try:
        report = verify_product(
            registry,
            entry["id"],
            Path(args.product_root),
            repo_root=Path(args.repo_root),
        )
    except VerifierError as error:
        _emit({"command": "verify", "id": args.id, **_failure(error)})
        return error.exit_code
    _emit({"command": "verify", **report})
    return EXIT_OK


def _totals(registry: Mapping[str, Any], passed: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    """Report each role separately so a supporting witness never inflates the canonical tally."""
    rows = [entry for entry in passed if entry["role"] == role]
    return {
        "registered": sum(1 for entry in registry["products"] if entry["role"] == role),
        "passed": len(rows),
        "ids": sorted(entry["id"] for entry in rows),
        "index_rows": sum(entry.get("indexed_files", 0) for entry in rows),
        "files": sum(entry.get("total_files", 0) for entry in rows),
        "bytes": sum(entry.get("total_bytes", 0) for entry in rows),
    }


def command_verify_known(args: argparse.Namespace) -> int:
    registry = load_registry(Path(args.registry))
    results_root = Path(args.results_root)
    located: dict[str, Path] = {}
    products: list[dict[str, Any]] = []
    categories: list[str] = []

    for entry in registry["products"]:
        try:
            placement = _locate(registry, entry, results_root)
        except VerifierError as error:
            products.append({"id": entry["id"], "role": entry["role"], **_failure(error)})
            categories.append(error.category)
            continue
        if placement["status"] != "MATCHED":
            # An absent or ambiguously placed registered product is always a failure, flag or not.
            products.append(
                {
                    "id": entry["id"],
                    "role": entry["role"],
                    "result": "FAIL",
                    "category": "MISSING_PRODUCT",
                    "check": "verify_known_placement",
                    "placement": placement["status"],
                    "candidates": placement["candidates"],
                }
            )
            categories.append("MISSING_PRODUCT")
            continue
        root = placement["root"]
        try:
            report = verify_product(
                registry,
                entry["id"],
                root,
                repo_root=Path(args.repo_root),
            )
        except VerifierError as error:
            products.append({"id": entry["id"], "role": entry["role"], "placement": "MATCHED", **_failure(error)})
            categories.append(error.category)
            continue
        located[entry["id"]] = root
        products.append({**report, "placement": "MATCHED"})

    links: list[dict[str, Any]] = []
    for entry in registry["products"]:
        if entry["id"] not in located:
            continue
        try:
            links.extend({"id": entry["id"], **item} for item in verify_source_link_targets(registry, entry, located))
        except VerifierError as error:
            links.append({"id": entry["id"], **_failure(error)})
            categories.append(error.category)

    relations: list[dict[str, Any]] = []
    for relation in registry["relations"]:
        try:
            relations.append(verify_relation(registry, relation, located))
        except VerifierError as error:
            relations.append({"type": relation["type"], **_failure(error)})
            categories.append(error.category)

    passed = [entry for entry in products if entry.get("result") == "PASS"]
    _emit(
        {
            "command": "verify-known",
            "freeze_commit": registry["freeze_commit"],
            "require_all": bool(args.require_all),
            "products": products,
            "source_links": links,
            "relations": relations,
            "canonical": _totals(registry, passed, "CANONICAL"),
            "supporting": _totals(registry, passed, "SUPPORTING"),
            "result": "FAIL" if categories else "PASS",
            "verification_scope": VERIFICATION_SCOPE,
            "scope_exclusions": list(SCOPE_EXCLUSIONS),
        }
    )
    return _exit_code(categories)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.analysis.search.history")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list")
    listing.set_defaults(handler=command_list)

    single = subparsers.add_parser("verify")
    single.add_argument("--id", required=True)
    single.add_argument("--product-root", required=True)
    single.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    single.set_defaults(handler=command_verify)

    known = subparsers.add_parser("verify-known")
    known.add_argument("--results-root", required=True)
    known.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    known.add_argument(
        "--require-all",
        action="store_true",
        help="retained for the pinned CLI surface; an absent registered product always fails regardless",
    )
    known.set_defaults(handler=command_verify_known)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Every outcome, success or failure, leaves exactly one JSON document on stdout."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except VerifierError as error:
        _emit({"command": args.command, **_failure(error)})
        return error.exit_code
    except Exception as exc:
        _emit(
            {
                "command": args.command,
                "result": "FAIL",
                "category": "INTERNAL_ERROR",
                "check": "unexpected_exception",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
