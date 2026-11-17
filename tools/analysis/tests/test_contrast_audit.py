import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from datasets.IXI import read_ixi_manifest
from experiments import contrast_audit as audit, contrast_learning
from experiments.contrast_learning import options_for
from experiments.train_matching import teacher_subjects
from models.Matching import FeatureEncoder, voxel_grid
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.tests.test_contrast_learning import cohort as cohort
from utils.losses import stable_ncc_map


@pytest.fixture(autouse=True)
def cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_oracle_has_a_fixed_four_voxel_transition_and_does_not_use_moving_labels():
    labels = torch.zeros(1, 1, 13, 14, 15, dtype=torch.long)
    labels[..., 6, :, :] = 2
    alpha = audit.region_weight(labels)
    assert alpha.shape == labels.shape
    assert torch.count_nonzero(alpha[..., 6, :, :]) == 0
    assert alpha[0, 0, 7, 7, 7] == pytest.approx(0.15625)
    assert alpha[0, 0, 8, 7, 7] == pytest.approx(0.5)
    assert alpha[0, 0, 9, 7, 7] == pytest.approx(0.84375)
    assert torch.all(alpha[..., 10, :, :] == 1)
    assert torch.all(audit.region_weight(torch.zeros_like(labels)) == 1)


def test_candidate_audit_separates_coverage_ranking_and_harmful_switches():
    labels = torch.tensor([[2, 7, 22, 8], [3, 3, 3, 3], [2, 3, 22, 3]])
    fixed = torch.tensor([2, 7, 22, 8])
    center = torch.tensor([2, 3, 3, 8])
    probability = torch.tensor([[[0.1, 0.8, 0.9, 0.1], [0.8, 0.1, 0.05, 0.8], [0.1, 0.1, 0.05, 0.1]]])
    result = audit.class_agreement(probability, labels, fixed, center, torch.tensor([True, False, False, True]))
    row = result["foreground"]
    assert row["candidate_class_coverage"] == 1
    assert row["before"] == row["after"] == 0.5
    assert row["correct_to_wrong"] == row["wrong_to_correct"] == 0.5
    assert row["correct_class_probability_mass"] == pytest.approx((0.2 + 0.8 + 0.95 + 0.1) / 4)
    assert result["cortex"]["queries"] == result["deep_gray"]["queries"] == 2
    assert result["ambiguous"]["queries"] == 2


def test_independent_window_reference_agrees_at_boundaries_and_low_contrast():
    torch.manual_seed(2)
    a = 0.9 + torch.randn(1, 1, 11, 12, 13) * 0.001
    b = 0.9 + torch.randn_like(a) * 0.001
    points = torch.tensor([[0, 5, 10], [0, 6, 11], [0, 6, 12]])
    expected = audit.point_reference(a, b, points)
    actual = stable_ncc_map(a, b)[0, 0, points[0], points[1], points[2]]
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-6)


def test_fixed_ncc_steps_preserve_zero_step_and_bound_each_update():
    torch.manual_seed(4)
    a = torch.rand(1, 1, 8, 9, 10) + 0.1
    b = torch.rand_like(a) + 0.1
    field = torch.zeros(1, 3, 8, 9, 10)
    options = dict(step_size=0.25, max_update=0.5, reg_weight=0.01, outside_weight=1.0)
    assert torch.equal(audit.ncc_steps(a, b, field, options, 0), field)
    result = audit.ncc_steps(a, b, field, options, 3)
    assert torch.isfinite(result).all() and result.abs().max() <= 1.5
    assert result.abs().max() > 0


@pytest.mark.parametrize("kind", ("b0", "b1", "normalised_t1", "resampled_t1", "shuffled_t2"))
def test_stable_controls_share_the_fixed_ten_epoch_recipe_and_new_optimizer(tmp_path, kind):
    args = SimpleNamespace(
        kind=kind,
        manifest=tmp_path / "original.json",
        prepared=tmp_path,
        checkpoint=tmp_path / "parent.pth",
        exp="stable",
        precision="bf16",
        seed=1,
        epochs=10,
        workers=8,
        ncc_mode="stable",
    )
    options = options_for(args)
    assert options.max_epoch == 10 and options.seed == 1 and options.ncc_mode == "stable"
    assert options.lr == 2e-5 and options.w_reg == options.finetune == 1
    assert options.init_ckpt == str(args.checkpoint) and not options.resume
    assert options.ixi_arm == ("b0" if kind == "b0" else "b1")


def test_seqvp_cell_adapter_uses_xyz_components_and_zyx_storage_on_anisotropic_grids():
    observed = []

    @dataclass
    class Verdict:
        status: str = "certified_feasible"

    def certify(vertices, spacing, *, config):
        assert vertices.shape == (8, 3) and spacing == (1.0, 1.0, 1.0)
        # x is the fastest vertex index; physical components are ordered x,y,z.
        np.testing.assert_allclose(vertices[1] - vertices[0], [-0.1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(vertices[2] - vertices[0], [0, 0.3, 0], atol=1e-6)
        np.testing.assert_allclose(vertices[4] - vertices[0], [0, 0, 0.2], atol=1e-6)
        assert config.delta == 1 / 1024
        observed.append(vertices)
        return Verdict()

    module = SimpleNamespace(
        certify_cell=certify, CertificateConfig=SimpleNamespace, AdaptiveCertificateConfig=SimpleNamespace
    )
    grid = voxel_grid((4, 5, 6))
    field = grid * torch.tensor([0.2, 0.3, -0.1]).view(1, 3, 1, 1, 1)
    rows = audit.seqvp_cells(field, module)
    assert observed and len(rows) * 2 == len(observed)
    assert all(row["SeqVP"]["status"] == "certified_feasible" for row in rows)
    assert all(row["ours"]["status"] == "CERTIFIED" for row in rows)


def test_report_cli_does_not_touch_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "summary", lambda _args: None)
    monkeypatch.setattr(audit, "configure_precision", lambda *_args: pytest.fail("CPU report configured CUDA"))
    monkeypatch.setattr("sys.argv", ["contrast_audit", "summary", "--out", str(tmp_path)])
    audit.main()


@pytest.fixture
def toy_audit(tmp_path, monkeypatch):
    shape = (19, 20, 21)
    rng = np.random.default_rng(3)
    groups = {}
    for key, split, count in (("training", "Train", 4), ("validation", "Val", 2)):
        rows = []
        for i in range(count):
            folder = tmp_path / f"{split}_{i}"
            folder.mkdir()
            a = rng.uniform(0.1, 1, size=shape).astype(np.float32)
            labels = np.where(np.indices(shape)[0] < shape[0] // 2, 3, 42).astype(np.int16)
            with (folder / "t1.pkl").open("wb") as stream:
                pickle.dump((a, labels), stream)
            np.save(folder / "t2.npy", rng.uniform(0.1, 1, size=shape).astype(np.float32))
            rows.append(
                dict(
                    id=f"{split}_{i}",
                    ixi_id=f"IXI_{split}_{i}",
                    split=split,
                    t1=str(folder / "t1.pkl"),
                    t2=str(folder / "t2.npy"),
                    t1_sha256=sha256_file(folder / "t1.pkl"),
                    t2_sha256=sha256_file(folder / "t2.npy"),
                )
            )
        groups[key] = rows
    manifest = tmp_path / "manifest.json"
    names = [r["id"] for r in groups["validation"]]
    atomic_write_json(
        manifest,
        dict(
            reviewed=True,
            shape=shape,
            expected_counts={"training": 4, "validation": 2},
            validation_pairs=[names, names[::-1]],
            **groups,
        ),
    )
    document = read_ixi_manifest(manifest)
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    for name in ("normalised_t1", "resampled_t1", "shuffled_t2"):
        atomic_write_json(prepared / "targets" / name / "manifest.json", document)
    base = tmp_path / "base.pth"
    base.write_bytes(b"parent")
    fit_ids, heldout_ids = teacher_subjects(document)
    recipe = dict(
        updates=1,
        seed=0,
        solver_steps=3,
        step_size=0.25,
        max_update=0.5,
        reg_weight=0.01,
        outside_weight=1.0,
        temperature=0.2,
    )
    for family in ("t1", "paired"):
        for kind in ("teacher", "student"):
            options = dict(
                input_channels=2 if kind == "teacher" else 1,
                width=2,
                channels=4,
                stride=1,
                precision="fp32",
                checkpoint_blocks=False,
            )
            torch.manual_seed(0)
            encoder = FeatureEncoder(**options)
            saved = dict(
                architecture="residual_fpn_native_v1",
                kind=kind,
                source=family,
                encoder_options=options,
                state_dict=encoder.state_dict(),
                recipe=recipe,
                manifest_sha256=document["fingerprint"],
                base_sha256=sha256_file(base),
                step=1,
                teacher_fit_ids=fit_ids,
                teacher_heldout_ids=heldout_ids,
            )
            folder = source / (f"teachers/{family}" if kind == "teacher" else f"matching/{family}_solver")
            (folder / "ckpt").mkdir(parents=True)
            torch.save(saved, folder / "ckpt/last.pth")
            atomic_write_json(folder / "recipe.json", {k: v for k, v in saved.items() if k != "state_dict"})

    class Base(torch.nn.Module):
        cfg = SimpleNamespace(image_channels=2)

        def forward(self, moving, _fixed, **_options):
            return moving, moving.new_zeros(1, 3, *moving.shape[2:])

    monkeypatch.setattr(audit, "load_base", lambda *_args: Base())
    return SimpleNamespace(
        manifest=manifest,
        prepared=prepared,
        source=source,
        base_checkpoint=base,
        b0_checkpoint=base,
        checkpoint=base,
        device=torch.device("cpu"),
        out=tmp_path / "output",
        dataset="IXI",
        limit=0,
        points=16,
        family="finetune",
    )


def test_real_diagnostic_pipeline_on_small_images_reuses_saved_teacher_and_query_outputs(toy_audit):
    args = toy_audit
    audit.inputs(args)
    assert audit.read(args.out / "numerics.json")["passed"]
    assert len(audit.read(args.out / "target_audit.json")["rows"]) == 16
    args.out = args.out / "teacher"
    audit.teachers(args)
    first = audit.read(args.out / "report.json")
    assert len(first["rows"]) == 14  # center/random and five descriptor sources, at two radii
    details = next((args.out / "pairs").glob("*/report.json"))
    assert all("energy_after" in row and "energy_before" in row for row in audit.read(details)["refinements"])
    audit.teachers(args)
    assert audit.read(args.out / "report.json") == first
    args.out = args.out.parent / "solvers"
    audit.solvers(args)
    rows = audit.read(args.out / "report.json")["rows"]
    assert len(rows) == 14
    assert all(
        np.isfinite(row[key])
        for row in rows
        for key in ("ncc_data_loss", "ncc_energy", "matching_energy", "diffusion_R")
    )


def test_finetuned_regional_repeat_is_train_only_and_includes_every_fixed_alpha(toy_audit):
    args = toy_audit
    # The primary scratch paths are only read for their frozen endpoint identities.
    for arm in ("scratch_b0", "scratch_b1"):
        atomic_write_json(args.source / "registration" / arm / "report.json", {"checkpoint": str(args.base_checkpoint)})
    audit.regional(args)
    saved = audit.read(args.out / "report.json")
    assert len(saved["rows"]) == 6 and not saved["failures"]
    assert {row["split"] for row in saved["rows"]} == {"Train"}
    assert [row["alpha"] for row in saved["rows"]] == [0.0, 0.25, 0.5, 0.75, 1.0, "label_oracle"]


def test_preflight_copies_completed_external_tables_and_rejects_a_missing_direction(toy_audit, monkeypatch):
    args = toy_audit
    external = args.source / "external"
    external.mkdir()
    subjects = []
    for i in range(2):
        path = external / f"s{i}.npz"
        path.write_bytes(b"external data")
        subjects.append(dict(id=path.stem, file=path.name, sha256=sha256_file(path)))
    args.external_manifest = external / "manifest.json"
    atomic_write_json(args.external_manifest, dict(subjects=subjects, pairs=[["s0", "s1"]]))
    atomic_write_json(
        args.source / "preflight.json",
        {"files": {str(path.resolve()): sha256_file(path) for path in (args.manifest, args.external_manifest)}},
    )
    (args.source / "head.txt").write_text("original_reviewed_head")
    monkeypatch.setattr(audit, "endpoints", lambda _args: {"base": (args.base_checkpoint, None, None)})
    for dataset, count in (("LPBA", 120), ("OASIS", 100)):
        directory = args.source / dataset / "base"
        atomic_write_json(
            directory / "report.json",
            dict(
                cases=count,
                failures=[],
                control_failures=[],
                binding={"base_sha256": sha256_file(args.base_checkpoint), "metric_sha256": None},
            ),
        )
        (directory / "FF").mkdir()
        (directory / "FF/per_case.csv").write_text(
            "case_id,dice_mean\n" + "".join(f"a{i}__b{i},0.8\n" for i in range(count))
        )
    audit.preflight(args)
    original = (args.source / "LPBA/base/FF/per_case.csv").read_bytes()
    assert (args.out / "LPBA/base/FF/per_case.csv").read_bytes() == original
    assert not audit.read(args.out / "LPBA/base/readout_origin.json")["recomputed"]
    (args.source / "LPBA/base/FF/per_case.csv").write_text("case_id,dice_mean\nonly__one,0.8\n")
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        audit.preflight(args)


def test_regional_signal_requires_both_macro_gain_and_preserved_cortex_after_repair():
    rows = [
        {
            "split": "Val",
            "case_id": case,
            "mode": "REP",
            "alpha": alpha,
            "dice_mean": value,
            "dice_lbl_2": cortex,
            "dice_lbl_22": cortex,
        }
        for case in ("a__b", "b__a")
        for alpha, value, cortex in (("base", 0.78, 0.80), ("label_oracle", 0.787, 0.80))
    ]
    assert audit.regional_criteria(rows, [])["Val"]["crosses_predeclared_criterion"]
    for row in rows:
        if row["alpha"] == "label_oracle":
            row["dice_lbl_2"] = row["dice_lbl_22"] = 0.79
    assert not audit.regional_criteria(rows, [])["Val"]["crosses_predeclared_criterion"]


def test_closure_reuses_frozen_targets_and_generates_a_matching_normalisation_control(cohort):
    old = cohort.out / "old"
    options = SimpleNamespace(**(vars(cohort) | {"out": old / "targets"}))
    contrast_learning.prepare(options)
    for kind in ("resampled_t1", "shuffled_t2"):
        source = old / "targets/targets" / kind / "manifest.json"
        atomic_write_json(old / "registration" / kind / "report.json", {"manifest_sha256": sha256_file(source)})
    out = cohort.out / "new/targets"
    args = SimpleNamespace(manifest=cohort.manifest, source=old, out=out, raw_root=cohort.raw_root, threads=1)
    audit.targets(args)
    for row in read_ixi_manifest(out / "targets/normalised_t1/manifest.json")["training"]:
        normal = np.load(row["t2"])
        restored = np.load(old / "targets/targets/resampled_t1/Train" / row["id"] / "aux.npy")
        np.testing.assert_allclose(normal, restored, atol=1e-7)  # Identity native grid in this fixture.
    first = (out / "targets/normalised_t1/manifest.json").read_bytes()
    audit.targets(args)
    assert (out / "targets/normalised_t1/manifest.json").read_bytes() == first


def test_normalisation_rejects_changed_frozen_qc_geometry(cohort):
    old = cohort.out / "old"
    contrast_learning.prepare(SimpleNamespace(**(vars(cohort) | {"out": old / "targets"})))
    for kind in ("resampled_t1", "shuffled_t2"):
        source = old / "targets/targets" / kind / "manifest.json"
        atomic_write_json(old / "registration" / kind / "report.json", {"manifest_sha256": sha256_file(source)})
    subject = read_ixi_manifest(cohort.manifest)["training"][0]
    qc_path = Path(subject["t2"]).parent / "qc.json"
    qc = audit.read(qc_path)
    qc["binding"]["geometry"]["origin"][0] += 1
    atomic_write_json(qc_path, qc)
    out = cohort.out / "changed"
    with pytest.raises(ValueError, match="frozen normalization QC changed"):
        audit.targets(
            SimpleNamespace(manifest=cohort.manifest, source=old, out=out, raw_root=cohort.raw_root, threads=1)
        )
    assert not (out / "targets/normalised_t1/manifest.json").exists()


def test_finetuned_regional_audit_does_not_require_scratch_reports(toy_audit):
    assert not (toy_audit.source / "registration").exists()
    audit.regional(toy_audit)
    assert audit.read(toy_audit.out / "report.json")["rows"]
