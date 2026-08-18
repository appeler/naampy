import hashlib
import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_training import train_gender_lstm as trainer


def _name_set_sha256(names):
    encoded = json.dumps(
        sorted(names), ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_training_artifact(tmp_path):
    names = [
        "aarav",
        "aditi",
        "akash",
        "ananya",
        "arjun",
        "diya",
        "esha",
        "isha",
        "kabir",
        "kavya",
        "kiara",
        "meera",
        "neha",
        "priya",
        "rahul",
        "rohan",
        "saanvi",
        "tara",
        "varun",
        "vivaan",
        "ayaan",
        "devika",
        "gaurav",
        "lakshmi",
    ]
    partitions = trainer.PARTITIONS * 6
    female_counts = [index + 1 for index in range(len(names))]
    male_counts = [len(names) - index for index in range(len(names))]
    represented_counts = [
        female + male for female, male in zip(female_counts, male_counts, strict=True)
    ]
    order = sorted(range(len(names)), key=names.__getitem__)
    table = pa.Table.from_pydict(
        {
            "normalized_name": [names[index] for index in order],
            "female_label_record_count": [female_counts[index] for index in order],
            "male_label_record_count": [male_counts[index] for index in order],
            "represented_binary_label_record_count": [
                represented_counts[index] for index in order
            ],
            "partition": [partitions[index] for index in order],
        },
        schema=trainer.EXPECTED_TRAINING_SCHEMA,
    )
    parquet_path = tmp_path / "training.parquet"
    pq.write_table(table, parquet_path)
    partition_membership = {
        partition: _name_set_sha256(
            [
                name
                for name, assigned in zip(names, partitions, strict=True)
                if assigned == partition
            ]
        )
        for partition in trainer.PARTITIONS
    }
    manifest = {
        "schema_version": 1,
        "artifact_role": "name_pattern_model_training_data",
        "target": ("female share among female and male electoral-roll source labels"),
        "reference_population": (
            "synthetic aggregate electoral-roll registration records"
        ),
        "output": {
            "filename": parquet_path.name,
            "format": "parquet",
            "sha256": trainer.file_sha256(parquet_path),
        },
        "split": {
            "seed": 0,
            "fractions": {
                "training": 0.70,
                "validation": 0.10,
                "calibration": 0.10,
                "test": 0.10,
            },
            "exported_membership_sha256": partition_membership,
        },
    }
    manifest_path = tmp_path / "training.json"
    manifest_path.write_text(json.dumps(manifest))
    return parquet_path, manifest_path, trainer.file_sha256(manifest_path)


def test_staged_workflow_uses_only_permitted_partitions(tmp_path, monkeypatch):
    parquet_path, data_manifest_path, data_manifest_hash = _write_training_artifact(
        tmp_path
    )
    original_loader = trainer.load_partition_data
    loaded_partitions = []

    def tracked_loader(parquet, manifest, partitions):
        loaded_partitions.append(partitions)
        return original_loader(parquet, manifest, partitions)

    monkeypatch.setattr(trainer, "load_partition_data", tracked_loader)
    development_directory = tmp_path / "development"
    development_manifest_path = trainer.run_development_stage(
        parquet_path=parquet_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_hash,
        output_directory=development_directory,
        configuration=trainer.TrainingConfiguration(
            training_seeds=(0, 1),
            epochs=1,
            samples_per_epoch=8,
            batch_size=4,
            learning_rate=1e-3,
        ),
        requested_device="cpu",
    )
    assert loaded_partitions == [("training",), ("validation",)]
    assert not list(development_directory.glob("*.pt"))
    assert len(list(development_directory.glob("*.safetensors"))) == 2

    loaded_partitions.clear()
    final_directory = tmp_path / "final"
    ensemble_manifest_path = trainer.run_final_fit_stage(
        parquet_path=parquet_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_hash,
        development_manifest_path=development_manifest_path,
        development_manifest_sha256=trainer.file_sha256(development_manifest_path),
        output_directory=final_directory,
        requested_device="cpu",
    )
    assert loaded_partitions == [("training", "validation")]
    ensemble_manifest = json.loads(ensemble_manifest_path.read_text())
    assert ensemble_manifest["partition_access"] == {
        "loaded": ["training", "validation"],
        "reserved": ["calibration", "test"],
    }
    assert all(
        model["filename"].endswith(".safetensors")
        for model in ensemble_manifest["ensemble"]["models"]
    )

    loaded_partitions.clear()
    runtime_manifest_path = tmp_path / "runtime.json"
    calibration_path = trainer.run_calibration_stage(
        parquet_path=parquet_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_hash,
        ensemble_manifest_path=ensemble_manifest_path,
        ensemble_manifest_sha256=trainer.file_sha256(ensemble_manifest_path),
        calibration_manifest_output_path=tmp_path / "calibration.json",
        runtime_manifest_output_path=runtime_manifest_path,
        requested_device="cpu",
    )
    assert loaded_partitions == [("calibration",)]

    loaded_partitions.clear()
    test_report_path = trainer.run_test_stage(
        parquet_path=parquet_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_hash,
        ensemble_manifest_path=ensemble_manifest_path,
        ensemble_manifest_sha256=trainer.file_sha256(ensemble_manifest_path),
        calibration_manifest_path=calibration_path,
        calibration_manifest_sha256=trainer.file_sha256(calibration_path),
        runtime_manifest_path=runtime_manifest_path,
        runtime_manifest_sha256=trainer.file_sha256(runtime_manifest_path),
        output_path=tmp_path / "test.json",
        requested_device="cpu",
        bootstrap_iterations=2,
        bootstrap_seed=3,
    )
    assert loaded_partitions == [("test",)]
    test_report = json.loads(test_report_path.read_text())
    assert test_report["stage"] == "test_evaluation"
    assert test_report["metrics"]["calibrated"]["intervals"]["iterations"] == 2


def test_hash_gate_fails_before_final_fit_loads_rows(tmp_path, monkeypatch):
    parquet_path, data_manifest_path, data_manifest_hash = _write_training_artifact(
        tmp_path
    )
    development_manifest_path = tmp_path / "development.json"
    development_manifest_path.write_text("{}")
    loaded_partitions = []
    monkeypatch.setattr(
        trainer,
        "load_partition_data",
        lambda *args: loaded_partitions.append(args) or pytest.fail("loaded rows"),
    )

    with pytest.raises(ValueError, match="development manifest hash mismatch"):
        trainer.run_final_fit_stage(
            parquet_path=parquet_path,
            data_manifest_path=data_manifest_path,
            data_manifest_sha256=data_manifest_hash,
            development_manifest_path=development_manifest_path,
            development_manifest_sha256="0" * 64,
            output_directory=tmp_path / "final",
            requested_device="cpu",
        )

    assert loaded_partitions == []


@pytest.mark.parametrize("selected_epoch", [0, 2])
def test_final_fit_rejects_epoch_outside_budget_before_loading_rows(
    tmp_path, monkeypatch, selected_epoch
):
    parquet_path, data_manifest_path, data_manifest_hash = _write_training_artifact(
        tmp_path
    )
    development_manifest_path = trainer.run_development_stage(
        parquet_path=parquet_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_hash,
        output_directory=tmp_path / "development",
        configuration=trainer.TrainingConfiguration(
            training_seeds=(0, 1),
            epochs=1,
            samples_per_epoch=4,
            batch_size=2,
            learning_rate=1e-3,
        ),
        requested_device="cpu",
    )
    development_manifest = json.loads(development_manifest_path.read_text())
    development_manifest["models"][0]["selected_epoch"] = selected_epoch
    malformed_path = tmp_path / f"malformed-{selected_epoch}.json"
    malformed_path.write_text(json.dumps(development_manifest))
    loaded_partitions = []
    monkeypatch.setattr(
        trainer,
        "load_partition_data",
        lambda *args: loaded_partitions.append(args) or pytest.fail("loaded rows"),
    )

    with pytest.raises(ValueError, match="selected_epoch must be within"):
        trainer.run_final_fit_stage(
            parquet_path=parquet_path,
            data_manifest_path=data_manifest_path,
            data_manifest_sha256=data_manifest_hash,
            development_manifest_path=malformed_path,
            development_manifest_sha256=trainer.file_sha256(malformed_path),
            output_directory=tmp_path / "final",
            requested_device="cpu",
        )

    assert loaded_partitions == []


def test_cli_exposes_only_explicit_stages():
    completed_process = subprocess.run(
        [sys.executable, "-m", "model_training.train_gender_lstm", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    for stage in ("development", "fit-final", "calibrate", "score-test"):
        assert stage in completed_process.stdout
    assert "--out " not in completed_process.stdout
