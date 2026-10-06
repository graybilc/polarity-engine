"""
End-to-End integration test for Nextflow main.nf execution pipeline.
"""

import subprocess
from pathlib import Path
import pytest
import torch


class TestNextflowPipeline:
    """
    Integration test suite executing main.nf and validating downstream graph artifacts.
    """

    @pytest.mark.integration
    def test_nextflow_pipeline_e2e_stub(self, tmp_path: Path) -> None:
        sample_cif = Path("tests/fixtures/tiny_sample.cif").resolve()
        if not sample_cif.exists():
            pytest.skip(f"Integration fixture '{sample_cif}' not found.")

        ref_cif_copy = tmp_path / "ref_stub_sample.cif"
        ref_cif_copy.write_bytes(sample_cif.read_bytes())

        outdir = tmp_path / "pipeline_output"
        work_dir = tmp_path / "work"

        cmd = [
            "nextflow",
            "run",
            "main.nf",
            "-stub",
            "-work-dir",
            str(work_dir.resolve()),
            "--cif_file",
            str(sample_cif),
            "--ref_cif",
            str(ref_cif_copy),
            "--outdir",
            str(outdir.resolve()),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        assert (
            result.returncode == 0
        ), f"Nextflow -stub failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.integration
    def test_nextflow_pipeline_full_execution(self, tmp_path: Path) -> None:
        sample_cif = Path("tests/fixtures/tiny_sample.cif").resolve()
        if not sample_cif.exists():
            pytest.skip(f"Integration fixture '{sample_cif}' not found.")

        # Create copy for ref_cif to guarantee distinct file paths
        ref_cif_copy = tmp_path / "ref_full_sample.cif"
        ref_cif_copy.write_bytes(sample_cif.read_bytes())

        outdir = tmp_path / "output_e2e"
        cmd = [
            "nextflow",
            "run",
            "main.nf",
            "--cif_file",
            str(sample_cif),
            "--ref_cif",
            str(ref_cif_copy),
            "--outdir",
            str(outdir.resolve()),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        assert (
            result.returncode == 0
        ), f"Pipeline execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        state_codes = ["000", "100", "010", "001", "110", "101", "011", "111"]
        for code in state_codes:
            graph_path = outdir / "graphs" / f"graph_{code}.pt"
            assert graph_path.exists(), f"Missing output tensor: {graph_path}"

            graph_data = torch.load(graph_path, weights_only=False)
            assert hasattr(graph_data, "x")
            assert hasattr(graph_data, "edge_index")
