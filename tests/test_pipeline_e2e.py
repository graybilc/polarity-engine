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
        """
        End-to-end integration test for Nextflow pipeline stub execution.

        Arrange:
            - Resolve sample CIF fixture and create reference copy.
            - Setup temporary output and work directories.
            - Construct Nextflow command with -stub flag.

        Act:
            - Execute the Nextflow workflow subprocess.

        Assert:
            - Verify that the pipeline returns a success code (0).
        """
        # Arrange
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

        # Act
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Assert
        assert (
            result.returncode == 0
        ), f"Nextflow -stub failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.integration
    def test_nextflow_pipeline_full_execution(self, tmp_path: Path) -> None:
        """
        End-to-end integration test for full Nextflow pipeline execution.

        Arrange:
            - Resolve sample CIF fixture and create reference copy.
            - Setup output and variant-specific graph directories matching main.nf logic.
            - Construct Nextflow command with explicit parameter overrides.

        Act:
            - Execute the full Nextflow workflow subprocess.

        Assert:
            - Verify process return code is 0.
            - Confirm existence of all 8 state graph tensors under the dynamic mimic-residue directory.
            - Validate loaded tensors contain valid 'x' and 'edge_index' attributes.
        """
        # Arrange
        sample_cif = Path("tests/fixtures/tiny_sample.cif").resolve()
        if not sample_cif.exists():
            pytest.skip(f"Integration fixture '{sample_cif}' not found.")

        ref_cif_copy = tmp_path / "ref_full_sample.cif"
        ref_cif_copy.write_bytes(sample_cif.read_bytes())

        outdir = tmp_path / "output_e2e"
        mimic_res = "GLU"

        # main.nf automatically dynamically resolves graphs_dir to ${params.outdir}/graphs/${params.mimic_res.toUpperCase()}
        graphs_dir = outdir / "graphs" / mimic_res.upper()

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
            "--mimic_res",
            mimic_res,
        ]

        # Act
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Assert
        assert (
            result.returncode == 0
        ), f"Pipeline execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        state_codes = ["000", "100", "010", "001", "110", "101", "011", "111"]
        for code in state_codes:
            graph_path = graphs_dir / f"graph_{code}.pt"
            assert graph_path.exists(), f"Missing output tensor: {graph_path}"

            graph_data = torch.load(graph_path, weights_only=False)
            assert hasattr(graph_data, "x")
            assert hasattr(graph_data, "edge_index")
