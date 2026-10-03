"""
End-to-End integration test for Nextflow main.nf execution pipeline.
"""

import subprocess
import pytest
import torch
from pathlib import Path


class TestNextflowPipeline:
    """
    Integration test suite executing main.nf and validating downstream graph artifacts.
    """

    @pytest.mark.integration
    def test_nextflow_pipeline_e2e_stub(self, tmp_path: Path) -> None:
        """
        Tests DAG DAG topology and channel emissions using Nextflow -stub mode.

        Arrange:
            Define Nextflow execution command targeting temporary output directory.
        Act:
            Run Nextflow process via subprocess.
        Assert:
            Assert process exit code is 0 (SUCCESS).
        """
        # Arrange
        outdir = tmp_path / "pipeline_output"
        cmd = [
            "nextflow",
            "run",
            "main.nf",
            "-stub",
            "--outdir",
            str(outdir),
        ]

        # Act
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Assert
        assert result.returncode == 0, f"Nextflow failed with error:\n{result.stderr}"
        assert (outdir / "graphs").exists()

    @pytest.mark.integration
    def test_nextflow_pipeline_full_execution(self, tmp_path: Path) -> None:
        """
        Executes end-to-end Nextflow run using sample fixture CIF and validates PyG graphs.

        Arrange:
            Resolve absolute path for sample CIF fixture and temp output path.
        Act:
            Run Nextflow pipeline to completion.
        Assert:
            Verify all 8 graph output tensors exist and load properly into PyTorch Geometric Data objects.
        """
        # Arrange
        # Convert relative path to resolved absolute path for Nextflow CLI safety
        sample_cif = Path("tests/fixtures/tiny_sample.cif").resolve()
        if not sample_cif.exists():
            pytest.skip(f"Integration fixture '{sample_cif}' not found.")

        outdir = tmp_path / "output_e2e"
        cmd = [
            "nextflow",
            "run",
            "main.nf",
            "--cif_file",
            str(sample_cif),
            "--outdir",
            str(outdir.resolve()),
        ]

        # Act
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Assert
        assert (
            result.returncode == 0
        ), f"Pipeline execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        state_codes = ["000", "100", "010", "001", "110", "101", "011", "111"]
        for code in state_codes:
            graph_path = outdir / "graphs" / f"graph_state_{code}.pt"
            assert graph_path.exists(), f"Missing output tensor: {graph_path}"

            # Verify PyTorch Geometric tensor loading
            graph_data = torch.load(graph_path)
            assert hasattr(graph_data, "x")
            assert hasattr(graph_data, "edge_index")
