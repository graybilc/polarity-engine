process BUILD_STATE_GRAPH {
    tag "Build Graph ${state_code} (${mimic_res})"
    label 'process_medium'

    publishDir "${params.outdir}/graphs/${params.mimic_res.toUpperCase()}", mode: 'copy', overwrite: true

    input:
    tuple val(state_code), path(mutated_dict_pt)
    val mimic_res

    output:
    tuple val(state_code), path("graph_${state_code}.pt"), emit: graph_pt

    script:
    """
    uv run python ${projectDir}/scripts/build_state_graph.py \
        --input "${mutated_dict_pt}" \
        --state-code "${state_code}" \
        --cutoff "${params.cutoff}" \
        --output "graph_${state_code}.pt"
    """

    stub:
    """
    python -c "import torch; from torch_geometric.data import Data; data = Data(x=torch.zeros(30, 32), edge_index=torch.zeros((2, 0), dtype=torch.long)); data.nodes = [('A', '655')]*10 + [('A', '659')]*10 + [('A', '663')]*10; data.state_code = '${state_code}'; torch.save(data, 'graph_${state_code}.pt')"
    """
}
