process BUILD_STATE_GRAPH {
    tag "Build Graph ${state_code}"
    label 'process_medium'

    publishDir "${params.outdir}/graphs", mode: 'copy', overwrite: true

    input:
    tuple val(state_code), path(mutated_dict_pt)

    output:
    tuple val(state_code), path("graph_${state_code}.pt"), emit: graph_pt

    script:
    """
    uv run python ${projectDir}/scripts/build_state_graph.py \\
        --input "${mutated_dict_pt}" \\
        --state-code "${state_code}" \\
        --cutoff "${params.cutoff}" \\
        --output "graph_${state_code}.pt"
    """

    stub:
    """
    touch "graph_${state_code}.pt"
    """
}
