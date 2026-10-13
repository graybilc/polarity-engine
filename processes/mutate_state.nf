process MUTATE_STATE {
    tag "Mutate State ${state_code}"
    label 'process_medium'

    publishDir "${params.outdir}/mutated_states", mode: 'copy', overwrite: true

    input:
    tuple val(state_code), path(wt_parsed_pt)

    output:
    tuple val(state_code), path("mutated_${state_code}.pt"), emit: mutated_dict

    script:
    """
    uv run python ${projectDir}/scripts/mutate_state.py \\
        --input "${wt_parsed_pt}" \\
        --state-code "${state_code}" \\
        --out "mutated_${state_code}.pt"
    """

    stub:
    """
    touch mutated_${state_code}.pt
    """
}
