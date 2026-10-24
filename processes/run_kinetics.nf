process RUN_KINETICS {
    tag "Running Kinetic Simulation (${params.mimic_res} - ${params.regime})"
    label 'process_medium'

    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
    path graphs
    path site_mapping

    output:
    path "${params.output_csv}", emit: trajectories_csv

    script:
    """
    python -m polarity_engine.cli simulate \\
        --graphs_dir . \\
        --mappings site_mapping.json \\
        --alpha 1.2 \\
        --regime "${params.regime}" \\
        --output "${params.output_csv}"
    """
}
