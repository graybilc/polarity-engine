process RUN_KINETICS {
    tag "Running Kinetic Simulation (${mimic_res} - ${params.regime})"
    label 'process_medium'

    publishDir "${params.outdir}/trajectories", mode: 'copy', overwrite: true

    input:
    val mimic_res
    val regime
    path graphs_dir
    path site_mapping

    output:
    path "simulation_trajectories_${mimic_res.toLowerCase()}_${regime.toLowerCase()}.csv", emit: trajectories_csv

    script:
    def output_filename = "simulation_trajectories_${mimic_res.toLowerCase()}_${regime.toLowerCase()}.csv"
    """
    python -m polarity_engine.cli simulate \\
        --graphs_dir "." \\
        --mappings "${site_mapping}" \\
        --alpha 1.2 \\
        --regime "${regime}" \\
        --output "${output_filename}"
    """
}
