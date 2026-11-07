process RUN_KINETICS {
    tag "Running Kinetic Simulation (${params.mimic_res} - ${params.regime})"
    label 'process_medium'

    publishDir "${params.outdir}/trajectories/${params.mimic_res}", mode: 'copy', overwrite: true

    input:
    val mimic_res
    val regime
    val alpha
    path graphs_dir
    path site_mapping

    output:
    path "simulation_trajectories_${params.mimic_res.toLowerCase()}_${regime.toLowerCase()}.csv", emit: trajectories_csv

    script:
    def output_filename = "simulation_trajectories_${params.mimic_res.toLowerCase()}_${regime.toLowerCase()}.csv"
    """
    python -m polarity_engine.cli simulate \
        --graphs_dir "." \
        --mappings "${site_mapping}" \
        --alpha ${alpha} \
        --regime "${regime}" \
        --mimic-res "${mimic_res}" \
        --output "${output_filename}"
    """
}
