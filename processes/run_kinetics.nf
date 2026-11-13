process RUN_KINETICS {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    val mimic_res
    val regime
    val alpha
    val graphs_dir
    path graphs
    path site_mapping

    output:
    path "simulation_trajectories_*.csv"

    script:
    """
    python -m polarity_engine.cli simulate \
        --graphs_dir "." \
        --mappings "${site_mapping}" \
        --alpha ${alpha} \
        --regime "${regime}" \
        --mimic-res "${mimic_res}" \
        --output "simulation_trajectories_${mimic_res.toLowerCase()}_${regime}.csv"
    """
}
