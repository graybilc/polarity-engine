process ALIGN_CIF {
    tag "Align ${target_cif.name}"
    label 'process_low'

    publishDir "${params.outdir}/alignments", mode: 'copy', overwrite: true

    input:
    path target_cif, stageAs: 'target.cif'
    path ref_cif,    stageAs: 'reference.cif'
    val sites
    val chain_id

    output:
    path 'site_mapping.json', emit: site_mapping

    script:
    """
    uv run python ${projectDir}/scripts/align_cif_sequences.py \\
        --target "target.cif" \\
        --reference "reference.cif" \\
        --sites "${sites}" \\
        --chain "${chain_id}" \\
        --output "site_mapping.json"
    """

    stub:
    """
    echo '{"sites": {}}' > site_mapping.json
    """
}
