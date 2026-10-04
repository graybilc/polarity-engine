// processes/align_cif.nf

process ALIGN_CIF {
    tag 'Align CIF Target Sites'
    publishDir "${params.outdir}/alignment", mode: 'copy'

    input:
    path cif_file
    val sites
    val chain

    output:
    path 'site_mapping.json', emit: site_mapping

    script:
    """
    uv run python ${projectDir}/scripts/align_cif_sequences.py \
        --cif ${cif_file} \
        --sites "${sites}" \
        --chain ${chain} \
        --out site_mapping.json
    """

    stub:
    '''
    touch site_mapping.json
    '''
}
