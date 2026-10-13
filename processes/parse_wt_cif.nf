process PARSE_WT_CIF {
    tag "Parse ${target_cif.name}"
    label 'process_low'

    publishDir "${params.outdir}/parsed", mode: 'copy', overwrite: true

    input:
    path target_cif,    stageAs: 'target.cif'
    path ref_cif,       stageAs: 'reference.cif'
    path site_mapping

    output:
    path "wt_parsed.pt", emit: wt_parsed

    script:
    """
    uv run python ${projectDir}/scripts/parse_wt_cif.py \\
        --cif "target.cif" \\
        --ref-cif "reference.cif" \\
        --mapping "${site_mapping}" \\
        --chain "${params.lgl_chain}" \\
        --out "wt_parsed.pt"
    """

    stub:
    """
    touch wt_parsed.pt
    """
}
