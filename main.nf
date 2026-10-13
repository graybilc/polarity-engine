nextflow.enable.dsl = 2

params.cif_file  = "${projectDir}/data/8r3y.cif"
params.ref_cif   = "${projectDir}/data/8r3y.cif"
params.outdir    = "${projectDir}/output/module2"
params.lgl_chain = 'L'
params.cutoff    = 8.0
params.sites     = '655,659,663'

include { ALIGN_CIF         } from './processes/align_cif'
include { PARSE_WT_CIF      } from './processes/parse_wt_cif'
include { MUTATE_STATE      } from './processes/mutate_state'
include { BUILD_STATE_GRAPH } from './processes/build_state_graph'

workflow {
    ch_state_codes = Channel.fromList([
        '000', '100', '010', '001',
        '110', '101', '011', '111'
    ])

    ch_cif_file = Channel.value(file(params.cif_file, checkIfExists: !workflow.stubRun))
    ch_ref_cif  = Channel.value(file(params.ref_cif, checkIfExists: !workflow.stubRun))

    // 0. Align Target Phosphorylation Sites against Reference
    ALIGN_CIF(
        ch_cif_file,
        ch_ref_cif,
        params.sites,
        params.lgl_chain
    )

    // 1. Parse Target CIF and Impute Missing Loops
    PARSE_WT_CIF(
        ch_cif_file,
        ch_ref_cif,
        ALIGN_CIF.out.site_mapping
    )

    // 2. State Combinatorics
    ch_mutate_inputs = ch_state_codes.combine(PARSE_WT_CIF.out.wt_parsed)
    MUTATE_STATE(ch_mutate_inputs)

    // 3. Graph Tensor Assembly
    BUILD_STATE_GRAPH(MUTATE_STATE.out.mutated_dict)
}
