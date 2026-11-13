params.cif_file   = "${projectDir}/data/8r3y.cif"
params.ref_cif    = "${projectDir}/data/8r3y.cif"
params.lgl_chain  = 'L'
params.cutoff     = 8.0
params.sites      = '655,659,663'
params.mimic_res  = params.mimic_res ?: 'ASP'
params.regime     = 'processive'
params.alpha      = 1.2

// Automatically isolate outputs and working directories by mimic residue
params.outdir     = "${projectDir}/output"
params.graphs_dir = "${params.outdir}/graphs/${params.mimic_res.toUpperCase()}"
params.output_csv = "simulation_trajectories_${params.mimic_res.toLowerCase()}_${params.regime}.csv"

include { ALIGN_CIF         } from './processes/align_cif'
include { PARSE_WT_CIF      } from './processes/parse_wt_cif'
include { MUTATE_STATE      } from './processes/mutate_state'
include { BUILD_STATE_GRAPH } from './processes/build_state_graph'
include { RUN_KINETICS      } from './processes/run_kinetics'

workflow {
    // Generate all 27 ternary state codes (3^3 space: '000' to '222') natively in Groovy
    ch_state_codes = Channel.from(['0', '1', '2'])
        .combine(Channel.from(['0', '1', '2']))
        .combine(Channel.from(['0', '1', '2']))
        .map { a, b, c -> "${a}${b}${c}" }

    ch_cif_file = Channel.value(file(params.cif_file, checkIfExists: !workflow.stubRun))
    ch_ref_cif  = Channel.value(file(params.ref_cif, checkIfExists: !workflow.stubRun))

    // 0. Align Target Phosphorylation Sites against Reference
    ALIGN_CIF(
        ch_cif_file,
        ch_ref_cif,
        params.sites,
        params.lgl_chain
    )

    // 1. Parse Target CIF and Impute Missing Loops (converting native SEP to SER for '000' baseline)
    PARSE_WT_CIF(
        ch_cif_file,
        ch_ref_cif,
        ALIGN_CIF.out.site_mapping
    )

    // 2. State Combinatorics (Full 3^3 Matrix)
    ch_mutate_inputs = ch_state_codes.combine(PARSE_WT_CIF.out.wt_parsed)
    MUTATE_STATE(ch_mutate_inputs, params.mimic_res)

    // 3. Graph Tensor Assembly for all 27 states
    BUILD_STATE_GRAPH(MUTATE_STATE.out.mutated_dict, params.mimic_res)

    // 4. Kinetic Simulation across the 3^3 trajectory network
    ch_all_graphs = BUILD_STATE_GRAPH.out.graph_pt
        .map { state_code, graph_file -> graph_file }
        .collect()

    RUN_KINETICS(
        params.mimic_res,
        params.regime,
        params.alpha,
        params.graphs_dir,
        ch_all_graphs,
        ALIGN_CIF.out.site_mapping
    )
}
