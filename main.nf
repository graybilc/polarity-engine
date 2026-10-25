params.cif_file   = "${projectDir}/data/8r3y.cif"
params.ref_cif    = "${projectDir}/data/8r3y.cif"
params.lgl_chain  = 'L'
params.cutoff     = 8.0
params.sites      = '655,659,663'
params.mimic_res  = 'GLU'
params.regime     = 'processive'

// Standardized dynamic paths
params.graphs_dir = "${projectDir}/output/graphs_${params.mimic_res.toLowerCase()}"
params.output_csv = "simulation_trajectories_${params.mimic_res.toLowerCase()}_${params.regime}.csv"

include { ALIGN_CIF         } from './processes/align_cif'
include { PARSE_WT_CIF      } from './processes/parse_wt_cif'
include { MUTATE_STATE      } from './processes/mutate_state'
include { BUILD_STATE_GRAPH } from './processes/build_state_graph'
include { RUN_KINETICS      } from './processes/run_kinetics'

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

    // 1. Parse Target CIF (Capturing native SEP at 663 from 8r3y) and Impute Missing Loops
    PARSE_WT_CIF(
        ch_cif_file,
        ch_ref_cif,
        ALIGN_CIF.out.site_mapping
    )

    // 2. State Combinatorics
    // (States with '0' at site 663 will mutate SEP -> SER; states with '1' retain native SEP or apply mimic_res)
    ch_mutate_inputs = ch_state_codes.combine(PARSE_WT_CIF.out.wt_parsed)
    MUTATE_STATE(ch_mutate_inputs)

    // 3. Graph Tensor Assembly (Outputs 27-channel node feature tensors including SEP/SER and rSASA)
    BUILD_STATE_GRAPH(MUTATE_STATE.out.mutated_dict)

    // 4. Kinetic Simulation & rSASA Gating Solver
    ch_all_graphs = BUILD_STATE_GRAPH.out.graph_pt
        .map { state_code, graph_file -> graph_file }
        .collect()

    RUN_KINETICS(
        params.mimic_res,
        params.regime,
        ch_all_graphs,
        ALIGN_CIF.out.site_mapping
    )
}
