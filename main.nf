nextflow.enable.dsl = 2

params.cif_file  = "${projectDir}/data/8r3y.cif"
params.outdir    = "${projectDir}/output/module2"
params.lgl_chain = 'L'
params.cutoff    = 8.0
params.sites     = '655,659,663'

ch_state_codes = Channel.fromList([
    '000', '100', '010', '001',
    '110', '101', '011', '111'
])

// 0. Align & Verify Target Phosphorylation Sites (ADR-0004)
process ALIGN_CIF_SEQUENCES {
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

// 1. One-shot parsing using StructureParser.parse()
process PARSE_WT_CIF {
    tag 'Parse WT Reference'

    input:
    path cif_file
    path site_mapping_json

    output:
    path 'wt_parsed.pt', emit: wt_parsed

    script:
    """
    uv run python -c "
    import json
    from polarity_engine.parsers import StructureParser

    with open('${site_mapping_json}') as f:
        mapping_data = json.load(f)

    # Parse structural coordinates
    wt_parsed_dict = StructureParser.parse(
        '${cif_file}',
        chain_ids=['L']
    )

    # Attach/apply site mapping metadata
    wt_parsed_dict['site_mapping'] = mapping_data.get('sites', mapping_data)

    StructureParser.save_parsed_dict(wt_parsed_dict, 'wt_parsed.pt')
    "
    """
    stub:
    '''
    touch wt_parsed.pt
    '''
}

// 2. State Mutation
process MUTATE_STATE {
    tag "Mutate State ${state_code}"
    publishDir "${params.outdir}/mutated_dicts", mode: 'copy'

    input:
    tuple val(state_code), path(wt_parsed_pt)

    output:
    tuple val(state_code), path("mutated_dict_${state_code}.pt"), emit: mutated_dict

    script:
    """
    uv run python -c  "
import torch
from polarity_engine.parsers import StructureParser
from polarity_engine.state_generator import mutate_structure_dict

wt_dict = StructureParser.load_parsed_dict('${wt_parsed_pt}')
mutated_dict = mutate_structure_dict(wt_dict, '${state_code}')
torch.save(mutated_dict, 'mutated_dict_${state_code}.pt')
"
    """

    stub:
    """
    touch mutated_dict_${state_code}.pt
    """
}

// 3. Recompute SASA for Mutated Heavy Atoms
process COMPUTE_STATE_RSASA {
    tag "rSASA State ${state_code}"
    publishDir "${params.outdir}/features", mode: 'copy'

    input:
    tuple val(state_code), path(mutated_dict_pt)

    output:
    tuple val(state_code), path("rsasa_${state_code}.npy"), emit: rsasa_npy

    script:
    """
    uv run python -c  "
import torch
import numpy as np
from polarity_engine.parsers import StructureParser

mutated_dict = torch.load('${mutated_dict_pt}')

# Compute All-Atom FreeSASA and extract normalized node vector
sasa_map = StructureParser.compute_allatom_sasa_dict(mutated_dict)
rsasa_vec = StructureParser._compute_rsasa_vector(mutated_dict['nodes'], sasa_map)

np.save('rsasa_${state_code}.npy', rsasa_vec)
"
    """

    stub:
    """
    touch rsasa_${state_code}.npy
    """
}

// 4. Assemble Graph Tensor
process BUILD_STATE_GRAPH {
    tag "Graph State ${state_code}"
    publishDir "${params.outdir}/graphs", mode: 'copy'

    input:
    tuple val(state_code), path(mutated_dict_pt), path(rsasa_npy)

    output:
    tuple val(state_code), path("graph_state_${state_code}.pt"), emit: graph_pt

    script:
    """
    uv run python -c  "
import torch
import numpy as np
from polarity_engine.builder import ProteinGraphBuilder

mutated_dict = torch.load('${mutated_dict_pt}')
rsasa_np = np.load('${rsasa_npy}')

# Reuse cached ANM MSF vector computed during PARSE_WT_CIF
anm_msf_np = mutated_dict['anm_msf']

builder = ProteinGraphBuilder(distance_cutoff=${params.cutoff})
graph_data = builder.build_from_parsed_dict(
    parsed_dict=mutated_dict,
    rsasa_np=rsasa_np,
    anm_msf_np=anm_msf_np,
    name='state_${state_code}'
)

torch.save(graph_data, 'graph_state_${state_code}.pt')
"
    """

    stub:
    """
    touch graph_state_${state_code}.pt
    """
}

workflow {
    // 0. Resolve target site coordinates / fallback anchors
    ALIGN_CIF_SEQUENCES(
        params.cif_file,
        params.sites,
        params.lgl_chain
    )

    // 1. Pass mapping JSON into parse step
    PARSE_WT_CIF(
        params.cif_file,
        ALIGN_CIF_SEQUENCES.out.site_mapping
    )

    // 2. State Combinatorics
    ch_mutate_inputs = ch_state_codes.combine(PARSE_WT_CIF.out.wt_parsed)
    MUTATE_STATE(ch_mutate_inputs)

    // 3. Feature Generation
    COMPUTE_STATE_RSASA(MUTATE_STATE.out.mutated_dict)

    // 4. Graph Tensor Assembly
    ch_graph_inputs = MUTATE_STATE.out.mutated_dict.join(COMPUTE_STATE_RSASA.out.rsasa_npy)
    BUILD_STATE_GRAPH(ch_graph_inputs)
}
