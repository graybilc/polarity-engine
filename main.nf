nextflow.enable.dsl=2

params.cif_file  = "${projectDir}/data/8r3y.cif"
params.outdir    = "${projectDir}/output/module2"
params.lgl_chain = "B"
params.cutoff    = 8.0

ch_state_codes = Channel.fromList([
    '000', '100', '010', '001',
    '110', '101', '011', '111'
])

// 1. One-shot parsing using StructureParser.parse()
process PARSE_WT_CIF {
    tag "Parse WT Reference"
    publishDir "${params.outdir}/parsed_cache", mode: 'copy'

    input:
    path cif_file

    output:
    path "wt_parsed.pt", emit: wt_parsed

    script:
    """
    python3 -c "
from polarity_engine.parsers import StructureParser

# Full end-to-end extraction: Coords, rSASA, and ANM MSF
wt_parsed_dict = StructureParser.parse(
    '${cif_file}',
    chain_ids=['${params.lgl_chain}']
)

StructureParser.save_parsed_dict(wt_parsed_dict, 'wt_parsed.pt')
"
    """

    stub:
    """
    touch wt_parsed.pt
    """
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
    python3 -c "
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
    python3 -c "
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
    python3 -c "
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
    PARSE_WT_CIF(params.cif_file)

    ch_mutate_inputs = ch_state_codes.combine(PARSE_WT_CIF.out.wt_parsed)
    MUTATE_STATE(ch_mutate_inputs)

    COMPUTE_STATE_RSASA(MUTATE_STATE.out.mutated_dict)

    ch_graph_inputs = MUTATE_STATE.out.mutated_dict.join(COMPUTE_STATE_RSASA.out.rsasa_npy)
    BUILD_STATE_GRAPH(ch_graph_inputs)
}
