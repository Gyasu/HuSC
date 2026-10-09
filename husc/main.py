#!/usr/bin/env python3
"""
HuSC (Human Spatial Constraint) score computation.

For each residue of an AlphaFold model, HuSC compares the gnomAD missense
variation in its 3D contact set (the residue plus all residues whose CB atoms,
CA for glycine, lie within the contact radius) with a permutation null of
neutral variation (see husc.null_model):

    HuSC = Phi^-1(p),   p = mid-p P(null MAF sum <= observed MAF sum)

where the MAF sum is the summed minor allele frequency of the missense variants
in the contact set. Negative HuSC means less missense variation than expected
(constraint), positive means more. With 10,000 permutations HuSC lies in about
[-3.89, +3.89].

Usage:
    python -m husc.main -c config.json -i input_pairs.txt [-l husc.log] [-w] [-v]
"""

import csv
import gzip
import json
import logging
import os
import warnings
import zlib
from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple
from statistics import NormalDist

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, PPBuilder, is_aa
from Bio.SeqUtils import seq1
from tqdm import tqdm

from husc import data, null_model
from husc.utils import pdb_utils, seq_utils

warnings.simplefilter('ignore', BiopythonWarning)

PDB_CHAIN = 'A'

OUTPUT_COLUMNS = [
    'uniprot_id', 'enst_id', 'uniprot_pos', 'uniprot_aa',
    'seq_separations', 'num_contacts', 'syn_var_sites',
    'total_syn_sites', 'mis_var_sites', 'total_mis_sites',
    'cs_syn_poss', 'cs_mis_poss', 'cs_gc_content', 'cs_syn_prob',
    'cs_syn_obs', 'cs_mis_prob', 'cs_mis_obs', 'cs_mis_exp',
    'HuSC', 'mis_p_value', 'cs_syn_exp', 'syn_p_value',
    'enst_syn_obs', 'enst_mis_obs', 'enst_syn_exp', 'enst_mis_exp',
    'plddt', 'uniprot_length'
]

# Reference data shared by all proteins
Inputs = namedtuple('Inputs', ['cds', 'variants', 'uniprot_to_enst', 'transcripts', 'cpg_maf', 'cpg_scale'])


class SkipProtein(Exception):
    """A protein that cannot be scored; the message is the reason counted in the log summary."""


def parse_cmd():
    parser = ArgumentParser(description='Compute HuSC (Human Spatial Constraint) scores for protein residues')
    parser.add_argument('-c', '--config', required=True,
                        help='JSON configuration file (data paths, radius, seed)')
    parser.add_argument('-i', '--input', required=True,
                        help='File of UniProt ID and AlphaFold model pairs, e.g. "P13929 AF-P13929-F1-model_v6.pdb"')
    parser.add_argument('-w', '--overwrite', action='store_true',
                        help='Recompute proteins whose output file already exists')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show a progress bar and echo log messages to the terminal')
    parser.add_argument('-l', '--log', default='husc.log',
                        help='Log file (default: husc.log)')
    return parser.parse_args()


def protein_rng(seed, uniprot_id):
    """Random generator for one protein, so results do not depend on input order or chunking."""
    return np.random.default_rng([seed, zlib.crc32(uniprot_id.encode())])


def load_structure(pdb_file):
    """
    Parse an AlphaFold model.

    Returns
    -------
    tuple
        (first polypeptide sequence as str, chain PDB_CHAIN of the first model)
    """
    if not os.path.exists(pdb_file):
        raise SkipProtein('missing_structure_file')
    try:
        with gzip.open(pdb_file, 'rt') as handle:
            structure = PDBParser(PERMISSIVE=1).get_structure('protein', handle)
        chain = structure[0][PDB_CHAIN]
    except (OSError, ValueError, KeyError) as e:
        raise SkipProtein('bad_structure') from e
    peptides = [pp.get_sequence() for pp in PPBuilder().build_peptides(structure)]
    peptides = [p for p in peptides if p]
    if not peptides:
        raise SkipProtein('bad_structure')
    return str(peptides[0]), chain


def contact_map(residues, radius):
    """{residue: [contacting residues]} within radius (CB atoms; CA for glycine)."""
    contacts = defaultdict(list)
    for c in pdb_utils.search_for_all_contacts(residues, radius=radius):
        contacts[c.get_res_a()].append(c.get_res_b())
        contacts[c.get_res_b()].append(c.get_res_a())
    return contacts


def permutation_nulls(cds, length, syn_exp, mis_exp, maf_dist, inputs, rng):
    """Missense and synonymous permutation null matrices, (N_PERMUTATIONS, length) each."""
    snv_rates = null_model.get_snv_mutation_rates(cds)
    nulls = {}
    for kind, expected in (('mis', mis_exp), ('syn', syn_exp)):
        codons, rates = snv_rates[kind]
        rates, cpg_mask = null_model.scale_cpg_rates(codons, rates, cds, inputs.cpg_scale)
        nulls[kind] = null_model.permute_variants(
            codons, rates, length, expected, maf_dist, inputs.cpg_maf, cpg_mask, rng
        )
    return nulls['mis'], nulls['syn']


def score_protein(uniprot_id, model_name, inputs, config, rng):
    """
    Compute HuSC and the per-residue statistics for one protein.

    Returns
    -------
    list of list
        One output row per scored residue, in OUTPUT_COLUMNS order.

    Raises
    ------
    SkipProtein
        If the protein cannot be scored.
    """
    pep_seq, chain = load_structure(os.path.join(config['pdb_dir'], model_name) + '.gz')

    if uniprot_id not in inputs.uniprot_to_enst:
        raise SkipProtein('no_enst_for_uniprot')
    try:
        enst_id = data.select_transcript(
            len(pep_seq), inputs.uniprot_to_enst[uniprot_id], inputs.cds, inputs.variants
        )
    except ValueError:
        raise SkipProtein('no_valid_cds')
    except KeyError:
        raise SkipProtein('no_variant_data')

    residues = [r for r in chain.get_residues() if is_aa(r, standard=True)]
    if not residues or len(residues) / len(pep_seq) < 1.0 / 3.0:
        raise SkipProtein('bad_structure')
    contacts = contact_map(residues, config['radius'])

    cds = inputs.cds[enst_id][:-3]  # drop the stop codon
    codon_rates = seq_utils.get_codon_mutation_rates(cds)
    codon_ns_counts = seq_utils.count_poss_ns_variants(cds)
    codon_ns_sites = seq_utils.count_ns_sites(cds)
    if len(codon_rates) < len(residues):
        raise SkipProtein('bad_structure')

    if enst_id not in inputs.transcripts:
        raise SkipProtein('no_enst_mp')
    stats = inputs.transcripts[enst_id]
    if len(stats.syn_af) == 0:
        raise SkipProtein('empty_af_dist')

    variants = inputs.variants[enst_id]
    mis_maf, syn_maf = data.sum_maf_by_position(variants)

    # Expected counts calibrated by the transcript's synonymous o/e;
    # non-CpG SNVs draw MAFs from the transcript's non-CpG synonymous MAFs
    syn_exp, mis_exp = null_model.calibrated_expected(stats.syn_obs, stats.syn_exp, stats.mis_exp)
    maf_dist, _ = null_model.syn_maf_by_cpg_class(variants, cds)
    if len(maf_dist) == 0:
        maf_dist = stats.syn_af
    try:
        mis_null, syn_null = permutation_nulls(cds, len(pep_seq), syn_exp, mis_exp, maf_dist, inputs, rng)
    except ValueError:
        raise SkipProtein('permutation_error')

    rows = []
    for pos, aa in enumerate(pep_seq, start=1):
        try:
            res = chain[pos]
        except KeyError:
            continue
        if seq1(res.get_resname()) != aa:
            raise SkipProtein('residue_mismatch')

        # sorted, because NeighborSearch returns contacts in a run-dependent order
        neighbors = sorted(r.get_id()[1] for r in contacts[res])
        sites = neighbors + [pos]
        if max(sites) > len(codon_rates):
            raise SkipProtein('contact_out_of_range')

        seq_context = seq_utils.get_codon_seq_context(sites, cds)
        if len(seq_context) == 0:
            logging.warning(f'No sequence context for {uniprot_id} position {pos}; residue skipped')
            continue

        # Focal residue first, then neighbors, as in the original summation order
        order = [pos] + neighbors
        mis_obs = sum(mis_maf.get(j, 0) for j in order)
        syn_obs = sum(syn_maf.get(j, 0) for j in order)
        mis_exp_cs, mis_p = null_model.permutation_test(mis_null, sites, mis_obs)
        syn_exp_cs, syn_p = null_model.permutation_test(syn_null, sites, syn_obs)
        husc = NormalDist().inv_cdf(mis_p)

        rows.append([
            uniprot_id, enst_id, pos, aa,
            ';'.join(str(j - pos) for j in neighbors),
            len(sites),
            sum(1 for j in order if j in syn_maf),
            '{:.3f}'.format(sum(codon_ns_sites[j - 1][1] for j in order)),
            sum(1 for j in order if j in mis_maf),
            '{:.3f}'.format(sum(codon_ns_sites[j - 1][0] for j in order)),
            sum(codon_ns_counts[j - 1][1] for j in order),
            sum(codon_ns_counts[j - 1][0] for j in order),
            '{:.3f}'.format(seq_utils.gc_content(seq_context)),
            '{:.3e}'.format(sum(codon_rates[j - 1][0] for j in order)),
            syn_obs,
            '{:.3e}'.format(sum(codon_rates[j - 1][1] for j in order)),
            mis_obs,
            '{:.3e}'.format(mis_exp_cs),
            '{:.3f}'.format(husc),
            '{:.3e}'.format(mis_p),
            '{:.3e}'.format(syn_exp_cs),
            '{:.3e}'.format(syn_p),
            stats.syn_obs,
            stats.mis_obs,
            syn_exp,
            mis_exp,
            '{:.2f}'.format(res['CA'].get_bfactor()),
            len(pep_seq),
        ])
    return rows


def write_rows(path, rows):
    """Write the output table; a temporary file is renamed so interrupted runs leave no partial output."""
    tmp = path + '.tmp'
    with open(tmp, 'wt') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)
    os.replace(tmp, path)


def load_inputs(config, uniprot_ids):
    """Load the reference data needed for the given proteins."""
    logging.info('Loading UniProt-to-Ensembl mapping')
    uniprot_to_enst = data.load_json(config['uniprot_to_enst'])
    enst_ids = {e for u in uniprot_ids for e in uniprot_to_enst.get(u, [])}
    logging.info('Loading Ensembl CDS sequences')
    cds = data.load_cds(config['ensembl_cds'])
    logging.info(f'Loading gnomAD variants for {len(enst_ids)} transcripts')
    variants = data.load_variants(config['gnomad_variants'], enst_ids)
    logging.info('Loading transcript mutation probabilities and counts')
    transcripts = data.read_transcript_table(config['enst_mp_counts'], enst_ids)
    cpg_maf = np.loadtxt(config['cpg_maf'], ndmin=1)
    with open(config['cpg_scale']) as handle:
        cpg_scale = json.load(handle)
    logging.info(f'{len(cpg_maf)} pooled CpG synonymous MAFs; CpG rate factors {cpg_scale["factors"]}')
    return Inputs(cds, variants, uniprot_to_enst, transcripts, cpg_maf, cpg_scale)


def main():
    args = parse_cmd()
    config = data.load_config(args.config)

    logging.basicConfig(filename=args.log, level=logging.INFO, filemode='w',
                        format='%(levelname)s:%(asctime)s:%(message)s')
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'HuSC: radius {config["radius"]} A, seed {config["seed"]}, '
                 f'{null_model.N_PERMUTATIONS} permutations, calibration k={null_model.CALIB_K}')

    output_dir = os.path.abspath(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    pairs = data.read_input_pairs(args.input)
    if not args.overwrite:
        done = [p for p in pairs if os.path.exists(os.path.join(output_dir, f'{p[0]}_husc.tsv'))]
        pairs = [p for p in pairs if p not in done]
        logging.info(f'{len(done)} proteins already have output and are skipped')
    if not pairs:
        return

    inputs = load_inputs(config, {u for u, _ in pairs})

    skipped = Counter()
    for uniprot_id, model_name in tqdm(pairs, disable=not args.verbose):
        try:
            rows = score_protein(uniprot_id, model_name, inputs, config,
                                 protein_rng(config['seed'], uniprot_id))
        except SkipProtein as e:
            skipped[str(e)] += 1
            logging.warning(f'{uniprot_id} ({model_name}) skipped: {e}')
            continue
        write_rows(os.path.join(output_dir, f'{uniprot_id}_husc.tsv'), rows)
        logging.info(f'{uniprot_id}: {len(rows)} residues scored')

    logging.info(f'Done: {len(pairs) - sum(skipped.values())} proteins scored')
    for reason, n in sorted(skipped.items()):
        logging.info(f'skipped, {reason}: {n}')


if __name__ == '__main__':
    main()
