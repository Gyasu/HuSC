#!/usr/bin/env python3
"""
HuSC (Human Spatial Constraint) scores for protein complexes.

This is the multi-chain counterpart of ``husc/main.py`` and follows the logic
of COSMIS' ``cosmis-complex`` script:

1. Every chain of the structure is mapped to a UniProt protein (and from there
   to an Ensembl transcript with gnomAD variants, a CDS, and expected counts).
2. Residue-residue contacts (CB-CB, CA for Gly, default 8 A) are searched over
   *all* mapped chains at once, so a residue's spatial neighborhood includes
   residues from the other subunits (inter-chain contacts).
3. For a focal residue, observed variant burden, possible sites and mutation
   rates are summed over the focal residue and all contacting residues, each
   looked up in the data of the chain/protein it belongs to.
4. The null distribution is built by permuting variants within each protein
   (one permutation matrix per chain) and summing the permuted burden over the
   same neighborhood, i.e. across chains. The HuSC score is then computed
   exactly as in ``main.py``.

Input file (``-i``): one complex per line, whitespace separated::

    <complex_id> <structure_file> <chain>:<uniprot_id>[:<offset>] [...]

Instead of (or in addition to) inline ``chain:uniprot`` entries, a COSMIS
``cosmis-complex`` chain map file (one ``CHAIN:UNIPROT`` per line) can be given::

    KCNQ1  KCNQ1.pdb  KCNQ1_chain_to_uniprot.txt

for example (hemoglobin, PDB 2PGH, PDB numbering starts one residue after the
UniProt initiator Met)::

    2PGH  2PGH.cif  A:P69905  B:P68871  C:P69905  D:P68871

``structure_file`` is resolved against ``pdb_dir`` in the config unless it is an
existing path. PDB and mmCIF files, optionally gzipped, are supported.

``offset`` is added to the structure's residue number to get the UniProt
position. If omitted, a constant offset is inferred by locating the chain's
first residues in the UniProt sequence (0 for AlphaFold-style models, whose
numbering already follows UniProt). Chains that still do not match the UniProt
sequence (>10% mismatching residues) are skipped.
"""

import csv
import logging
import os
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field

import gzip
import numpy as np
from Bio import SeqIO
from Bio import BiopythonWarning
from Bio.PDB import is_aa
from Bio.SeqUtils import seq1

from husc.utils import seq_utils, pdb_utils
from husc.main import (
    count_variants,
    get_uniprot_accession,
    load_datasets,
    parse_config,
    retrieve_data,
)

warnings.simplefilter('ignore', BiopythonWarning)

# fraction of a chain's modeled residues that must agree with the UniProt
# sequence for the chain to be used
MIN_SEQ_MATCH = 0.9


def parse_cmd():
    parser = ArgumentParser(
        description='Compute HuSC (Human Spatial Constraint) scores for the '
                    'subunits of protein complexes'
    )
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='JSON file specifying configuration options')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Complex definitions: "<complex_id> <structure_file> '
                             '<chain>:<uniprot>[:<offset>] ..." (one complex per line)')
    parser.add_argument('-r', '--radius', type=float, default=8.0,
                        help='Contact radius in Angstroms (default: 8.0)')
    parser.add_argument('-w', '--overwrite', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--chains', type=str, default=None,
                        help='Comma-separated chain IDs to score (default: all '
                             'mapped chains). Other chains still contribute to '
                             'the neighborhoods of the scored chains.')
    parser.add_argument('--shared-permutations', action='store_true',
                        help='Use one permutation matrix per UniProt protein '
                             'instead of one per chain, so identical subunits '
                             '(homo-oligomers) share the same null draws. '
                             'Default (COSMIS behavior): independent per chain.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for the permutations')
    parser.add_argument('-l', '--log', default='husc_complex.log',
                        help='Log file (default: husc_complex.log)')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Per-protein (transcript) data
# --------------------------------------------------------------------------- #
@dataclass
class ProteinData:
    """Everything HuSC needs to know about one protein, independent of chain."""
    uniprot_id: str
    enst_id: str
    pep_seq: str
    cds: str
    mis_counts: dict
    syn_counts: dict
    site_var_mis: dict
    site_var_syn: dict
    codon_mutation_rates: list
    all_cds_ns_counts: list
    cds_ns_sites: list
    total_mis_exp: int
    total_syn_exp: int
    mis_dist: list
    syn_dist: list
    enst_syn_obs: int
    enst_mis_obs: int
    perm_cache: dict = field(default_factory=dict)


class SkipComplex(Exception):
    """Raised for conditions that make a chain or complex unusable."""


def load_uniprot_sequences(configs):
    """UniProt ID -> protein sequence, from the UniProt FASTA in the config."""
    print('Loading UniProt protein sequences...')
    with gzip.open(configs['uniprot_pep'], 'rt') as handle:
        return {
            get_uniprot_accession(r): str(r.seq)
            for r in SeqIO.parse(handle, format='fasta')
        }


def build_protein_data(uniprot_id, pep_seq, cds_dict, variant_dict,
                       uniprot_to_enst, enst_mp_counts):
    """
    Collect transcript, variant, and expected-count data for one protein.
    Mirrors the per-protein part of ``husc.main.main``.
    """
    try:
        enst_ids = uniprot_to_enst[uniprot_id]
    except KeyError:
        raise SkipComplex(f'no Ensembl transcript mapped to {uniprot_id}')
    try:
        enst_id, cds, variants = retrieve_data(
            pep_seq, uniprot_id, enst_ids, cds_dict, variant_dict
        )
    except ValueError:
        raise SkipComplex(f'no valid CDS compatible with {uniprot_id}')
    except KeyError:
        raise SkipComplex(f'no gnomAD variant data for {uniprot_id}')

    cds = str(cds)[:-3]  # remove stop codon
    mis_counts, syn_counts = count_variants(variants)
    try:
        mp = enst_mp_counts[enst_id]
    except KeyError:
        raise SkipComplex(f'no mutation probability data for {enst_id}')

    return ProteinData(
        uniprot_id=uniprot_id,
        enst_id=enst_id,
        pep_seq=pep_seq,
        cds=cds,
        mis_counts=mis_counts,
        syn_counts=syn_counts,
        site_var_mis={pos: 1 for pos in mis_counts},
        site_var_syn={pos: 1 for pos in syn_counts},
        codon_mutation_rates=seq_utils.get_codon_mutation_rates(cds),
        all_cds_ns_counts=seq_utils.count_poss_ns_variants(cds),
        cds_ns_sites=seq_utils.count_ns_sites(cds),
        total_mis_exp=mp[-3],
        total_syn_exp=mp[-4],
        mis_dist=mp[-1],
        syn_dist=mp[-2],
        enst_syn_obs=mp[2],
        enst_mis_obs=mp[4],
    )


def make_permutations(pd):
    """Permutation matrices (missense, synonymous) for one protein."""
    length = len(pd.pep_seq)
    codon_mis_probs = [x[1] for x in pd.codon_mutation_rates]
    codon_syn_probs = [x[0] for x in pd.codon_mutation_rates]
    # Same as husc/main.py (by design, confirmed): the missense null draws MAFs from the
    # synonymous (neutral) frequency spectrum. load_datasets() uses the empirical MAF lists
    # (use_af_dist defaults to True; the isdigit() parsing bug is fixed).
    mis = seq_utils.permute_variants_dist(
        pd.total_mis_exp, length, codon_mis_probs, pd.syn_dist
    )
    syn = seq_utils.permute_variants_dist(
        pd.total_syn_exp, length, codon_syn_probs, pd.syn_dist
    )
    return mis, syn


# --------------------------------------------------------------------------- #
# Chain <-> UniProt mapping
# --------------------------------------------------------------------------- #
@dataclass
class ChainInfo:
    chain_id: str
    protein: ProteinData
    offset: int
    perm_key: str
    residues: dict  # UniProt position -> Bio.PDB Residue (sequence-verified)


def parse_chain_spec(spec):
    """Parse 'CHAIN:UNIPROT[:OFFSET]' into (chain, uniprot, offset or None)."""
    parts = spec.strip().split(':')
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(f'bad chain spec {spec!r}; use CHAIN:UNIPROT[:OFFSET]')
    return parts[0], parts[1], int(parts[2]) if len(parts) == 3 else None


def parse_complex_line(line, search_dirs=('.',)):
    """
    Parse '<id> <structure> <spec> [<spec> ...]'.

    Each ``spec`` is either ``CHAIN:UNIPROT[:OFFSET]`` or the path of a COSMIS
    ``cosmis-complex``-style chain map file (one ``CHAIN:UNIPROT[:OFFSET]`` per
    line). Map files are looked up as given, then in ``search_dirs``.
    """
    fields = line.split()
    if len(fields) < 3:
        raise ValueError(f'expected at least 3 fields, got: {line!r}')
    complex_id, structure_file = fields[0], fields[1]
    chains = []
    for spec in fields[2:]:
        if ':' in spec:
            chains.append(parse_chain_spec(spec))
            continue
        candidates = [spec] + [os.path.join(d, spec) for d in search_dirs]
        map_file = next((c for c in candidates if os.path.isfile(c)), None)
        if map_file is None:
            raise ValueError(f'chain map file not found: {spec}')
        with open(map_file, 'rt') as f:
            chains.extend(
                parse_chain_spec(ln) for ln in f
                if ln.strip() and not ln.startswith('#')
            )
    seen = set()
    for chain_id, _, _ in chains:
        if chain_id in seen:
            raise ValueError(f'chain {chain_id} is mapped more than once')
        seen.add(chain_id)
    return complex_id, structure_file, chains


def chain_standard_residues(chain):
    """Standard amino acid residues of a chain that are not insertion-coded."""
    out = {}
    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag != ' ' or icode.strip() or not is_aa(res, standard=True):
            continue
        out.setdefault(resseq, res)
    return out


def infer_offset(pdb_residues, pep_seq, k=8):
    """
    Infer the constant offset (UniProt position = resseq + offset) of a chain.

    The first ``k`` consecutive modeled residues are located in the UniProt
    sequence; among the hits, the offset with the most matching residues wins.
    Returns None if no offset gives >= MIN_SEQ_MATCH agreement.
    """
    resseqs = sorted(pdb_residues)
    pdb_aa = {r: seq1(pdb_residues[r].get_resname()) for r in resseqs}

    def match_fraction(offset):
        n_match = sum(
            1 for r, aa in pdb_aa.items()
            if 1 <= r + offset <= len(pep_seq) and pep_seq[r + offset - 1] == aa
        )
        return n_match / len(pdb_aa)

    # offset 0 first: AlphaFold models and experimental structures already
    # numbered like UniProt
    if match_fraction(0) >= MIN_SEQ_MATCH:
        return 0

    # first run of k consecutive residue numbers
    run = []
    for r in resseqs:
        if run and r != run[-1] + 1:
            if len(run) >= k:
                break
            run = []
        run.append(r)
    if len(run) < k:
        return None
    kmer = ''.join(pdb_aa[r] for r in run[:k])
    candidates = []
    start = pep_seq.find(kmer)
    while start != -1:
        candidates.append((start + 1) - run[0])  # UniProt pos of run[0] - resseq
        start = pep_seq.find(kmer, start + 1)
    best = max(candidates, key=match_fraction, default=None)
    if best is not None and match_fraction(best) >= MIN_SEQ_MATCH:
        return best
    return None


def map_chain(chain, protein, offset, chain_id, uniprot_id):
    """
    Map a structure chain onto its UniProt sequence: returns (offset, residues)
    where residues is {UniProt position: Residue} restricted to residues whose
    amino acid agrees with the UniProt sequence.
    """
    pdb_residues = chain_standard_residues(chain)
    if not pdb_residues:
        raise SkipComplex(f'chain {chain_id} has no standard amino acid residues')
    pep_seq = protein.pep_seq
    if offset is None:
        offset = infer_offset(pdb_residues, pep_seq)
        if offset is None:
            raise SkipComplex(
                f'chain {chain_id} does not match {uniprot_id}; give an explicit '
                f'offset as {chain_id}:{uniprot_id}:<offset>'
            )
    residues, mismatches = {}, 0
    for resseq, res in pdb_residues.items():
        upos = resseq + offset
        if not 1 <= upos <= len(pep_seq) or seq1(res.get_resname()) != pep_seq[upos - 1]:
            mismatches += 1
            continue
        residues[upos] = res
    if len(residues) / len(pdb_residues) < MIN_SEQ_MATCH:
        raise SkipComplex(
            f'chain {chain_id} matches {uniprot_id} at only '
            f'{len(residues)}/{len(pdb_residues)} residues with offset {offset}'
        )
    if mismatches:
        logging.warning(
            f'Chain {chain_id} ({uniprot_id}): {mismatches} residues differ from '
            f'the UniProt sequence and are excluded'
        )
    return offset, residues


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def get_complex_headers():
    """Column names of the complex output file (superset of the monomer file)."""
    return [
        'complex_id', 'chain_id', 'uniprot_id', 'enst_id', 'uniprot_pos',
        'uniprot_aa', 'seq_separations', 'num_contacts', 'num_intra_contacts',
        'num_inter_contacts', 'inter_chain_contacts', 'syn_var_sites',
        'total_syn_sites', 'mis_var_sites', 'total_mis_sites', 'cs_syn_poss',
        'cs_mis_poss', 'cs_gc_content', 'cs_syn_prob', 'cs_syn_obs',
        'cs_mis_prob', 'cs_mis_obs', 'mis_pmt_mean', 'mis_pmt_sd', 'HuSC',
        'mis_p_value', 'syn_pmt_mean', 'syn_pmt_sd', 'syn_p_value',
        'enst_syn_obs', 'enst_mis_obs', 'enst_syn_exp', 'enst_mis_exp',
        'plddt', 'uniprot_length',
    ]


def score_complex(complex_id, structure, chain_specs, proteins, args, error_counts):
    """Compute HuSC scores for every residue of the requested chains."""
    model = structure[0]

    # ---- map chains to proteins ------------------------------------------- #
    chains = {}
    for chain_id, uniprot_id, offset in chain_specs:
        if chain_id not in model:
            logging.warning(f'{complex_id}: chain {chain_id} not in structure; skipped')
            error_counts['bad_chain'] += 1
            continue
        protein = proteins[uniprot_id]
        try:
            offset, residues = map_chain(model[chain_id], protein, offset,
                                         chain_id, uniprot_id)
        except SkipComplex as e:
            logging.warning(f'{complex_id}: {e}; chain skipped')
            error_counts['bad_chain'] += 1
            continue
        perm_key = uniprot_id if args.shared_permutations else chain_id
        chains[chain_id] = ChainInfo(chain_id, protein, offset, perm_key, residues)
        logging.info(
            f'{complex_id}: chain {chain_id} -> {uniprot_id} ({protein.enst_id}), '
            f'offset {offset}, {len(residues)} residues'
        )
    if not chains:
        raise SkipComplex('no chain could be mapped')

    to_score = list(chains)
    if args.chains:
        wanted = set(args.chains.split(','))
        to_score = [c for c in chains if c in wanted]

    # ---- contacts across all mapped chains -------------------------------- #
    # Only sequence-verified residues of mapped chains take part, so every
    # neighbor can be looked up in the data of its own protein.
    res_key = {}
    all_residues = []
    for ci in chains.values():
        for upos, res in ci.residues.items():
            res_key[id(res)] = (ci.chain_id, upos)
            all_residues.append(res)
    all_contacts = pdb_utils.search_for_all_contacts(all_residues, radius=args.radius)
    indexed_contacts = defaultdict(list)
    for c in all_contacts:
        a, b = c.get_res_a(), c.get_res_b()
        indexed_contacts[id(a)].append(res_key[id(b)])
        indexed_contacts[id(b)].append(res_key[id(a)])

    # ---- permutation matrices (one per chain, or per protein if shared) ---- #
    mis_matrices, syn_matrices = {}, {}
    for ci in chains.values():
        if ci.perm_key in mis_matrices:
            continue
        logging.info(f'{complex_id}: permuting variants for {ci.perm_key}')
        mis_matrices[ci.perm_key], syn_matrices[ci.perm_key] = make_permutations(ci.protein)

    # ---- score each residue ------------------------------------------------ #
    rows = []
    for chain_id in to_score:
        ci = chains[chain_id]
        pd = ci.protein
        for upos in range(1, len(pd.pep_seq) + 1):
            res = ci.residues.get(upos)
            if res is None:
                continue
            seq_aa = pd.pep_seq[upos - 1]
            neighbors = indexed_contacts[id(res)]
            intra = [j for c, j in neighbors if c == chain_id]
            inter = [(c, j) for c, j in neighbors if c != chain_id]
            hood = [(chain_id, upos)] + neighbors

            mis_var_sites = syn_var_sites = 0
            total_mis_sites = total_syn_sites = 0.0
            mis_obs = syn_obs = 0
            mis_poss = syn_poss = 0
            syn_rate = mis_rate = 0.0
            codons_by_chain = defaultdict(list)
            for c, j in hood:
                npd = chains[c].protein
                mis_var_sites += npd.site_var_mis.get(j, 0)
                syn_var_sites += npd.site_var_syn.get(j, 0)
                mis_obs += npd.mis_counts.get(j, 0)
                syn_obs += npd.syn_counts.get(j, 0)
                mis_poss += npd.all_cds_ns_counts[j - 1][0]
                syn_poss += npd.all_cds_ns_counts[j - 1][1]
                syn_rate += npd.codon_mutation_rates[j - 1][0]
                mis_rate += npd.codon_mutation_rates[j - 1][1]
                total_mis_sites += npd.cds_ns_sites[j - 1][0]
                total_syn_sites += npd.cds_ns_sites[j - 1][1]
                codons_by_chain[c].append(j)

            # GC content of the sequence context, pooled over all chains
            seq_context = ''.join(
                seq_utils.get_codon_seq_context(js, chains[c].protein.cds)
                for c, js in codons_by_chain.items()
            )
            if not seq_context:
                continue
            gc_fraction = seq_utils.gc_content(seq_context)

            sites = [(chains[c].perm_key, j) for c, j in hood]
            mis_mean, mis_sd, mis_p = seq_utils.get_permutation_stats_multi(
                mis_matrices, sites, mis_obs)
            syn_mean, syn_sd, syn_p = seq_utils.get_permutation_stats_multi(
                syn_matrices, sites, syn_obs)

            if mis_sd > 0:
                z = (mis_obs - mis_mean) / mis_sd
                husc = '{:.3f}'.format(np.sign(z) * np.log10(np.abs(z) + 1))
            else:
                husc = 'NA'

            rows.append({
                'complex_id': complex_id,
                'chain_id': chain_id,
                'uniprot_id': pd.uniprot_id,
                'enst_id': pd.enst_id,
                'uniprot_pos': upos,
                'uniprot_aa': seq_aa,
                'seq_separations': ';'.join(str(j - upos) for j in intra),
                'num_contacts': len(hood),
                'num_intra_contacts': len(intra),
                'num_inter_contacts': len(inter),
                'inter_chain_contacts': ';'.join(f'{c}:{j}' for c, j in inter),
                'syn_var_sites': syn_var_sites,
                'total_syn_sites': '{:.3f}'.format(total_syn_sites),
                'mis_var_sites': mis_var_sites,
                'total_mis_sites': '{:.3f}'.format(total_mis_sites),
                'cs_syn_poss': syn_poss,
                'cs_mis_poss': mis_poss,
                'cs_gc_content': '{:.3f}'.format(gc_fraction),
                'cs_syn_prob': '{:.3e}'.format(syn_rate),
                'cs_syn_obs': syn_obs,
                'cs_mis_prob': '{:.3e}'.format(mis_rate),
                'cs_mis_obs': mis_obs,
                'mis_pmt_mean': mis_mean,
                'mis_pmt_sd': mis_sd,
                'HuSC': husc,
                'mis_p_value': '{:.3e}'.format(mis_p),
                'syn_pmt_mean': '{:.3f}'.format(syn_mean),
                'syn_pmt_sd': '{:.3f}'.format(syn_sd),
                'syn_p_value': '{:.3e}'.format(syn_p),
                'enst_syn_obs': pd.enst_syn_obs,
                'enst_mis_obs': pd.enst_mis_obs,
                'enst_syn_exp': pd.total_syn_exp,
                'enst_mis_exp': pd.total_mis_exp,
                # pLDDT for AlphaFold models; B-factor for experimental structures
                'plddt': '{:.2f}'.format(res['CA'].get_bfactor()) if 'CA' in res else 'NA',
                'uniprot_length': len(pd.pep_seq),
            })
    return rows


def main():
    args = parse_cmd()
    configs = parse_config(args.config)
    logging.basicConfig(
        filename=args.log, level=logging.INFO, filemode='w',
        format='%(levelname)s:%(asctime)s:%(message)s'
    )
    if args.seed is not None:
        np.random.seed(args.seed)

    output_dir = os.path.abspath(configs['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    cds_dict, variant_dict, uniprot_to_enst, enst_mp_counts = load_datasets(configs)
    uniprot_seqs = load_uniprot_sequences(configs)

    with open(args.input, 'rt') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]

    error_counts = defaultdict(int)
    proteins = {}       # UniProt ID -> ProteinData (shared across complexes)
    failed_proteins = set()
    header = get_complex_headers()

    for line in lines:
        try:
            complex_id, structure_file, chain_specs = parse_complex_line(
                line, search_dirs=(os.path.dirname(os.path.abspath(args.input)),
                                   configs['pdb_dir'])
            )
        except ValueError as e:
            logging.error(f'Bad input line: {e}')
            error_counts['bad_input'] += 1
            continue
        logging.info(f'Processing complex {complex_id} ({structure_file})')

        output_file = os.path.join(output_dir, f'{complex_id}_husc_complex.tsv')
        if os.path.exists(output_file) and not args.overwrite:
            logging.info(f'{output_file} exists; skipping')
            continue

        # protein-level data, built once per UniProt ID
        usable = []
        for chain_id, uniprot_id, offset in chain_specs:
            if uniprot_id not in proteins and uniprot_id not in failed_proteins:
                try:
                    if uniprot_id not in uniprot_seqs:
                        raise SkipComplex(f'{uniprot_id} not in UniProt FASTA')
                    proteins[uniprot_id] = build_protein_data(
                        uniprot_id, uniprot_seqs[uniprot_id], cds_dict,
                        variant_dict, uniprot_to_enst, enst_mp_counts)
                except SkipComplex as e:
                    logging.error(f'{complex_id}: {e}')
                    failed_proteins.add(uniprot_id)
                    error_counts['no_protein_data'] += 1
            if uniprot_id in proteins:
                usable.append((chain_id, uniprot_id, offset))

        path = structure_file
        if not os.path.exists(path):
            path = os.path.join(configs['pdb_dir'], structure_file)
        structure = pdb_utils.load_structure(path, complex_id)
        if structure is None:
            logging.error(f'{complex_id}: cannot read structure {path}')
            error_counts['bad_structure'] += 1
            continue

        try:
            rows = score_complex(complex_id, structure, usable, proteins, args,
                                 error_counts)
        except SkipComplex as e:
            logging.error(f'{complex_id}: {e}')
            error_counts['bad_structure'] += 1
            continue

        if not rows:
            logging.error(f'{complex_id}: no residues scored; nothing written')
            error_counts['no_scores'] += 1
            continue

        with open(output_file, 'wt') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=header, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f'{complex_id}: wrote {len(rows)} residues to {output_file}')

    logging.info('HuSC complex computation completed.')
    for k, v in error_counts.items():
        logging.info(f'{k}: {v}')


if __name__ == '__main__':
    main()
