#!/usr/bin/env python3
"""
HuSC (Human Spatial Constraint) Score Computation Pipeline

This script computes HuSC scores for protein residues by integrating:
- 3D structural information from AlphaFold models
- Population genetic variation data from gnomAD
- Mutation rate estimates from coding sequences

HuSC scores quantify the constraint on amino acid sites based on their
spatial context and observed vs. expected missense variant frequencies.
"""

import csv
import gzip
import json
import os
import logging
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# BioPython imports
from Bio import SeqIO
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from Bio.PDB import PPBuilder

# Local imports
from husc.utils import seq_utils, pdb_utils

warnings.simplefilter('ignore', BiopythonWarning)


def parse_cmd():
    """
    Parse command-line arguments for computing HuSC scores.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including config file, input file, radius, database,
        overwrite flag, verbosity, and log file.
    """
    parser = ArgumentParser(
        description='Compute HuSC (Human Spatial Constraint) scores for protein residues'
    )
    
    parser.add_argument(
        '-c', '--config', dest='config', required=True,
        type=str, help='JSON file specifying configuration options'
    )
    parser.add_argument(
        '-i', '--input', dest='input', required=True,
        type=str, help='Input file with UniProt ID and PDB file pairs (e.g., "P13929 AF-P13929-F1-model_v4")'
    ) 
    parser.add_argument(
        '-w', '--overwrite', dest='overwrite', required=False,
        action='store_true', help='Overwrite existing HuSC score files'
    )
    parser.add_argument(
        '-r', '--radius', dest='radius', type=float, default=8.0,
        help='Radius (in Angstroms) for defining residue contacts (default: 8.0)'
    )
    parser.add_argument(
        '-v', '--verbose', dest='verbose', required=False,
        action='store_true', help='Output verbose data including contact counts and neighborhood variants'
    )
    parser.add_argument(
        '-l', '--log', dest='log', default='husc.log',
        help='Log file path for detailed computation logs (default: husc.log)'
    )
    
    return parser.parse_args()


def get_ensembl_accession(record):
    """
    Extract Ensembl transcript accession from a SeqRecord.

    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        Sequence record with Ensembl ID

    Returns
    -------
    str
        Ensembl transcript accession (without version number)
    """
    parts = record.id.split('.')
    return parts[0]


def get_ccds_accession(record):
    """
    Extract CCDS accession from a SeqRecord.

    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        Sequence record with CCDS ID

    Returns
    -------
    str
        CCDS accession
    """
    parts = record.id.split('|')
    return parts[0]


def get_uniprot_accession(record):
    """
    Extract UniProt accession from a SeqRecord.

    Parameters
    ----------
    record : Bio.SeqRecord.SeqRecord
        Sequence record with UniProt ID

    Returns
    -------
    str
        UniProt accession
    """
    parts = record.id.split('|')
    return parts[1]


def get_pdb_chain(pdb_file, pdb_chain):
    """
    Load a specific chain from a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file (may be gzipped)
    pdb_chain : str
        Chain identifier (e.g., 'A')

    Returns
    -------
    Bio.PDB.Chain.Chain or None
        The requested PDB chain object, or None if not found
    """
    pdb_parser = PDBParser(PERMISSIVE=1)
    
    try:
        if pdb_file.endswith('.gz'):
            with gzip.open(pdb_file, 'rt') as file:
                structure = pdb_parser.get_structure(id="given_pdb", file=file)
        else:
            structure = pdb_parser.get_structure(id="given_pdb", file=pdb_file)
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f'PDB file cannot be retrieved: {pdb_file}')
        return None
    
    try:
        chain = structure[0][pdb_chain]
    except KeyError:
        print(f'Chain {pdb_chain} not found in {pdb_file}')
        return None
    
    return chain


def parse_config(config):
    """
    Load and validate configuration file.

    Parameters
    ----------
    config : str
        Path to JSON configuration file

    Returns
    -------
    dict
        Configuration dictionary with dataset paths and parameters
    """
    with open(config, 'rt') as ipf:
        configs = json.load(ipf)

    # TODO: Add sanity checks for required keys if needed
    return configs


def count_variants(variants):
    """
    Compute position-specific variant frequencies from gnomAD data.

    This function processes variant data to calculate minor allele frequencies
    (MAF) for both missense and synonymous variants at each position.

    Parameters
    ----------
    variants : list
        List of variant tuples: [(variant_string, allele_count, allele_number), ...]
        where variant_string is in format 'A123B' (wt_aa + position + mut_aa)

    Returns
    -------
    tuple of (dict, dict)
        Two dictionaries mapping position (int) to cumulative MAF:
        - missense_counts: MAF sum for missense variants
        - synonymous_counts: MAF sum for synonymous variants
    """
    missense_counts = defaultdict(int)
    synonymous_counts = defaultdict(int)
    
    for variant in variants:
        variant_str = variant[0]
        allele_count = int(variant[1])
        allele_number = int(variant[2])

        # Calculate minor allele frequency
        maf = allele_count / allele_number
        if maf > 0.5:
            maf = 1 - maf
                        
        wt_aa = variant_str[0]  # Wild-type amino acid
        mut_aa = variant_str[-1]  # Mutant amino acid
        position = variant_str[1:-1]  # Position in the protein sequence

        if wt_aa != mut_aa:  # Missense variant
            missense_counts[int(position)] += maf
        else:  # Synonymous variant
            synonymous_counts[int(position)] += maf
    
    return missense_counts, synonymous_counts


def retrieve_data(pep_seq, uniprot_id, enst_ids, cds_dict, variant_dict):
    """
    Retrieve coding sequence and variant data for a UniProt protein.

    This function selects the most appropriate Ensembl transcript based on
    sequence compatibility and variant data availability.

    Parameters
    ----------
    pep_seq : str
        Protein sequence from the PDB structure
    uniprot_id : str
        UniProt accession identifier
    enst_ids : list of str
        Ensembl transcript IDs mapped to this UniProt ID
    cds_dict : dict
        Dictionary mapping transcript IDs to CDS SeqRecords
    variant_dict : dict
        Dictionary mapping transcript IDs to gnomAD variant data

    Returns
    -------
    tuple of (str, str, list)
        - Selected Ensembl transcript ID
        - Coding DNA sequence
        - List of variants for the selected transcript

    Raises
    ------
    ValueError
        If no valid transcripts are compatible with the protein sequence
    KeyError
        If no variant data is available for any valid transcript
    """
    # Filter transcripts with valid CDS
    valid_transcripts = []
    for enst_id in enst_ids:
        cds_seq = cds_dict[enst_id].seq
        
        if not seq_utils.is_valid_cds(cds_seq):
            print(f"Warning: Invalid CDS for transcript {enst_id}. Skipping.")
            continue
        
        # Check if protein and CDS lengths are compatible
        if len(pep_seq) == len(cds_seq) // 3 - 1:
            valid_transcripts.append(enst_id)

    if not valid_transcripts:
        raise ValueError(
            f"No valid transcripts compatible with UniProt ID {uniprot_id}: {enst_ids}"
        )

    # If only one valid transcript, use it
    if len(valid_transcripts) == 1:
        enst_id = valid_transcripts[0]
        cds_seq = cds_dict[enst_id].seq
        
        if enst_id not in variant_dict:
            raise KeyError(f"No variant data for {uniprot_id} in gnomAD")
        
        variants = variant_dict[enst_id]['variants']
        return enst_id, cds_seq, variants

    # If multiple valid transcripts, select the one with most variants
    selected_transcript = None
    max_variants = -1
    
    for enst_id in valid_transcripts:
        num_variants = len(variant_dict.get(enst_id, {}).get('variants', []))
        if num_variants > max_variants:
            max_variants = num_variants
            selected_transcript = enst_id

    if selected_transcript is None:
        raise KeyError(
            f"None of the valid transcripts have variant data for UniProt ID {uniprot_id}"
        )

    cds_seq = cds_dict[selected_transcript].seq
    variants = variant_dict[selected_transcript]['variants']

    return selected_transcript, cds_seq, variants


def get_dataset_headers():
    """
    Return column headers for the HuSC output file.

    Returns
    -------
    list of str
        Column names for the tab-separated output file
    
    Notes
    -----
    Update this function when adding new features to the output.
    """
    header = [
        'uniprot_id', 'enst_id', 'uniprot_pos', 'uniprot_aa',
        'seq_separations', 'num_contacts', 'syn_var_sites',
        'total_syn_sites', 'mis_var_sites', 'total_mis_sites',
        'cs_syn_poss', 'cs_mis_poss', 'cs_gc_content', 'cs_syn_prob',
        'cs_syn_obs', 'cs_mis_prob', 'cs_mis_obs', 'mis_pmt_mean', 
        'mis_pmt_sd', 'HuSC', 'mis_p_value', 'syn_pmt_mean', 
        'syn_pmt_sd', 'syn_p_value', 'enst_syn_obs', 'enst_mis_obs', 
        'enst_syn_exp', 'enst_mis_exp', 'plddt', 'uniprot_length'
    ]
    return header


def load_datasets(configs):
    """
    Load all required datasets for HuSC score computation.

    Parameters
    ----------
    configs : dict
        Configuration dictionary containing file paths:
        - 'ensembl_cds': Ensembl CDS FASTA file (gzipped)
        - 'gnomad_variants': gnomAD transcript-level variants (JSON)
        - 'uniprot_to_enst': UniProt-to-Ensembl mapping (JSON)
        - 'enst_mp_counts': Transcript mutation probabilities and counts

    Returns
    -------
    tuple
        (enst_cds_dict, enst_variants, uniprot_to_enst, enst_mp_counts)
        - enst_cds_dict: dict mapping transcript IDs to CDS sequences
        - enst_variants: dict mapping transcript IDs to variant lists
        - uniprot_to_enst: dict mapping UniProt IDs to Ensembl IDs
        - enst_mp_counts: dict with mutation probability distributions
    """
    # Load Ensembl CDS sequences
    print('Loading Ensembl CDS sequences...')
    with gzip.open(configs['ensembl_cds'], 'rt') as cds_handle:
        enst_cds_dict = SeqIO.to_dict(
            SeqIO.parse(cds_handle, format='fasta'),
            key_function=get_ensembl_accession
        )

    # Load gnomAD transcript-level variants
    print('Loading gnomAD variant database...')
    with open(configs['gnomad_variants'], 'rt') as variant_handle:
        enst_variants = json.load(variant_handle)

    # Load UniProt-to-Ensembl transcript mapping
    print('Loading UniProt-to-Ensembl mapping...')
    with open(configs['uniprot_to_enst'], 'rt') as map_handle:
        uniprot_to_enst = json.load(map_handle)

    # Load transcript mutation probabilities and variant counts
    print('Loading transcript mutation probabilities and variant counts...')
    # 'use_af_dist': false in the config reproduces the published HuSC scores,
    # which used a constant allele frequency for the permutations (see README)
    enst_mp_counts = seq_utils.read_enst_mp_count_dist(
        configs['enst_mp_counts'], use_af_dist=configs.get('use_af_dist', True)
    )

    return enst_cds_dict, enst_variants, uniprot_to_enst, enst_mp_counts


def get_pep_seq_from_pdb(pdb_file, pdb_chain):
    """
    Extract protein sequence from a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file (may be gzipped)
    pdb_chain : str
        Chain identifier

    Returns
    -------
    Bio.Seq.Seq or None
        Protein sequence from the PDB structure, or None if extraction fails
    """
    if pdb_file.endswith('.gz'):
        with gzip.open(pdb_file, 'rt') as f:
            structure = PDBParser().get_structure('protein', f)
    else:
        structure = PDBParser().get_structure('protein', pdb_file)
    
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        if pp.get_sequence():
            return pp.get_sequence()
    
    return None


def main():
    """
    Main pipeline for computing HuSC (Human Spatial Constraint) scores.

    This pipeline integrates structural, sequence, and variant data to compute
    constraint scores for each residue in a protein. The workflow includes:
    
    1. Parse command-line arguments and load configuration
    2. Load reference datasets (CDS, variants, mappings)
    3. For each UniProt–AlphaFold pair:
       a. Extract protein sequence from structure
       b. Map to appropriate Ensembl transcript
       c. Identify spatial contacts within specified radius
       d. Compute observed and expected variant counts
       e. Calculate HuSC scores via permutation analysis
    4. Write results to tab-separated files

    Output
    ------
    Creates <uniprot_id>_husc.tsv files in the configured output directory.
    Each file contains per-residue HuSC scores and associated statistics.
    """
    args = parse_cmd()
    configs = parse_config(args.config)

    # Configure logging
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        filemode='w',
        format='%(levelname)s:%(asctime)s:%(message)s'
    )

    logging.info("Starting HuSC computation pipeline...")

    # Create output directory
    output_dir = os.path.abspath(configs['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Load all required datasets
    cds_dict, variant_dict, uniprot_to_enst, enst_mp_counts = load_datasets(configs)
    
    # Load UniProt-PDB pairs from input file
    with open(args.input, 'rt') as mapping_file:
        uniprot_pdb_pairs = [line.strip().split() for line in mapping_file]
    
    # Initialize error tracking
    error_counts = {
        "no_enst_for_uniprot": 0,
        "no_valid_cds": 0,
        "no_variant_data": 0,
        "bad_structure": 0,
        "no_enst_mp": 0,
    }

    # Process each UniProt-PDB pair
    for uniprot_id, pdb_model_name in uniprot_pdb_pairs:
        logging.info(f"Processing {uniprot_id} with model {pdb_model_name}")

        # Determine PDB file path and chain
        pdb_file = os.path.join(configs['pdb_dir'], pdb_model_name) + '.gz'
        pdb_chain = 'A'
        pep_seq = get_pep_seq_from_pdb(pdb_file, pdb_chain)

        # Skip if output exists and overwrite not requested
        output_file = os.path.join(output_dir, f"{uniprot_id}_husc.tsv")
        if os.path.exists(output_file) and not args.overwrite:
            logging.info(f"{uniprot_id}_husc.tsv already exists. Skipping.")
            continue

        husc_scores = []

        # Map UniProt ID to Ensembl transcripts
        try:
            enst_ids = uniprot_to_enst[uniprot_id]
        except KeyError:
            logging.error(f"No Ensembl transcript IDs mapped to {uniprot_id}") 
            error_counts['no_enst_for_uniprot'] += 1
            continue

        # Retrieve CDS and variant data
        try:
            right_enst, cds, variants = retrieve_data(
                pep_seq, uniprot_id, enst_ids, cds_dict, variant_dict
            )
        except ValueError:
            logging.error(f"No valid CDS found for {uniprot_id}")
            error_counts['no_valid_cds'] += 1
            continue
        except KeyError:
            logging.error(f"No gnomAD variant data for {uniprot_id}")
            error_counts['no_variant_data'] += 1
            continue

        logging.info(f"Computing HuSC features for: {uniprot_id}, {right_enst}, {pdb_file}")

        # Load PDB chain structure
        chain = get_pdb_chain(pdb_file, pdb_chain)
        if chain is None:
            logging.warning(f"Missing chain {pdb_chain} in {pdb_file}. Skipping.")
            error_counts["bad_structure"] += 1
            continue

        # Extract amino acid residues
        all_aa_residues = [aa for aa in chain.get_residues() if is_aa(aa, standard=True)]
            
        # Check structure quality
        if not all_aa_residues or len(all_aa_residues) / len(pep_seq) < 1.0 / 3.0:
            logging.warning(f"Low-quality structure for {uniprot_id} ({pdb_model_name}). Skipping.")
            error_counts["bad_structure"] += 1
            continue
        
        # Identify all residue contacts within specified radius
        all_contacts = pdb_utils.search_for_all_contacts(all_aa_residues, radius=args.radius)

        # Index contacts by residue for efficient lookup
        indexed_contacts = defaultdict(list)
        for c in all_contacts:
            indexed_contacts[c.get_res_a()].append(c.get_res_b())
            indexed_contacts[c.get_res_b()].append(c.get_res_a())

        # Remove stop codon from CDS
        cds = cds[:-3]
        
        # Compute mutation rates and variant counts
        codon_mutation_rates = seq_utils.get_codon_mutation_rates(cds)
        all_cds_ns_counts = seq_utils.count_poss_ns_variants(cds)
        cds_ns_sites = seq_utils.count_ns_sites(cds)

        # Verify sequence-structure correspondence
        if len(codon_mutation_rates) < len(all_aa_residues):
            logging.warning(f"Residue-sequence length mismatch for {uniprot_id}. Skipping.")
            error_counts["bad_structure"] += 1
            continue

        # Count observed variants
        mis_counts, syn_counts = count_variants(variants)
        site_var_mis = {pos: 1 for pos in mis_counts}
        site_var_syn = {pos: 1 for pos in syn_counts}

        # Retrieve transcript-level mutation statistics
        try:
            total_mis_exp = enst_mp_counts[right_enst][-3]
            total_syn_exp = enst_mp_counts[right_enst][-4]
            mis_dist = enst_mp_counts[right_enst][-1]
            syn_dist = enst_mp_counts[right_enst][-2]
        except KeyError:
            logging.error(f"Missing mutation probability data for {right_enst}")
            error_counts["no_enst_mp"] += 1
            continue
        
        # Generate permutation distributions for statistical testing
        try:
            codon_mis_probs = [x[1] for x in codon_mutation_rates]
            codon_syn_probs = [x[0] for x in codon_mutation_rates]
            # The missense null draws MAFs from the *synonymous* spectrum (syn_dist), by design (confirmed):
            # synonymous variation is the neutral proxy, consistent with the expected
            # missense count, which is also predicted from the synonymous regression.
            mis_pmt_matrix = seq_utils.permute_variants_dist(
                total_mis_exp, len(pep_seq), codon_mis_probs, syn_dist
            )
            syn_pmt_matrix = seq_utils.permute_variants_dist(
                total_syn_exp, len(pep_seq), codon_syn_probs, syn_dist
            )
        except ValueError:
            logging.error(f'Protein length mismatch during permutation for {uniprot_id}')
            continue

        # Compute HuSC scores for each residue
        valid_case = True
        for seq_pos, seq_aa in enumerate(pep_seq, start=1):
            # Get corresponding residue from structure
            try:
                res = chain[seq_pos]
            except KeyError:
                continue
            
            # Verify sequence-structure consistency
            if seq1(res.get_resname()) != seq_aa:
                logging.warning(f"Residue mismatch at position {seq_pos} in {uniprot_id}. Skipping protein.")
                valid_case = False
                break
                
            # Extract pLDDT confidence score
            plddt = res['CA'].get_bfactor()

            # Get contacting residues
            contact_res = indexed_contacts[res]
            num_contacts = len(contact_res)
            contacts_pdb_pos = [r.get_id()[1] for r in contact_res]
            seq_seps = ';'.join(
                str(x) for x in [i - seq_pos for i in contacts_pdb_pos]
            )

            # Initialize variant counts for focal residue
            mis_var_sites = site_var_mis.setdefault(seq_pos, 0)
            total_mis_sites = cds_ns_sites[seq_pos - 1][0]
            syn_var_sites = site_var_syn.setdefault(seq_pos, 0)
            total_syn_sites = cds_ns_sites[seq_pos - 1][1]
            total_missense_obs = mis_counts.setdefault(seq_pos, 0)
            total_synonymous_obs = syn_counts.setdefault(seq_pos, 0)
            total_missense_poss = all_cds_ns_counts[seq_pos - 1][0]
            total_synonyms_poss = all_cds_ns_counts[seq_pos - 1][1]
            total_synonymous_rate = codon_mutation_rates[seq_pos - 1][0]
            total_missense_rate = codon_mutation_rates[seq_pos - 1][1]
            
            # Accumulate counts across spatial neighborhood
            for j in contacts_pdb_pos:
                # Count observed variants in contacting residues
                mis_var_sites += site_var_mis.setdefault(j, 0)
                syn_var_sites += site_var_syn.setdefault(j, 0)
                total_missense_obs += mis_counts.setdefault(j, 0)
                total_synonymous_obs += syn_counts.setdefault(j, 0)

                # Count expected variants
                try:
                    total_missense_poss += all_cds_ns_counts[j - 1][0]
                    total_synonyms_poss += all_cds_ns_counts[j - 1][1]
                    total_synonymous_rate += codon_mutation_rates[j - 1][0]
                    total_missense_rate += codon_mutation_rates[j - 1][1]
                    total_mis_sites += cds_ns_sites[j - 1][0]
                    total_syn_sites += cds_ns_sites[j - 1][1]
                except IndexError:
                    valid_case = False
                    break
            
            if not valid_case:
                break

            # Extract sequence context for GC content calculation
            try:
                seq_context = seq_utils.get_codon_seq_context(
                    contacts_pdb_pos + [seq_pos], cds
                )
            except IndexError:
                break

            # Compute GC content of sequence context
            if len(seq_context) == 0:
                print('No nucleotides found in sequence context!')
                continue
            gc_fraction = seq_utils.gc_content(seq_context)

            # Perform permutation test for missense variants
            mis_pmt_mean, mis_pmt_sd, mis_p_value = seq_utils.get_permutation_stats(
                mis_pmt_matrix, contacts_pdb_pos + [seq_pos], total_missense_obs
            )
            
            # Perform permutation test for synonymous variants
            syn_pmt_mean, syn_pmt_sd, syn_p_value = seq_utils.get_permutation_stats(
                syn_pmt_matrix, contacts_pdb_pos + [seq_pos], total_synonymous_obs
            )

            # Calculate HuSC score
            z_score = (total_missense_obs - mis_pmt_mean) / mis_pmt_sd
            husc = np.sign(z_score) * np.log10(np.abs(z_score) + 1)

            # Append results for this residue
            husc_scores.append(
                [
                    uniprot_id, right_enst, seq_pos, seq_aa, seq_seps,
                    num_contacts + 1,
                    syn_var_sites,
                    '{:.3f}'.format(total_syn_sites),
                    mis_var_sites,
                    '{:.3f}'.format(total_mis_sites),
                    total_synonyms_poss,
                    total_missense_poss,
                    '{:.3f}'.format(gc_fraction),
                    '{:.3e}'.format(total_synonymous_rate),
                    total_synonymous_obs,
                    '{:.3e}'.format(total_missense_rate),
                    total_missense_obs,
                    mis_pmt_mean,
                    mis_pmt_sd,
                    '{:.3f}'.format(husc),  
                    '{:.3e}'.format(mis_p_value),
                    '{:.3f}'.format(syn_pmt_mean),
                    '{:.3f}'.format(syn_pmt_sd),
                    '{:.3e}'.format(syn_p_value),
                    enst_mp_counts[right_enst][2],
                    enst_mp_counts[right_enst][4],
                    total_syn_exp,
                    total_mis_exp,
                    '{:.2f}'.format(plddt),
                    len(pep_seq)
                ]
            )

        if not valid_case:
            continue

        # Write HuSC scores to output file
        with open(output_file, "wt") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(get_dataset_headers())
            writer.writerows(husc_scores)

        logging.info(f"Successfully computed HuSC scores for {uniprot_id}")

    # Log final error summary
    logging.info("HuSC computation pipeline completed.")
    for error_type, count in error_counts.items():
        logging.info(f"{error_type}: {count}")


if __name__ == '__main__':
    main()