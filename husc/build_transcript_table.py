#!/usr/bin/env python3
"""
Build the per-transcript table (config['enst_mp_counts']) used by the HuSC null:

    enst_id  length  syn_prob  syn_count  mis_prob  mis_count  syn_dist  mis_dist  syn_exp  mis_exp

for every Ensembl transcript with a valid CDS (ATG ... stop, length a multiple
of 3) and at least one gnomAD variant (optionally restricted to a list of
transcript IDs; the published table used resources/enst_ids.txt):

- syn_prob, mis_prob: summed trinucleotide-context mutation rates of all
  possible synonymous / missense SNVs of the CDS
- syn_count, mis_count: observed gnomAD synonymous / missense variants
- syn_dist, mis_dist: their allele frequencies in percent (AC / AN * 100),
  comma-separated in the order of the variants file
- syn_exp, mis_exp: expected counts from one least-squares line of syn_count on
  syn_prob across all transcripts, applied to syn_prob and to mis_prob

Usage:
    python -m husc.build_transcript_table -c config.json [-t resources/enst_ids.txt]
"""

import os
from argparse import ArgumentParser

import ijson
import numpy as np

from husc import data
from husc.utils import seq_utils

HEADER = ['enst_id', 'length', 'syn_prob', 'syn_count', 'mis_prob', 'mis_count',
          'syn_dist', 'mis_dist', 'syn_exp', 'mis_exp']


def parse_cmd():
    parser = ArgumentParser(description='Build the per-transcript mutability / variant count table for HuSC')
    parser.add_argument('-c', '--config', required=True,
                        help='HuSC JSON configuration file; the table is written to its enst_mp_counts path')
    parser.add_argument('-t', '--transcripts',
                        help='Optional file of Ensembl transcript IDs (one per line) to restrict the table to')
    return parser.parse_args()


def transcript_row(cds, variants):
    """Table row (without the expected counts) for one transcript, or None if its CDS is invalid."""
    if not seq_utils.is_valid_cds(cds):
        return None
    rates = seq_utils.get_codon_mutation_rates(cds)
    syn_prob = sum(r[0] for r in rates)
    mis_prob = sum(r[1] for r in rates)
    syn_af, mis_af = [], []
    for variant_str, allele_count, allele_number in variants:
        af = int(allele_count) / int(allele_number) * 100
        (syn_af if variant_str[0] == variant_str[-1] else mis_af).append(repr(af))
    return [len(cds) // 3 - 1, '{:.3e}'.format(syn_prob), len(syn_af),
            '{:.3e}'.format(mis_prob), len(mis_af), ','.join(syn_af), ','.join(mis_af)]


def main():
    args = parse_cmd()
    config = data.load_config(args.config)
    out_path = config['enst_mp_counts']
    tmp_path = out_path + '.tmp'
    cds = data.load_cds(config['ensembl_cds'])
    keep = None
    if args.transcripts:
        with open(args.transcripts) as handle:
            keep = {line.strip().split('.')[0] for line in handle if line.strip()}

    # Pass 1: mutability, counts and AF lists, streamed to a temporary file
    syn_prob, syn_count = [], []
    with data.open_maybe_gzip(config['gnomad_variants'], 'rb') as handle, open(tmp_path, 'wt') as tmp:
        for enst_id, record in ijson.kvitems(handle, ''):
            if not record.get('variants') or enst_id not in cds or (keep is not None and enst_id not in keep):
                continue
            row = transcript_row(cds[enst_id], record['variants'])
            if row is None:
                continue
            tmp.write('\t'.join(str(x) for x in [enst_id] + row) + '\n')
            syn_prob.append(float(row[1]))
            syn_count.append(row[2])

    # Expected counts: least-squares line of synonymous count on synonymous mutability
    x = np.vstack([syn_prob, np.ones(len(syn_prob))]).T
    slope, intercept = (float(v) for v in np.linalg.lstsq(x, np.array(syn_count, dtype=float), rcond=None)[0])
    print(f'{len(syn_prob)} transcripts; expected count = {slope:.4e} * mutability + {intercept:.4f}')

    # Pass 2: add the expected counts
    with open(tmp_path, 'rt') as tmp, data.open_maybe_gzip(out_path, 'wt') as out:
        out.write('\t'.join(HEADER) + '\n')
        for line in tmp:
            row = line.rstrip('\n').split('\t')
            row += [repr(slope * float(row[2]) + intercept), repr(slope * float(row[4]) + intercept)]
            out.write('\t'.join(row) + '\n')
    os.remove(tmp_path)
    print(f'Table written to {out_path}')


if __name__ == '__main__':
    main()
