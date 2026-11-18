#!/usr/bin/env python3
"""
Build the two CpG inputs of the HuSC null model from gnomAD synonymous variation:

1. config['cpg_maf']: pooled synonymous MAFs (percent, one per line) at codons
   whose possible synonymous SNVs are all CpG transitions. Placed CpG SNVs draw
   their MAF from this list, because recurrent mutation raises CpG allele
   frequencies above those of a typical gene's synonymous variants.

2. config['cpg_scale']: CpG SNV rate factors by local CpG o/e (+-100 bp of CDS),
   a proxy for unmethylated CpG islands. Factors are fit so that, at all-CpG
   codons in each o/e bin, the observed fraction with >= 1 synonymous variant
   equals the saturating null's prediction (beta re-solved per gene after
   scaling, so each gene's expected total is unchanged).

Only synonymous variation is used. Genes are the proteins of the input file
(one transcript each, chosen as in the pipeline but with the UniProt sequence
length). Fit quality is reported by 2-fold cross-validation over genes; the
final factors are fit on all genes.

Usage:
    python -m husc.prepare_cpg -c config.json -i input_pairs.txt
"""

import json
import zlib
from argparse import ArgumentParser

import numpy as np

from husc import data, null_model

# Bin edges of local CpG o/e (the last bin is open-ended)
CPG_OE_EDGES = [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1e9]
FIT_ITERATIONS = 8


def parse_cmd():
    parser = ArgumentParser(description='Build the pooled CpG MAF list and CpG rate factors for HuSC')
    parser.add_argument('-c', '--config', required=True, help='HuSC JSON configuration file')
    parser.add_argument('-i', '--input', required=True, help='UniProt ID / model pairs defining the gene set')
    return parser.parse_args()


def gene_fit_data(cds, variants, stats):
    """Per-gene arrays for the CpG factor fit, or None if the gene has no all-CpG codon."""
    n = len(cds) // 3
    codons, rates = null_model.get_snv_mutation_rates(cds)['syn']
    is_cpg, _ = null_model.codon_cpg_class(cds)
    pure = np.where(is_cpg)[0]
    if not len(pure):
        return None
    has_syn = np.zeros(n, bool)
    syn_pos = variants.pos[(variants.ref == variants.alt) & (variants.pos <= n)]
    has_syn[syn_pos - 1] = True
    syn_exp, _ = null_model.calibrated_expected(stats.syn_obs, stats.syn_exp, stats.mis_exp)
    oe = null_model.local_cpg_oe(cds)
    return dict(codons=codons, rates=rates, n=n, expected=syn_exp, oe_of_snv=oe[codons],
                pure=pure, observed=has_syn[pure], pure_oe=oe[pure])


def observed_predicted(genes, factors):
    """Observed and predicted counts of all-CpG codons with >= 1 synonymous variant, per o/e bin."""
    scale = {'edges': CPG_OE_EDGES, 'factors': factors}
    n_bins = len(factors)
    obs, pred, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for g in genes:
        snv_factor = np.asarray(factors)[null_model.cpg_bins(g['oe_of_snv'], scale)]
        rates = np.where(g['rates'] > null_model.CPG_RATE, g['rates'] * snv_factor, g['rates'])
        p = null_model.saturating_snv_probs(rates, g['expected'])
        p_none = np.ones(g['n'])
        np.multiply.at(p_none, g['codons'], 1 - p)
        b = null_model.cpg_bins(g['pure_oe'], scale)
        np.add.at(obs, b, g['observed'])
        np.add.at(pred, b, 1 - p_none[g['pure']])
        np.add.at(count, b, 1)
    return obs, pred, count


def fit_factors(genes):
    """Fixed-point fit of the per-bin rate factors so that observed/predicted = 1."""
    factors = np.ones(len(CPG_OE_EDGES) - 1)
    for _ in range(FIT_ITERATIONS):
        obs, pred, _ = observed_predicted(genes, factors)
        # rates move p less than 1:1 near saturation, so overstep a little
        factors = np.clip(factors * (obs / pred) ** 1.5, 0.05, 2.0)
    return factors


def fmt(values):
    return ' '.join(f'{x:.3f}' for x in values)


def main():
    args = parse_cmd()
    config = data.load_config(args.config)

    uniprot_ids = sorted({u for u, _ in data.read_input_pairs(args.input)})
    uniprot_to_enst = data.load_json(config['uniprot_to_enst'])
    uniprot_seqs = data.load_uniprot_seqs(config['uniprot_pep'])
    cds = data.load_cds(config['ensembl_cds'])
    enst_ids = {e for u in uniprot_ids for e in uniprot_to_enst.get(u, [])}
    variants = data.load_variants(config['gnomad_variants'], enst_ids)
    transcripts = data.read_transcript_table(config['enst_mp_counts'], enst_ids)

    selected = set()
    for u in uniprot_ids:
        if u not in uniprot_seqs or u not in uniprot_to_enst:
            continue
        try:
            selected.add(data.select_transcript(len(uniprot_seqs[u]), uniprot_to_enst[u], cds, variants))
        except (ValueError, KeyError):
            continue
    selected = sorted(e for e in selected if e in transcripts)
    print(f'{len(uniprot_ids)} proteins -> {len(selected)} transcripts with CDS, variants and expected counts')

    cpg_mafs, n_non_cpg, genes = [], 0, []
    for e in selected:
        seq = cds[e][:-3]
        non_cpg, cpg = null_model.syn_maf_by_cpg_class(variants[e], seq)
        cpg_mafs.append(cpg)
        n_non_cpg += len(non_cpg)
        g = gene_fit_data(seq, variants[e], transcripts[e])
        if g is not None:
            g['fold'] = zlib.crc32(e.encode()) % 2
            genes.append(g)
    cpg_mafs = np.concatenate(cpg_mafs)
    np.savetxt(config['cpg_maf'], cpg_mafs, fmt='%.6e')
    print(f'{len(cpg_mafs)} CpG-codon synonymous MAFs written to {config["cpg_maf"]} '
          f'({n_non_cpg} non-CpG synonymous MAFs stay in per-gene lists)')

    # 2-fold cross-validation over genes, then the final fit on all genes
    print(f'CpG factor fit on {len(genes)} genes with all-CpG codons; local CpG o/e bins {CPG_OE_EDGES[:-1]}')
    held_out = {}
    for fold in (0, 1):
        train = [g for g in genes if g['fold'] != fold]
        test = [g for g in genes if g['fold'] == fold]
        f = fit_factors(train)
        for name, ff in (('unscaled', np.ones_like(f)), ('scaled', f)):
            obs, pred, count = observed_predicted(test, ff)
            held_out.setdefault(name, [np.zeros_like(f), np.zeros_like(f), np.zeros_like(f)])
            for acc, x in zip(held_out[name], (obs, pred, count)):
                acc += x
        print(f'  fold {fold} factors: {fmt(f)}')
    for name, (obs, pred, count) in held_out.items():
        print(f'  held-out obs/pred, {name:8s}: {fmt(obs / pred)}   n={" ".join(str(int(c)) for c in count)}')

    factors = fit_factors(genes)
    obs, pred, count = observed_predicted(genes, factors)
    print(f'  final factors (all genes): {fmt(factors)}')
    print(f'  in-sample obs/pred:        {fmt(obs / pred)}')
    with open(config['cpg_scale'], 'w') as handle:
        json.dump({
            'edges': CPG_OE_EDGES,
            'factors': [round(float(x), 4) for x in factors],
            'n_genes': len(genes),
            'held_out_obs_pred_unscaled': [round(float(x), 3) for x in held_out['unscaled'][0] / held_out['unscaled'][1]],
            'held_out_obs_pred_scaled': [round(float(x), 3) for x in held_out['scaled'][0] / held_out['scaled'][1]],
            'note': 'CpG SNV rate factors by local CpG o/e (+-100 bp of CDS); built by husc.prepare_cpg',
        }, handle, indent=1)
    print(f'Factors written to {config["cpg_scale"]}')


if __name__ == '__main__':
    main()
