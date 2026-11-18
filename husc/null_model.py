"""
Permutation null model for HuSC.

For each transcript, the null places variants on the possible single-nucleotide
variants (SNVs) of the coding sequence and gives each placed variant a minor
allele frequency (MAF):

1. Expected counts. The transcript's expected missense and synonymous counts
   (from a regression on mutability) are multiplied by its synonymous
   observed/expected ratio, shrunk toward 1 with a pseudo-count CALIB_K.
2. Saturating placement. Each possible SNV is observed at most once, with
   probability p = 1 - exp(-beta * rate), where rate is its trinucleotide-context
   mutation rate and beta is solved per transcript so that the probabilities sum
   to the expected count.
3. CpG methylation. CpG transition rates are multiplied by a factor of the local
   CpG observed/expected ratio (+-100 bp of CDS), so CpG-island-like (mostly
   unmethylated) sites mutate more slowly. Factors are fit by husc.prepare_cpg.
4. MAFs. Each placed CpG SNV draws a MAF from a pooled list of synonymous MAFs at
   CpG codons (recurrent mutation raises CpG allele frequencies); other SNVs draw
   from the transcript's own non-CpG synonymous MAFs.

The test statistic is the MAF sum over a residue's contact set. Its p-value is
the mid-p probability of observing this sum or less under the null, so small p
means fewer or rarer variants than expected (constraint).
"""

import numpy as np

from husc.utils.genetic_code import GENETIC_CODE
from husc.utils.trinucleotide_context_rates import MUTATION_RATES_UNIQUE

# SNVs above this trinucleotide-context rate are CpG transitions (>=1.7e-7; all others <=3.2e-8)
CPG_RATE = 1e-7

# Pseudo-count shrinking the per-transcript synonymous o/e calibration toward 1
CALIB_K = 10

# Number of permutations per transcript
N_PERMUTATIONS = 10000

# Half-width (nt) of the CDS window used for the local CpG o/e ratio
CPG_WINDOW = 100


def calibrated_expected(syn_obs, syn_exp, mis_exp, k=CALIB_K):
    """
    Scale expected synonymous and missense counts by the transcript's shrunken
    synonymous observed/expected ratio, (syn_obs + k) / (syn_exp + k).

    Returns
    -------
    tuple of (int, int)
        Calibrated (syn_exp, mis_exp), rounded to whole variants.
    """
    factor = (syn_obs + k) / (syn_exp + k)
    return int(round(syn_exp * factor)), int(round(mis_exp * factor))


def get_snv_mutation_rates(cds):
    """
    List every possible SNV of the CDS with its codon and trinucleotide-context
    mutation rate, split into synonymous and missense (stop-gains are excluded).
    The first and last codons only use the nucleotides with a full trinucleotide
    context inside the CDS, as in seq_utils.get_codon_mutation_rates, so
    per-codon sums of these rates equal its output.

    Parameters
    ----------
    cds : str
        Coding sequence without the stop codon.

    Returns
    -------
    dict
        {'syn': (codon_indices, rates), 'mis': (codon_indices, rates)} with
        zero-based codon indices and rates as numpy arrays.
    """
    if len(cds) % 3 != 0:
        raise ValueError('Given CDS length is not a multiple of 3.')

    num_codons = len(cds) // 3
    snvs = {'syn': ([], []), 'mis': ([], [])}
    for codon_number in range(1, num_codons + 1):
        codon_sequence = cds[(codon_number - 1) * 3:codon_number * 3]
        if codon_number == 1:
            sequence_context = cds[:4]
            positions = [(i, sequence_context[i - 1:i + 2]) for i in range(1, 3)]
        elif codon_number == num_codons:
            sequence_context = cds[-4:]
            positions = [(i, sequence_context[i:i + 3]) for i in range(0, 2)]
        else:
            sequence_context = cds[(codon_number - 1) * 3 - 1:(codon_number - 1) * 3 + 4]
            positions = [(i, sequence_context[i:i + 3]) for i in range(3)]

        for i, trinucleotide in positions:
            for k, v in MUTATION_RATES_UNIQUE[trinucleotide].items():
                mutant_sequence = codon_sequence[:i] + k[1] + codon_sequence[i + 1:]
                if GENETIC_CODE[codon_sequence] == GENETIC_CODE[mutant_sequence]:
                    kind = 'syn'
                elif GENETIC_CODE[mutant_sequence] != 'STOP':
                    kind = 'mis'
                else:
                    continue
                snvs[kind][0].append(codon_number - 1)
                snvs[kind][1].append(v)
    return {k: (np.array(idx, dtype=int), np.array(r, dtype=float)) for k, (idx, r) in snvs.items()}


def saturating_snv_probs(rates, expected_total):
    """
    Probability that each possible SNV is observed, p = 1 - exp(-beta * rate),
    with beta found by bisection so that the probabilities sum to expected_total.
    Unlike a linear model, highly mutable SNVs saturate at 1.
    """
    rates = np.asarray(rates, dtype=float)
    if expected_total <= 0 or rates.sum() == 0:
        return np.zeros_like(rates)
    if expected_total >= np.count_nonzero(rates):
        return (rates > 0).astype(float)

    def total(beta):
        return np.sum(-np.expm1(-beta * rates))

    lo, hi = 0.0, expected_total / rates.sum()
    while total(hi) < expected_total:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if total(mid) < expected_total:
            lo = mid
        else:
            hi = mid
    return -np.expm1(-hi * rates)


def local_cpg_oe(cds, half_window=CPG_WINDOW):
    """
    CpG observed/expected ratio, CpG * L / (C * G), in a window of +-half_window
    nucleotides of CDS around each codon. High values mark CpG-island-like
    sequence, where CpGs are mostly unmethylated and mutate more slowly.
    """
    seq = str(cds).upper()
    n = len(seq) // 3
    cg = np.zeros(len(seq))
    cg[:-1] = [seq[i:i + 2] == 'CG' for i in range(len(seq) - 1)]
    cum = lambda a: np.concatenate([[0], np.cumsum(a)])
    CG = cum(cg)
    C = cum([ch == 'C' for ch in seq])
    G = cum([ch == 'G' for ch in seq])
    a = np.maximum(0, 3 * np.arange(n) - half_window)
    b = np.minimum(len(seq), 3 * np.arange(n) + 3 + half_window)
    return (CG[b] - CG[a]) * (b - a) / np.maximum((C[b] - C[a]) * (G[b] - G[a]), 1)


def cpg_bins(oe, cpg_scale):
    """Index of the local CpG o/e bin of each value, given cpg_scale['edges']."""
    edges = np.asarray(cpg_scale['edges'])
    return np.clip(np.searchsorted(edges, oe, side='right') - 1, 0, len(cpg_scale['factors']) - 1)


def scale_cpg_rates(snv_codons, snv_rates, cds, cpg_scale):
    """
    Multiply CpG SNV rates (rate > CPG_RATE) by the factor of the local CpG o/e
    bin of their codon. cpg_scale is a dict with bin 'edges' (len B+1) and
    'factors' (len B).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Scaled rates and the CpG mask (taken from the unscaled rates).
    """
    rates = np.asarray(snv_rates, dtype=float)
    is_cpg = rates > CPG_RATE
    factors = np.asarray(cpg_scale['factors'])[cpg_bins(local_cpg_oe(cds)[snv_codons], cpg_scale)]
    return np.where(is_cpg, rates * factors, rates), is_cpg


def codon_cpg_class(cds):
    """
    Classify codons by their possible synonymous SNVs: 'cpg' if all of them are
    CpG transitions, 'non_cpg' if none are, None if mixed or no synonymous SNV.
    Synonymous variants at mixed codons are ambiguous, because the protein-level
    variant string does not say which base changed.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Boolean masks (all-CpG codons, no-CpG codons), indexed by zero-based codon.
    """
    codons, rates = get_snv_mutation_rates(cds)['syn']
    lo = np.full(len(cds) // 3, np.inf)
    hi = np.zeros(len(cds) // 3)
    np.minimum.at(lo, codons, rates)
    np.maximum.at(hi, codons, rates)
    return (lo > CPG_RATE) & (hi > 0), (hi > 0) & (hi <= CPG_RATE)


def syn_maf_by_cpg_class(variants, cds):
    """
    Split a transcript's observed synonymous MAFs (in percent) into those at
    no-CpG codons and those at all-CpG codons (see codon_cpg_class).

    Parameters
    ----------
    variants : husc.data.Variants
        The transcript's gnomAD variants.
    cds : str
        Coding sequence without the stop codon.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        (non-CpG MAFs, CpG MAFs), in percent like the AF lists of the transcript table.
    """
    is_cpg, is_non_cpg = codon_cpg_class(cds)
    keep = (variants.ref == variants.alt) & (variants.an > 0) & (variants.pos <= len(is_cpg))
    codon = variants.pos[keep] - 1
    af = variants.ac[keep] / variants.an[keep]
    maf = np.minimum(af, 1 - af) * 100
    return maf[is_non_cpg[codon]], maf[is_cpg[codon]]


def _percent_to_maf(dist):
    """Convert allele frequencies in percent to fractions, folded to the minor allele."""
    d = np.asarray(dist, dtype=float) / 100
    return np.where(d > 0.5, 1 - d, d)


def permute_variants(snv_codons, snv_rates, length, expected_total, maf_dist, maf_dist_cpg, cpg_mask,
                     rng, n=N_PERMUTATIONS, chunk=1000):
    """
    Draw the permutation null. In each permutation every possible SNV is
    observed independently with probability saturating_snv_probs(snv_rates,
    expected_total); each observed SNV gets a MAF drawn from maf_dist_cpg if
    cpg_mask marks it as CpG, otherwise from maf_dist (both in percent).

    Parameters
    ----------
    snv_codons, snv_rates : numpy.ndarray
        Zero-based codon index and (CpG-scaled) mutation rate of each possible SNV.
    length : int
        Number of codons.
    expected_total : int
        Expected number of variants of this class in the transcript.
    maf_dist, maf_dist_cpg : array-like
        MAFs in percent for non-CpG and CpG SNVs.
    cpg_mask : numpy.ndarray of bool
        Which SNVs are CpG.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        (n, length) matrix of per-codon MAF sums.
    """
    if len(maf_dist) == 0:
        raise ValueError('Empty allele frequency distribution')
    if len(snv_codons) and snv_codons.max() >= length:
        raise ValueError('SNV codon index exceeds peptide length {}'.format(length))

    maf, maf_cpg = _percent_to_maf(maf_dist), _percent_to_maf(maf_dist_cpg)
    cpg_mask = np.asarray(cpg_mask, bool)
    p = saturating_snv_probs(snv_rates, expected_total)
    out = np.empty((n, length))
    for start in range(0, n, chunk):
        c = min(chunk, n - start)
        hits = rng.random((c, len(p))) < p
        rows, cols = np.nonzero(hits)
        mafs = maf[rng.integers(len(maf), size=len(rows))]
        cpg_hits = cpg_mask[cols]
        mafs[cpg_hits] = maf_cpg[rng.integers(len(maf_cpg), size=cpg_hits.sum())]
        flat_index = rows * length + snv_codons[cols]
        out[start:start + c] = np.bincount(
            flat_index, weights=mafs, minlength=c * length
        ).reshape(c, length)
    return out


def permutation_test(pmt_matrix, sites, observed):
    """
    Compare an observed contact-set MAF sum with its permutation null.

    Parameters
    ----------
    pmt_matrix : numpy.ndarray
        (n, length) matrix from permute_variants.
    sites : list of int
        One-based codon positions of the contact set.
    observed : float
        Observed MAF sum over the contact set.

    Returns
    -------
    tuple of (float, float)
        Null mean of the MAF sum, and the mid-p value
        p = (#{null < obs} + 0.5 * #{null == obs} + 0.5) / (n + 1),
        the probability of observing this sum or less.
    """
    null = pmt_matrix[:, [pos - 1 for pos in sites]].sum(axis=1)
    n_below = np.sum(null < observed) + 0.5 * np.sum(null == observed)
    return np.mean(null), (n_below + 0.5) / (len(null) + 1)
