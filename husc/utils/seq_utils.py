from husc.utils.genetic_code import GENETIC_CODE
from husc.utils.trinucleotide_context_rates import MUTATION_RATES_UNIQUE
import numpy as np

# SNVs above this trinucleotide-context rate are CpG transitions (>=1.7e-7; all others <=3.2e-8)
CPG_RATE = 1e-7

def count_codon_ns(codon):
    """
    Count the respective number of possible missense and synonymous SNPs of the
    given codon.

    Parameters
    ----------
    codon : str
        A three-letter genetic codon.

    Returns
    -------
    tuple
        A pair of # missense variants vs # synonymous variants.

    """
    nucleotides = {'A', 'T', 'C', 'G'}
    aa = GENETIC_CODE[codon]
    missense = 0
    synonymous = 0
    for i, x in enumerate(codon):
        for n in nucleotides - {x}:
            new_codon = codon[:i] + n + codon[i+1:]
            new_aa = GENETIC_CODE[new_codon]
            if new_aa == aa:
                synonymous += 1
            elif new_aa != aa and new_aa != '*':
                missense += 1
    return missense, synonymous


def gc_content(seq):
    """
    Compute the GC content of the given sequence.

    Parameters
    ----------
    seq : str
        A valid nucleotide (DNA) sequence.

    Returns
    -------
    float
        The fraction of nucleotides that are G and C.

    """
    # make sure that the input is valid
    if set(seq).difference(set('ATCG')):
        raise ValueError('Invalid input sequence: %s.' % seq)

    gc_count = 0
    for x in seq:
        if x in set('GC'):
            gc_count += 1
    return gc_count / len(seq)


def get_permutation_stats(pmt_matrix, cs_sites, n_obs, midp=False):
    """

    Parameters
    ----------
    pmt_matrix
    cs_sites : list
    n_obs : int
    midp : bool
        If True, permutations tied with n_obs count as half (mid-p value).

    Returns
    -------

    """
    if not isinstance(pmt_matrix, dict):
        contact_res_indices = [pos - 1 for pos in cs_sites]
        pmt = pmt_matrix[:, contact_res_indices].sum(axis=1)
    else:
        pmt = 0
        for res in cs_sites:
            chain_id = res.get_full_id()[2]
            res_index = res.get_full_id()[3][1] - 1
            pmt += pmt_matrix[chain_id][:, res_index]
    pmt_mean = np.mean(pmt)
    pmt_sd = np.std(pmt)
    if midp:
        n = np.sum(pmt < n_obs) + 0.5 * np.sum(pmt == n_obs)
        p_value = (n + 0.5) / (len(pmt) + 1)
    else:
        n = np.sum(pmt <= n_obs)
        p_value = (n + 1) / 10001
    return pmt_mean, pmt_sd, p_value


def get_permutation_stats_log(pmt_matrix, cs_sites, n_obs):
    """

    Parameters
    ----------
    pmt_matrix
    cs_sites : list
    n_obs : int

    Returns
    -------

    """
    if not isinstance(pmt_matrix, dict):
        contact_res_indices = [pos - 1 for pos in cs_sites]
        pmt = pmt_matrix[:, contact_res_indices].sum(axis=1)
    else:
        pmt = 0
        for res in cs_sites:
            chain_id = res.get_full_id()[2]
            res_index = res.get_full_id()[3][1] - 1
            pmt += pmt_matrix[chain_id][:, res_index]
    
    pmt_log = np.log10(pmt + 1)
    pmt_mean = np.mean(pmt_log)
    pmt_sd = np.std(pmt_log)
    n = np.sum(pmt_log <= n_obs)
    p_value = (n + 1) / 10001
    return pmt_mean, pmt_sd, p_value


def get_codon_seq_context(codon_nums, cds):
    """

    Parameters
    ----------
    codon_nums : list/int
        A list of ints, each represents a codon number.
    cds : str
        Coding sequence

    Returns
    -------
    str
        The sequence context of codons concatenated into a string.

    """
    # make sure the codon_nums are valid
    total_num_codons = len(cds) // 3
    if isinstance(codon_nums, int):
        if codon_nums > total_num_codons:
            print('The given coding sequence has %s codons:' % total_num_codons)
            print(cds)
            raise IndexError('Codon number %s out of range.' % codon_nums)
    else:
        for i in codon_nums:
            if i > total_num_codons:
                print('The given coding sequence has %s codons:' % total_num_codons)
                print(cds)
                raise IndexError('Codon number %s out of range.' % i)

    # if only given a single codon number
    if isinstance(codon_nums, int):
        if codon_nums == 1:
            # wrap to the last nucleotide
            return cds[-1:] + cds[:4]
        elif codon_nums == total_num_codons:
            # wrap to the first nucleotide
            return cds[-4:] + cds[:1]
        else:
            return cds[((codon_nums - 1) * 3 - 1):(codon_nums * 3) + 1]
    else:
        codon_seq_context = ''
        for i in codon_nums:
            codon_seq_context += cds[((i - 1) * 3 - 1):(i * 3) + 1]
        return codon_seq_context

def permute_variants_dist(m, length, p=None, dist=None, n=10000):
    """
    To be added ...

    Parameters
    ----------
    m
    length
    p
    n

    Returns
    -------

    """
    if not (p is None) and length != len(p):

        raise ValueError(
            'Peptide length did not match probability length {} != {}'.format(length, len(p))
        )
    
    if dist is None or len(dist) == 0:
        raise ValueError('Empty allele frequency distribution')

    # normalize p
    if p is not None:
        p = np.asarray(p, dtype=float)
        p = p / np.sum(p)

    # Convert MAF from percent to fraction and fold to the minor allele
    def to_maf(d):
        d = np.asarray(d, dtype=float) / 100
        return np.where(d > 0.5, 1 - d, d)
    maf = to_maf(dist)
    is_cpg = None
    if dist_cpg is not None and len(dist_cpg):
        is_cpg = np.asarray(cpg_mask, bool) if cpg_mask is not None else np.asarray(snv_rates) > CPG_RATE
    if is_cpg is not None:
        maf_cpg = to_maf(dist_cpg)

    # Each permutation places m variants on sites drawn by p, and each variant
    # gets an independent MAF drawn from dist
    m = int(m)
    m_sites = np.random.choice(length, size=(n, m), replace=True, p=p)
    m_mafs = maf[np.random.randint(len(maf), size=(n, m))]
    flat_index = (np.arange(n)[:, None] * length + m_sites).ravel()
    missense_matrix = np.bincount(flat_index, weights=m_mafs.ravel(), minlength=n * length)
    return missense_matrix.reshape(n, length)

def count_poss_ns_variants(cds):
    """

    Parameters
    ----------
    cds : str
        Coding sequence.

    Returns
    -------
    list
        A pair of # missense variants vs # synonymous variants for each
        codon in the coding sequence.

    """
    cds_ns = []
    for i in range(0, len(cds), 3):
        cds_ns.append(count_codon_ns(cds[i:i+3]))
    return cds_ns

def count_ns_sites(cds):
    """
    Count the number of nonsynonymous and synonymous sites in the given cds.

    Parameters
    ----------
    cds : str
        Coding DNA sequence.

    Returns
    -------
    tuple
        Number of nonsynonymous and synonymous sites as a tuple.

    """
    # make sure that the given CDS is valid
    if set(cds).difference(set('ATCG')):
        raise ValueError(
            'Invalid DNA sequence: containing letters other than ATCG.'
        )
    if len(cds) % 3 != 0:
        raise ValueError('Invalid CDS: CDS length is not a multiple of 3.')

    # now compute dN and dS
    cds_ns = []
    for i in range(0, len(cds), 3):
        codon = cds[i:i+3]
        wt_aa = GENETIC_CODE[codon]
        n = 0
        s = 0
        for k, x in enumerate(codon):
            for y in {'A', 'T', 'C', 'G'} - {x}:
                var_codon = codon[:k] + y + codon[k+1:]
                var_aa = GENETIC_CODE[var_codon]
                if wt_aa != var_aa:
                    n += 1.0 / 3
                else:
                    s += 1.0 / 3
        cds_ns.append((n, s))

    return cds_ns

def get_codon_mutation_rates(cds):
    """

    Parameters
    ----------
    cds : str
        Coding sequence.

    Returns
    -------
    mutation probabilities : list
        A list of tuples consisting of synonymous and nonsynonymous mutation
        probabilities.
    """
    if len(cds) % 3 != 0:
        raise ValueError('Given CDS length is not a multiple of 3.')

    num_codons = len(cds) // 3

    mutation_rates = []
    # one nucleotide before and one nucleotide after the codon
    for codon_number in range(1, num_codons + 1):
        # determine the codon sequence
        codon_sequence = cds[(codon_number - 1) * 3:codon_number * 3]

        synonymous_rate = 0
        nonsynonymous_rate = 0

        # determine the mutation rate of the first and the last codons
        # consider only two mutatable nucleotides
        if codon_number == 1 or codon_number == num_codons:
            # first codon
            if codon_number == 1:
                sequence_context = cds[:4]
                # i is the zero-indexed position of the mutated nucleotide
                for i in range(1, 3):
                    trinucleotide = sequence_context[i - 1:i + 2]
                    rates = MUTATION_RATES_UNIQUE[trinucleotide]
                    for k, v in rates.items():
                        mutant_sequence = codon_sequence[:i] + k[1] + codon_sequence[i + 1:]
                        if GENETIC_CODE[codon_sequence] == GENETIC_CODE[mutant_sequence]:
                            synonymous_rate += v
                        elif GENETIC_CODE[codon_sequence] != GENETIC_CODE[mutant_sequence] \
                            and GENETIC_CODE[mutant_sequence] != 'STOP':
                            nonsynonymous_rate += v
            # last codon
            else:
                sequence_context = cds[-4:]
                # i is the zero-indexed position of the mutated nucleotide
                for i in range(0, 2):
                    trinucleotide = sequence_context[i:i + 3]
                    rates = MUTATION_RATES_UNIQUE[trinucleotide]
                    for k, v in rates.items():
                        mutant_sequence = codon_sequence[:i] + k[1] + codon_sequence[i + 1:]
                        if GENETIC_CODE[codon_sequence] == GENETIC_CODE[mutant_sequence]:
                            synonymous_rate += v
                        elif GENETIC_CODE[codon_sequence] != GENETIC_CODE[mutant_sequence] \
                            and GENETIC_CODE[mutant_sequence] != 'STOP':
                            nonsynonymous_rate += v

        # codons other than the first and the last
        # consider all three mutatable nucleotides
        else:
            # one nucleotide before and one nucleotide after the codon
            sequence_context = cds[(codon_number - 1) * 3 - 1:(codon_number - 1) * 3 + 4]
            # mutate nucleotide in the codon iteratively
            for i in range(3):
                trinucleotide = sequence_context[i:i + 3]
                rates = MUTATION_RATES_UNIQUE[trinucleotide]
                for k, v in rates.items():
                    codon_seq_list = list(codon_sequence)
                    codon_seq_list[i] = k[1]
                    mutant_sequence = ''.join(codon_seq_list)
                    if GENETIC_CODE[codon_sequence] == GENETIC_CODE[mutant_sequence]:
                        synonymous_rate += v
                    elif GENETIC_CODE[codon_sequence] != GENETIC_CODE[mutant_sequence] \
                        and GENETIC_CODE[mutant_sequence] != 'STOP':
                        nonsynonymous_rate += v

        mutation_rates.append((synonymous_rate, nonsynonymous_rate))
    return mutation_rates


def get_snv_mutation_rates(cds):
    """
    List every possible single-nucleotide variant of the CDS with its codon and
    trinucleotide-context mutation rate, split into synonymous and missense.
    Mirrors get_codon_mutation_rates (same nucleotides considered for the first
    and last codons), so per-codon sums of these rates equal its output.

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
    with beta chosen so that the probabilities sum to expected_total. Unlike a
    linear model, highly mutable SNVs saturate at 1 and each SNV is observed at
    most once.
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


def permute_variants_saturating(snv_codons, snv_rates, length, expected_total, dist, n=10000, chunk=1000,
                                 dist_cpg=None, cpg_mask=None):
    """
    Saturating permutation null: in each permutation every possible SNV is
    observed independently with probability saturating_snv_probs(...), and each
    observed SNV gets a MAF drawn from dist (in percent). If dist_cpg is given,
    CpG SNVs (rate > CPG_RATE) draw their MAF from dist_cpg instead, since
    recurrent mutation gives them higher allele frequencies; cpg_mask overrides
    which SNVs count as CpG (needed when rates were rescaled). Returns an
    (n, length) matrix of per-codon MAF sums, like permute_variants_dist.
    """
    if dist is None or len(dist) == 0:
        raise ValueError('Empty allele frequency distribution')
    if len(snv_codons) and snv_codons.max() >= length:
        raise ValueError('SNV codon index exceeds peptide length {}'.format(length))

    # Convert MAF from percent to fraction and fold to the minor allele
    def to_maf(d):
        d = np.asarray(d, dtype=float) / 100
        return np.where(d > 0.5, 1 - d, d)
    maf = to_maf(dist)
    is_cpg = None
    if dist_cpg is not None and len(dist_cpg):
        is_cpg = np.asarray(cpg_mask, bool) if cpg_mask is not None else np.asarray(snv_rates) > CPG_RATE
    if is_cpg is not None:
        maf_cpg = to_maf(dist_cpg)

    p = saturating_snv_probs(snv_rates, expected_total)
    out = np.empty((n, length))
    for start in range(0, n, chunk):
        c = min(chunk, n - start)
        hits = np.random.random_sample((c, len(p))) < p
        rows, cols = np.nonzero(hits)
        mafs = maf[np.random.randint(len(maf), size=len(rows))]
        if is_cpg is not None:
            cpg_hits = is_cpg[cols]
            mafs[cpg_hits] = maf_cpg[np.random.randint(len(maf_cpg), size=cpg_hits.sum())]
        flat_index = rows * length + snv_codons[cols]
        out[start:start + c] = np.bincount(
            flat_index, weights=mafs, minlength=c * length
        ).reshape(c, length)
    return out


def local_cpg_oe(cds, half_window=100):
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


def scale_cpg_rates(snv_codons, snv_rates, cds, cpg_scale):
    """
    Multiply CpG SNV rates (rate > CPG_RATE) by a factor that depends on the
    local CpG o/e of their codon (methylation proxy). cpg_scale is a dict with
    bin 'edges' (len B+1) and 'factors' (len B). Returns (scaled rates, CpG mask).
    """
    rates = np.asarray(snv_rates, dtype=float)
    is_cpg = rates > CPG_RATE
    oe = local_cpg_oe(cds)[snv_codons]
    edges, factors = np.asarray(cpg_scale['edges']), np.asarray(cpg_scale['factors'])
    b = np.clip(np.searchsorted(edges, oe, side='right') - 1, 0, len(factors) - 1)
    return np.where(is_cpg, rates * factors[b], rates), is_cpg


def syn_maf_by_cpg_class(variants, cds):
    """
    Split a transcript's observed synonymous MAFs (in percent) by codon class.
    A codon is CpG if every possible synonymous SNV in it has rate > CPG_RATE,
    non-CpG if none has; synonymous variants at mixed codons are ambiguous
    (the protein-level variant string does not say which base changed) and dropped.

    Parameters
    ----------
    variants : list
        [(variant_string, allele_count, allele_number), ...] as in the gnomAD JSON.
    cds : str
        Coding sequence without the stop codon.

    Returns
    -------
    tuple of (list, list)
        (non-CpG MAFs, CpG MAFs), in percent like the AF lists of the expected-count file.
    """
    codons, rates = get_snv_mutation_rates(cds)['syn']
    lo = np.full(len(cds) // 3, np.inf)
    hi = np.zeros(len(cds) // 3)
    np.minimum.at(lo, codons, rates)
    np.maximum.at(hi, codons, rates)
    non_cpg, cpg = [], []
    for variant_str, allele_count, allele_number in variants:
        if variant_str[0] != variant_str[-1] or int(allele_number) == 0:
            continue
        c = int(variant_str[1:-1]) - 1
        if c >= len(hi) or hi[c] == 0:
            continue
        maf = int(allele_count) / int(allele_number)
        maf = min(maf, 1 - maf) * 100
        if lo[c] > CPG_RATE:
            cpg.append(maf)
        elif hi[c] <= CPG_RATE:
            non_cpg.append(maf)
    return non_cpg, cpg


def get_transcript_mutation_prob(transcript_cds):
    """

    Parameters
    ----------
    transcript_cds

    Returns
    -------

    """
    codon_mutation_prob = get_codon_mutation_rates(transcript_cds)
    syn_prob = 0
    mis_prob = 0
    for probs in codon_mutation_prob:
        syn_prob += probs[0]
        mis_prob += probs[1]
    return syn_prob, mis_prob

def is_valid_cds(cds):
    """
    Determines whether a given sequence is a valid transcript sequence or not.
    A valid transcript sequence is defined as one that starts with ATG, ends
    with a stop codon, that whose length is a multiple of three.

    Parameters
    ----------
    seq_record : SeqRecord
        An object of type SeqRecord.

    Returns
    -------
    bool
        True if the given sequence is a valid transcript sequence else False.

    """
    # stop codons
    stop_codons = {'TAG', 'TAA', 'TGA'}

    # the following conditionals skip noncanonical transcripts
    if not cds[:3] == 'ATG':
        print('The given CDS does not start with ATG.')
        return False
    if cds[-3:] not in stop_codons:
        print('The given CDS does not have a stop codon.')
        return False
    if len(cds) % 3 != 0:
        print('The length of the given CDS is not a multiple of 3.')
        return False

    return True

def read_enst_mp_count_dist(input_file):
    """
    Reads transcript-level mutation probabilities and variant counts from a disk
    file in which records are delimited by tabs.

    Parameters
    ----------
    input_file : str
        File containing transcript-level mutation probabilities and variant counts.

    Returns
    -------
    dict
        A dictionary indexed by transcript IDs. Each indexed value is a tuple
        of length, syn_prob, syn_count, mis_prob, mis_count,
        syn_count_exp, mis_count_exp, syn_dist, mis_dist.

    """
    enst_mp_counts = {}
    with open(input_file, 'rt') as ipf:
        for line in ipf:
            if not line.startswith('ENST'):
                continue
            fields = line.strip().split('\t')
            enst_id = fields[0]
            length = int(fields[1])
            syn_prob = float(fields[2])
            syn_count = int(fields[3])
            mis_prob = float(fields[4])
            mis_count = int(fields[5])

            # Read syn_dist and mis_dist fields as lists of allele frequencies (in percent).
            # Empty lists are kept empty; transcripts without a distribution are skipped downstream.
            try:
                # Process syn_dist and mis_dist by stripping quotes and skipping empty strings
                syn_dist = [float(x.strip().strip('"')) for x in fields[6].split(',') if x.strip().strip('"')]
                mis_dist = [float(x.strip().strip('"')) for x in fields[7].split(',') if x.strip().strip('"')]
            except ValueError as e:
                print(f'ValueError for {enst_id} in fields[6] or fields[7]: {fields[6]}, {fields[7]}')
                print(f'Error message: {e}')
                raise  # Re-raise the exception to stop the program

            # mis_dist: Assign [1] if x is empty, otherwise convert to a list of integers
            # mis_dist = [int(x) for x in fields[7].split(',') if x] or [1]
            # mis_dist = [int(x.strip('"')) for x in fields[9].split(',') if x.strip('"').isdigit()]

            # expected synonymous count predicted using a model fitted by
            # by regressing synonymous count to synonymous probability
            syn_count_exp = round(float(fields[8]))

            # expected missense count predicted using a model fitted by
            # by regressing missense count to missense probability
            mis_count_exp = round(float(fields[9]))

            enst_mp_counts[enst_id] = (
                length, syn_prob, syn_count,
                mis_prob, mis_count, syn_count_exp, mis_count_exp,
                syn_dist, mis_dist
            )
    return enst_mp_counts