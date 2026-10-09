from husc.utils.genetic_code import GENETIC_CODE
from husc.utils.trinucleotide_context_rates import MUTATION_RATES_UNIQUE
import numpy as np

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


def get_permutation_stats(pmt_matrix, cs_sites, n_obs):
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
    pmt_mean = np.mean(pmt)
    pmt_sd = np.std(pmt)
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
    
    # normalize p
    if p is not None:
        p = p / np.sum(p)
    missense_matrix = []
    for _ in range(n):
        v = np.zeros(length)
        m_sites = np.random.choice(range(length), m, replace=True, p=p)
        for s in m_sites:
            MAF = np.random.choice(dist)
            #Convert MAF to fraction
            MAF = MAF / 100

            if MAF > 0.5:
                result = 1 - MAF
            else:
                result = MAF            

            v[s] += result

        missense_matrix.append(v)
    return np.stack(missense_matrix)

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
                            and GENETIC_CODE[mutant_sequence] != '*':
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
                            and GENETIC_CODE[mutant_sequence] != '*':
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
                        and GENETIC_CODE[mutant_sequence] != '*':
                        nonsynonymous_rate += v

        mutation_rates.append((synonymous_rate, nonsynonymous_rate))
    return mutation_rates


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

            # Read syn_dist and mis_dist fields as lists of numbers
            try:
                # Process syn_dist and mis_dist by stripping quotes and skipping empty strings
                syn_dist = [float(x.strip().strip('"')) for x in fields[6].split(',') if x.strip() and x.strip().strip('"').isdigit()] or [0.0007]
                mis_dist = [float(x.strip().strip('"')) for x in fields[7].split(',') if x.strip() and x.strip().strip('"').isdigit()] or [0.0007]
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