"""
Loading of the HuSC input datasets and selection of the Ensembl transcript for
each UniProt protein.
"""

import gzip
import json
import re
from collections import namedtuple

import ijson
import numpy as np
from Bio import SeqIO

from husc.utils import seq_utils

# Configuration keys; values are paths unless noted
REQUIRED_CONFIG_KEYS = [
    'ensembl_cds',       # Ensembl CDS FASTA (gzipped)
    'uniprot_pep',       # UniProt proteome FASTA (gzipped); used by husc.prepare_cpg
    'gnomad_variants',   # gnomAD variants per transcript (JSON)
    'uniprot_to_enst',   # UniProt -> Ensembl transcript IDs (JSON)
    'enst_mp_counts',    # per-transcript mutation probabilities, counts and AF lists (TSV)
    'cpg_maf',           # pooled synonymous MAFs at CpG codons, built by husc.prepare_cpg
    'cpg_scale',         # CpG rate factors by local CpG o/e (JSON), built by husc.prepare_cpg
    'output_dir',        # directory for <uniprot_id>_husc.tsv files
    'pdb_dir',           # directory of AlphaFold models (<model name>.gz)
]
CONFIG_DEFAULTS = {
    'radius': 8.0,       # contact radius in Angstroms (CB atoms; CA for glycine)
    'seed': 0,           # base random seed; each protein's generator is seeded from it and the UniProt ID
}

# Per-transcript row of the enst_mp_counts table. syn_exp and mis_exp are the
# expected counts from the mutability regression; syn_af is the list of
# synonymous allele frequencies (percent) observed in gnomAD.
TranscriptStats = namedtuple(
    'TranscriptStats',
    ['length', 'syn_prob', 'syn_obs', 'mis_prob', 'mis_obs', 'syn_exp', 'mis_exp', 'syn_af']
)

# gnomAD variants of one transcript as parallel arrays: one-based codon
# position, reference and alternate amino acid (ASCII codes), allele count and
# allele number.
Variants = namedtuple('Variants', ['pos', 'ref', 'alt', 'ac', 'an'])

_VARIANT_RE = re.compile(r'^([A-Z*])(\d+)([A-Z*])$')


def open_maybe_gzip(path, mode='rt'):
    """Open a file, decompressing on the fly if its name ends in .gz."""
    return gzip.open(path, mode) if str(path).endswith('.gz') else open(path, mode)


def load_config(path):
    """Read the JSON configuration, check required keys and fill defaults."""
    with open(path, 'rt') as handle:
        config = json.load(handle)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise SystemExit(f'Missing keys in {path}: {", ".join(missing)}')
    return {**CONFIG_DEFAULTS, **config}


def read_input_pairs(path):
    """
    Read one protein per line: a UniProt ID, optionally followed by the AlphaFold
    model name (without .gz). The model name defaults to the AlphaFold DB v6
    model, AF-<UniProt ID>-F1-model_v6.pdb.
    """
    pairs = []
    with open(path, 'rt') as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            model = fields[1] if len(fields) > 1 else f'AF-{fields[0]}-F1-model_v6.pdb'
            pairs.append((fields[0], model))
    return pairs


def load_cds(path):
    """Ensembl CDS sequences as {transcript ID without version: sequence str}."""
    with gzip.open(path, 'rt') as handle:
        return {rec.id.split('.')[0]: str(rec.seq) for rec in SeqIO.parse(handle, 'fasta')}


def load_uniprot_seqs(path):
    """UniProt sequences as {accession: sequence str}."""
    with gzip.open(path, 'rt') as handle:
        return {rec.id.split('|')[1]: str(rec.seq) for rec in SeqIO.parse(handle, 'fasta')}


def load_json(path):
    with open_maybe_gzip(path) as handle:
        return json.load(handle)


def _to_variants(variant_list):
    """Convert [['A7V', ac, an], ...] to a Variants record; malformed strings are dropped."""
    pos, ref, alt, ac, an = [], [], [], [], []
    for variant_str, allele_count, allele_number in variant_list:
        m = _VARIANT_RE.match(variant_str)
        if m is None:
            continue
        ref.append(ord(m.group(1)))
        pos.append(int(m.group(2)))
        alt.append(ord(m.group(3)))
        ac.append(int(allele_count))
        an.append(int(allele_number))
    return Variants(
        np.array(pos, dtype=np.int32), np.array(ref, dtype=np.uint8), np.array(alt, dtype=np.uint8),
        np.array(ac, dtype=np.int64), np.array(an, dtype=np.int64)
    )


def iter_variants(path, enst_ids=None):
    """
    Stream the gnomAD variants JSON, yielding (transcript ID, Variants) for
    transcripts that have a variant list (and are in enst_ids, if given). The
    file is never loaded whole.
    """
    with open_maybe_gzip(path, 'rb') as handle:
        for enst_id, record in ijson.kvitems(handle, ''):
            if 'variants' not in record or (enst_ids is not None and enst_id not in enst_ids):
                continue
            yield enst_id, _to_variants(record['variants'])


def load_variants(path, enst_ids=None):
    """{transcript ID: Variants} for the requested transcripts (all if None)."""
    return dict(iter_variants(path, enst_ids))


def read_transcript_table(path, enst_ids=None):
    """
    Read per-transcript mutation probabilities, observed and expected variant
    counts, and synonymous allele frequency lists (percent).

    Returns
    -------
    dict
        {transcript ID: TranscriptStats}; restricted to enst_ids if given.
        Transcripts with an empty synonymous AF list are kept (with an empty array).
    """
    table = {}
    with open_maybe_gzip(path) as handle:
        for line in handle:
            if not line.startswith('ENST'):
                continue
            fields = line.rstrip('\n').split('\t')
            if enst_ids is not None and fields[0] not in enst_ids:
                continue
            syn_af = [float(x.strip().strip('"')) for x in fields[6].split(',') if x.strip().strip('"')]
            table[fields[0]] = TranscriptStats(
                length=int(fields[1]),
                syn_prob=float(fields[2]),
                syn_obs=int(fields[3]),
                mis_prob=float(fields[4]),
                mis_obs=int(fields[5]),
                syn_exp=round(float(fields[8])),
                mis_exp=round(float(fields[9])),
                syn_af=np.array(syn_af),
            )
    return table


def select_transcript(protein_length, enst_ids, cds_dict, variants):
    """
    Choose the transcript for a protein: among the mapped transcripts with a
    valid CDS whose length matches the protein, the one with the most gnomAD
    variants.

    Returns
    -------
    str
        Selected transcript ID.

    Raises
    ------
    ValueError
        If no mapped transcript has a valid, length-compatible CDS.
    KeyError
        If none of the compatible transcripts has variant data.
    """
    compatible = [
        e for e in enst_ids
        if e in cds_dict and seq_utils.is_valid_cds(cds_dict[e])
        and protein_length == len(cds_dict[e]) // 3 - 1
    ]
    if not compatible:
        raise ValueError(f'No valid CDS of matching length among {enst_ids}')
    with_data = [e for e in compatible if e in variants]
    if not with_data:
        raise KeyError(f'No gnomAD variant data for {compatible}')
    # max() keeps the first of tied transcripts, in mapping order
    return max(with_data, key=lambda e: len(variants[e].pos))


def minor_allele_freqs(variants):
    """MAF of each variant (0 where the allele number is 0)."""
    an = np.maximum(variants.an, 1)
    af = np.where(variants.an > 0, variants.ac / an, 0.0)
    return np.where(af > 0.5, 1 - af, af)


def sum_maf_by_position(variants):
    """
    Sum MAFs of missense and of synonymous variants at each codon position.

    Returns
    -------
    tuple of (dict, dict)
        {one-based position: MAF sum} for missense and synonymous variants;
        a position is present if it has at least one variant of that class.
    """
    maf = minor_allele_freqs(variants)
    out = ({}, {})
    for is_syn, counts in ((False, out[0]), (True, out[1])):
        mask = (variants.ref == variants.alt) == is_syn
        for pos, m in zip(variants.pos[mask].tolist(), maf[mask].tolist()):
            counts[pos] = counts.get(pos, 0) + m
    return out
