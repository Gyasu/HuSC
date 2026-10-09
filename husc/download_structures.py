#!/usr/bin/env python3
"""
Download the AlphaFold DB models of the proteins in an input file into
config['pdb_dir'], gzipped as <model name>.gz. Models that are already present
are skipped. For the whole human proteome, the bulk archive is faster:
https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar

Usage:
    python -m husc.download_structures -c config.json -i inputs/example.txt
"""

import gzip
import os
import time
import urllib.error
import urllib.request
from argparse import ArgumentParser

from tqdm import tqdm

from husc import data

AFDB_URL = 'https://alphafold.ebi.ac.uk/files/{}'


def main():
    parser = ArgumentParser(description='Download AlphaFold DB models for HuSC')
    parser.add_argument('-c', '--config', required=True, help='HuSC JSON configuration file (uses pdb_dir)')
    parser.add_argument('-i', '--input', required=True, help='Input file of UniProt IDs (and model names)')
    args = parser.parse_args()

    config = data.load_config(args.config)
    os.makedirs(config['pdb_dir'], exist_ok=True)
    failed = []
    for uniprot_id, model in tqdm(data.read_input_pairs(args.input)):
        path = os.path.join(config['pdb_dir'], model + '.gz')
        if os.path.exists(path):
            continue
        request = urllib.request.Request(AFDB_URL.format(model), headers={'User-Agent': 'Mozilla/5.0 (HuSC)'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
        except urllib.error.URLError as e:
            failed.append((uniprot_id, str(e)))
            continue
        with gzip.open(path + '.tmp', 'wb') as handle:
            handle.write(content)
        os.replace(path + '.tmp', path)
        time.sleep(0.1)  # be gentle with the AlphaFold DB server
    print(f'{len(failed)} downloads failed' + ''.join(f'\n  {u}: {e}' for u, e in failed))


if __name__ == '__main__':
    main()
