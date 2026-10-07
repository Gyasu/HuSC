#!/usr/bin/env python3


import os
import numpy as np
from Bio.PDB import NeighborSearch

from husc.utils import Contact


def search_for_all_contacts(residues, radius=8.0):
    """
    Search for all contacts in the given set of residues based on
    distances between CB atoms.

    Parameters
    ----------
    residues : list
        A list of Biopython Residue objects.
    radius : float
        The radius within which two residues are considered in contact.

    Returns
    -------
    list
        A list of Contact objects.

    """
    atom_list = []
    for r in residues:
        if r.get_resname() == 'GLY':
            try:
                atom_list.append(r['CA'])
            except KeyError:
                print('No CA atom found for GLY:', r, 'skipped ...')
                continue
        else:
            try:
                atom_list.append(r['CB'])
            except KeyError:
                print('No CB atom found for:', r.get_resname(), 'skipped ...')
                continue
            # atom_list += [a for a in r.get_atoms() if a.get_name()
            #               not in BACKBONE_ATOMS]
        # atom_list += [a for a in r.get_atoms()]
    ns = NeighborSearch(atom_list)
    all_contacts = [
        Contact(res_a=c[0], res_b=c[1])
        for c in ns.search_all(radius, level='R')
    ]
    return all_contacts


def load_structure(path, structure_id='complex'):
    """
    Load a PDB or mmCIF structure (optionally gzipped) with Biopython.

    The format is chosen from the file name: ``.cif``/``.mmcif`` (optionally
    followed by ``.gz``) are parsed as mmCIF; everything else as PDB.

    Parameters
    ----------
    path : str
        Path to the structure file.
    structure_id : str
        Identifier given to the Bio.PDB.Structure object.

    Returns
    -------
    Bio.PDB.Structure.Structure or None
        The parsed structure, or None if the file cannot be read.
    """
    import gzip
    from Bio.PDB import PDBParser, MMCIFParser

    name = path[:-3] if path.endswith('.gz') else path
    is_cif = name.lower().endswith(('.cif', '.mmcif'))
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(PERMISSIVE=1, QUIET=True)
    try:
        if path.endswith('.gz'):
            with gzip.open(path, 'rt') as handle:
                return parser.get_structure(structure_id, handle)
        return parser.get_structure(structure_id, path)
    except (FileNotFoundError, ValueError, IOError, KeyError) as e:
        print(f'Structure file cannot be read: {path} ({e})')
        return None
