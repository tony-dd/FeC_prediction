import re
import os
import random
import math
import itertools
import numpy as np
from ase.io import read
import pickle

def get_atoms():
    p1 = re.compile(r"cluste1")
    p2 = re.compile(r"csp-CALYPSO")
    path = '/data2/deh/calypso-FexCy-clusters'

    filename = os.listdir(path)
    folder1_names_list = []

    for folder1_name in filename:
        if p1.search(folder1_name) is not None:
            folder1_names_list.append(folder1_name)

    folder2_names_dict = {}

    for folder1_name in folder1_names_list:
        filename = os.listdir(path + '/' + folder1_name)
        folder2_names = []
        for folder2_name in filename:
            if p2.search(folder2_name) is not None:
                folder2_names.append(folder2_name)
        folder2_names_dict[folder1_name] = folder2_names

    atoms_list = []

    for i, folder1_name in enumerate(folder1_names_list):
        print(i, len(folder1_names_list))
        for j, folder2_name in enumerate(folder2_names_dict[folder1_name]):
            print('--', j, len(folder2_names_dict[folder1_name]))
            try:
                xml_path = '{0}/{1}/{2}/vasprun.xml'.format(path, folder1_name, folder2_name)
                ret_atoms = read(xml_path, index=':')
                atoms_list.extend(ret_atoms)
            except (FileNotFoundError):
                continue
    return atoms_list

def get_energy(atoms_list):
    energy_list = []
    for i, atoms in enumerate(atoms_list):
        e = atoms.get_potential_energy()
        energy_list.append(e)

    return energy_list

if __name__ == '__main__':
    dataset = []
    atoms_list = get_atoms()
    energy_list = get_energy(atoms_list)
    print(len(atoms_list), len(energy_list))
    for i in range(len(atoms_list)):
        dataset.append([atoms_list[i], energy_list[i]])
    print(len(dataset), len(dataset[0]))

    with open('/data2/deh/data/KF5/test_data4.pkl', 'wb') as f:
        pickle.dump(dataset, f)



