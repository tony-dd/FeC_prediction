from schnetpack import AtomsData
import torch
import pickle
import numpy as np

for k in range(9):

    f = open('/data2/deh/data/energy_force/Fe' + str(k+1) + 'Cx.pkl', 'rb')
    dataset = pickle.load(f)

    atoms, property_list = [], []
    for i in range(len(dataset)):
        atoms.append(dataset[i][0])
        energy = np.array([dataset[i][1]], dtype='float32')
        forces = np.array(dataset[i][2], dtype='float32')
        property_list.append({'energy': energy, 'forces': forces})

    print(len(property_list))

    new_dataset = AtomsData('./Fe' + str(k+1) + 'Cx_prop.db', available_properties=['energy', 'forces'])
    new_dataset.add_systems(atoms, property_list)

    print(len(new_dataset))

