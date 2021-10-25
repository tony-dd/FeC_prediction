import tensorflow as tf
import numpy as np
import pickle
import random
from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler
from read import get_atoms
from pymatgen.io.ase import AseAtomsAdaptor
aaa = AseAtomsAdaptor()

f = open('/data2/deh/data/data_train.pkl', 'rb')
dataset = pickle.load(f)

random.shuffle(dataset)

structures, labels = [], []
for i in range(len(dataset)):
    atoms = dataset[i][0]
    struct = aaa.get_molecule(atoms, cls=None)
    structures.append(struct)
    labels.append(dataset[i][1])

with tf.device('/gpu:0'):

    train_structures = structures[:int(len(structures) * 0.9)]
    vali_structures = structures[int(len(structures) * 0.9):]
    train_targets = labels[:int(len(structures) * 0.9)]
    vali_targets = labels[int(len(structures) * 0.9):]

    gc = CrystalGraph(bond_converter=GaussianDistance(
        np.linspace(0, 5, 100), 0.5), cutoff=10)
    model = MEGNetModel(100, 2, graph_converter=gc)

    scaler = StandardScaler.from_training_data(train_structures, train_targets, is_intensive=False)
    model.target_scaler = scaler
    model.train(train_structures, train_targets, vali_structures, vali_targets, epochs=1000, batch_size=256, verbose=2, patience=10)

model.save_model('/data2/deh/model/FeC_megnet_model_diff_epoch1000_batch256_patience_10.hdf5')
