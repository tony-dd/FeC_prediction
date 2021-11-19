import tensorflow as tf
import numpy as np
import itertools
from numpy import mean
import pickle
import random
import logging
from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler
from read import get_atoms
from scipy import linalg
from pymatgen.io.ase import AseAtomsAdaptor
aaa = AseAtomsAdaptor()

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def loss_mae(l1, l2):
    err = 0
    for i in range(len(l1)):
        err += abs(l1[i]-l2[i])
    mae = err/len(l1)
    return mae

def process(test_data):
    structures, labels = [], []
    for i in range(len(test_data)):
        atoms = test_data[i][0]
        struct = aaa.get_molecule(atoms, cls=None)
        structures.append(struct)
        labels.append(test_data[i][1])
    return structures, labels

dataset = []

for i in range(9):
    f = open('/data2/deh/data/KF9/test_data' + str(i+1) + '.pkl', 'rb')
    data = pickle.load(f)
    print(len(data))
    dataset.append(data)

Loss = []

print(len(dataset))

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

for i in range(len(dataset)):
    test_data = dataset[i]
    train_vali_data = []
    for j in range(len(dataset)):
        if i == j:
            continue
        train_vali_data += dataset[j]

    random.shuffle(train_vali_data)

    print('test_data:', len(test_data))

    train_vali_structures = []
    train_vali_labels = []

    train_structures = []
    train_labels = []
    vali_structures = []
    vali_labels = []
    test_structures = []
    test_labels = []

    train_vali_structures, train_vali_labels = process(train_vali_data)
    test_structures, test_labels = process(test_data)

    train_structures = train_vali_structures[:int(len(train_vali_structures) * 0.9)]
    vali_structures = train_vali_structures[int(len(train_vali_structures) * 0.9):]
    train_labels = train_vali_labels[:int(len(train_vali_labels) * 0.9)]
    vali_labels = train_vali_labels[int(len(train_vali_labels) * 0.9):]

    with tf.device('/gpu:0'):

        gc = CrystalGraph(bond_converter=GaussianDistance(
                np.linspace(0, 5, 100), 0.5), cutoff=10)
        model = MEGNetModel(100, 2, graph_converter=gc)

        scaler = StandardScaler.from_training_data(train_structures, train_labels, is_intensive=False)
        model.target_scaler = scaler
        model.train(train_structures, train_labels, vali_structures, vali_labels,
                    epochs=1000,
                    batch_size=256,
                    save_checkpoint=False,
                    callbacks=[callback])

        prediction = model.predict_structures(test_structures)

        name = 'cuda0_9Fold_megnet_patience10_' + str(i+1)
        model.save_model(name + '.hdf5')

    loss = loss_mae(prediction, test_labels)
    print(loss)
    Loss.append(loss)

print('Loss:', Loss)
print('MAE:', np.mean(Loss))
