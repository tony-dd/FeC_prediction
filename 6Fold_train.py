import tensorflow as tf
import numpy as np
import itertools
from numpy import mean
import pickle
import random
from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler
from read import get_atoms
from scipy import linalg
from pymatgen.io.ase import AseAtomsAdaptor
aaa = AseAtomsAdaptor()

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

f1 = open('/data2/deh/data/KF6/test_data1.pkl', 'rb')
data1 = pickle.load(f1)
f2 = open('/data2/deh/data/KF6/test_data2.pkl', 'rb')
data2 = pickle.load(f2)
f3 = open('/data2/deh/data/KF6/test_data3.pkl', 'rb')
data3 = pickle.load(f3)
f4 = open('/data2/deh/data/KF6/test_data4.pkl', 'rb')
data4 = pickle.load(f4)
f5 = open('/data2/deh/data/KF6/test_data5.pkl', 'rb')
data5 = pickle.load(f5)
f6 = open('/data2/deh/data/KF6/test_data6.pkl', 'rb')
data6 = pickle.load(f6)

Loss = []

dataset = [data1, data2, data3, data4, data5, data6]

gc = CrystalGraph(bond_converter=GaussianDistance(
    np.linspace(0, 5, 100), 0.5), cutoff=10)
model = MEGNetModel(100, 2, graph_converter=gc)

for i in range(6):
    test_data = dataset[i]
    train_vali_data = []
    for j in range(len(dataset)):
        if i == j:
            continue
        train_vali_data += dataset[j]

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

    with tf.device('/gpu:1'):

        scaler = StandardScaler.from_training_data(train_structures, train_labels, is_intensive=False)
        model.target_scaler = scaler
        model.train(train_structures, train_labels, vali_structures, vali_labels,
                    epochs=1000,
                    batch_size=256,
                    verbose=2,
                    patience=10)

        prediction = model.predict_structures(test_structures)

    loss = loss_mae(prediction, test_labels)
    Loss.append(loss)

print('Loss:', Loss)
print('MAE:', np.mean(Loss))
