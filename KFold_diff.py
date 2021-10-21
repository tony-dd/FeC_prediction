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

def renew(numbers):
    dict = {}
    rel = []

    for i in numbers:
        q = str(i[0]) + '_' + str(i[1])
        if q in dict.keys():
            dict[q].append(i[2])
        else:
            m = []
            dict[q] = m
            dict[q].append(i[2])

    for i in dict.keys():
        mean_c = mean(dict[i])
        k = i.split('_')
        k[0] = int(k[0])
        k[1] = int(k[1])
        k.append(mean_c)
        rel.append(k)
    return rel

def loss_mae(l1, l2):
    err = 0
    for i in range(len(l1)):
        err += abs(l1[i]-l2[i])
    mae = err/len(l1)
    return mae

def process1(train_data, Fe, C):
    structures, labels = [], []
    for i in range(len(train_data)):
        atoms = train_data[i][0]
        struct = aaa.get_molecule(atoms, cls=None)
        structures.append(struct)
        atoms_list = atoms.get_atomic_numbers()
        Fe_num, C_num = get_atoms(atoms_list)
        labels.append(train_data[i][1] - Fe*Fe_num - C*C_num)
    return structures, labels

def process2(test_data):
    structures, labels = [], []
    for i in range(len(test_data)):
        atoms = test_data[i][0]
        struct = aaa.get_molecule(atoms, cls=None)
        structures.append(struct)
        labels.append(test_data[i][1])
    return structures, labels

def solute(combine):
    X, Y = [], []
    for i in combine:
        a = np.array([[i[0][0], i[0][1]], [i[1][0], i[1][1]]])
        b = np.array([i[0][2], i[1][2]])
        if np.linalg.matrix_rank(a) == 2:
            rel = linalg.solve(a, b)
        print(rel)
        X.append(rel[0])
        Y.append(rel[1])
    return X, Y

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

for i in range(6):
    test_data = dataset[i]
    train_vali_data = []
    for j in range(len(dataset)):
        if i == j:
            continue
        train_vali_data += dataset[j]

    #to obtain the numbers of Fe、C and attribute of one molecule
    equation = []
    for i in range(len(train_vali_data)):
        atoms_list = train_vali_data[i][0].get_atomic_numbers()
        Fe, C = get_atoms(atoms_list)
        equation.append([Fe, C, train_vali_data[i][1]])

    #average attribute
    new_equation = renew(equation)

    combine = list(itertools.combinations(new_equation, 2))

    X, Y = solute(combine)

    Fe = np.mean(X)
    C = np.mean(Y)

    train_vali_structures = []
    train_vali_labels = []

    train_structures = []
    train_labels = []
    vali_structures = []
    vali_labels = []
    test_structures = []
    test_labels = []

    train_vali_structures, train_vali_labels = process1(train_vali_data, Fe, C)
    test_structures, test_labels = process2(test_data)

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
                    verbose=2,
                    patience=10)

        prediction = []

        predict = model.predict_structure(test_structures)

        for j, delE in enumerate(predict):
            atoms_test = test_data[j][0]
            atoms_list = atoms_test.get_atomic_numbers()
            Fe_num, C_num = get_atoms(atoms_list)
            prediction.append(delE + Fe_num * Fe + C_num * C)

    loss = loss_mae(prediction, test_labels)
    Loss.append(loss)

print('Loss:', Loss)
print('MAE:', np.mean(Loss))