import torch
import schnetpack as spk
from ase.db import connect
import random
import numpy as np
from schnetpack.datasets import MD17
from schnetpack.data import ConcatAtomsData
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def quchong(l):
    ll, tag = [], []
    for i in l:
        if i[1] not in tag:
            ll.append(i)
            tag.append(i[1])
    return ll

#sort and insert id
def insert_id(l):
    for i, item in enumerate(l):
        item.append(i + 1)
    return l

#return after prediction id_list
def get_id_lst(l):
    id_list = []
    for i in l:
        id_list.append(i[1])
    return id_list

def inverse_number(l):
    n = 0
    for i in range(len(l)):
        for j in range(i):
            if l[j] > l[i]:
                n += 1
    return n

def cal_acc(m, n):
    sum = 0
    for i in range(n):
        sum += i+1
    rel = 1 - (m/sum)
    return rel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

inverse, accuracy = [], []

for i in range(9):

    test = MD17('./Fe' + str(i+1) + 'Cx_sort.db')

    print('len(test):', len(test))

    test_loader = spk.AtomsLoader(test, batch_size=1)

    best_model = torch.load(os.path.join('/data2/deh/model/schnet_9fold/cuda0_forcetut_' + str(i+1), 'best_model'))

    energy_error = 0.0

    prediction = []

    for count, batch in enumerate(test_loader):
        # move batch to GPU, if necessary
        batch = {k: v.to(device) for k, v in batch.items()}

        # apply model
        pred = best_model(batch)

        prediction.append(pred['energy'].cpu().tolist()[0])

        # calculate absolute error of energies
        tmp_energy = torch.sum(torch.abs(pred[MD17.energy] - batch[MD17.energy]))
        tmp_energy = tmp_energy.detach().cpu().numpy()  # detach from graph & convert to numpy
        energy_error += tmp_energy

        # log progress
        percent = '{:3.2f}'.format(count / len(test_loader) * 100)
        print('Progress:', percent + '%' + ' ' * (5 - len(percent)), end="\r")

    energy_error /= len(test)

    prediction_and_id = insert_id(prediction)

    prediction_and_id.sort()

    id_lst = get_id_lst(prediction_and_id)

    inverse_num = inverse_number(id_lst)

    inverse.append(inverse_num)

    acc = cal_acc(inverse_num, len(test)-1)

    accuracy.append(acc)

    print('\nTest MAE:')
    print('    energy: {:10.3f} eV'.format(energy_error))

print('inverse:', inverse)
print('mean_inverse:', np.mean(inverse))

print('accuracy:', accuracy)
print('mean_accuracy:', np.mean(accuracy))