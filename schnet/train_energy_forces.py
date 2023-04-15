import schnetpack as spk
from ase.db import connect
import random
import numpy as np
from schnetpack.datasets import MD17
from schnetpack.data import ConcatAtomsData
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

seed = 1234
random.seed(seed)
np.random.seed(seed)

dataset, test_data, train_vali_data = [], [], []

file_name = os.listdir('../FeC')

for item in file_name:
    data = MD17(item)
    train_vali_data += data[:len(data) * 0.9]
    test_data += data[len(data) * 0.9:]

L_e, L_f = [], []

forcetut = './train_energy_forces'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

print('test:', len(test_data))
print('train:', len(train_vali_data))

train, val, _ = spk.train_test_split(
    data=train_vali_data,
    num_train=int(len(train_vali_data) * 0.9),
    num_val=len(train_vali_data) - int(len(train_vali_data) * 0.9),
    split_file=os.path.join(forcetut, "split.npz"),
)

atoms, properties = train_vali_data.get_properties(0)

print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

print('Forces:\n', properties[MD17.forces])
print('Shape:\n', properties[MD17.forces].shape)

train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)

means, stddevs = train_loader.get_statistics(
    spk.datasets.MD17.energy, divide_by_atoms=True
)

n_features = 128

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=25,
    n_interactions=3,
    cutoff=5.,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)

energy_model = spk.atomistic.Atomwise(
    n_in=n_features,
    property=MD17.energy,
    mean=means[MD17.energy],
    stddev=stddevs[MD17.energy],
    derivative=MD17.forces,
    negative_dr=True
)

model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

import torch

# tradeoff
rho_tradeoff = 0.1

# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch[MD17.energy] - result[MD17.energy]
    err_sq_energy = torch.mean(diff_energy ** 2)

    # compute the mean squared error on the forces
    diff_forces = batch[MD17.forces] - result[MD17.forces]
    err_sq_forces = torch.mean(diff_forces ** 2)

    # build the combined loss function
    err_sq = rho_tradeoff * err_sq_energy + (1 - rho_tradeoff) * err_sq_forces

    return err_sq

from torch.optim import Adam

# build optimizer
optimizer = Adam(model.parameters(), lr=5e-4)

import schnetpack.train as trn

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError(MD17.energy),
    spk.metrics.MeanAbsoluteError(MD17.forces)
]

# construct hooks
hooks = [
    trn.CSVHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=2, factor=0.2, min_lr=1e-4,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=forcetut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# determine number of epochs and train
n_epochs = 500
trainer.train(device=device, n_epochs=n_epochs)

test = MD17('./data/energy_force_data/Fe' + str(i+1) + 'Cx_prop.db')

print('len(test):', len(test))

test_loader = spk.AtomsLoader(test, batch_size=100)

best_model = torch.load(os.path.join(forcetut, 'best_model'))

energy_error = 0.0
forces_error = 0.0

for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}

    # apply model
    pred = best_model(batch)

    # calculate absolute error of energies

    print('pred energy:', pred[MD17.energy])
    print('batch energy:', batch[MD17.energy])

    tmp_energy = torch.sum(torch.abs(pred[MD17.energy] - batch[MD17.energy]))
    tmp_energy = tmp_energy.detach().cpu().numpy()  # detach from graph & convert to numpy
    energy_error += tmp_energy

    # calculate absolute error of forces, where we compute the mean over the n_atoms x 3 dimensions
    print('pred forces:', pred[MD17.forces])
    print('batch forces:', batch[MD17.forces])
    tmp_forces = torch.sum(
        torch.mean(torch.abs(pred[MD17.forces] - batch[MD17.forces]), dim=(1, 2))
    )


    tmp_forces = tmp_forces.detach().cpu().numpy()  # detach from graph & convert to numpy
    forces_error += tmp_forces

    # log progress
    percent = '{:3.2f}'.format(count / len(test_loader) * 100)
    print('Progress:', percent + '%' + ' ' * (5 - len(percent)), end="\r")

energy_error /= len(test)
forces_error /= len(test)

L_e.append(energy_error)
L_f.append(forces_error)

print('\nTest MAE:')
print('energy: {:10.3f} eV'.format(energy_error))

print('energy_mae:', L_e)
print('mean_energy_mae:', np.mean(L_e))
print('force_mae:', L_f)
print('mean_force_mae:', np.mean(L_f))