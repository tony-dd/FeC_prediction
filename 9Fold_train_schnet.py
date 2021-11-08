import os
import schnetpack as spk
from ase.db import connect
import random
import numpy as np
from schnetpack.datasets import MD17

seed = 1234
random.seed(seed)
np.random.seed(seed)

forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

fec_data = MD17('FeC_train_dataset.db')

print(len(fec_data))

train, val, test = spk.train_test_split(
        data=fec_data,
        num_train=int(len(fec_data) * 0.9),
        num_val=len(fec_data) - int(len(fec_data) * 0.9),
        split_file=os.path.join(forcetut, "split.npz"),
    )

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
    negative_dr=True
)

model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

import torch

# tradeoff
rho_tradeoff = 0.1

# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch[MD17.energy]-result[MD17.energy]
    err_sq_energy = torch.mean(diff_energy ** 2)

    return err_sq_energy

from torch.optim import Adam

# build optimizer
optimizer = Adam(model.parameters(), lr=5e-4)

import schnetpack.train as trn

# set up metrics
metrics = [spk.metrics.MeanAbsoluteError(MD17.energy)]

# construct hooks
hooks = [
    trn.CSVHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=10, factor=0.8, min_lr=5e-4,
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
    device = "cuda:1"
else:
    device = "cpu"

# determine number of epochs and train
n_epochs = 500
trainer.train(device=device, n_epochs=n_epochs)
