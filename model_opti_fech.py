import schnetpack as spk
import os
from schnetpack.datasets import MD17
import pickle
import torch
import numpy as np
from ase.units import eV, A
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read

best_model = torch.load(os.path.join('/data2/deh/model/schnet_9fold/Best_schnet_models/cuda1_forcetut_5', 'best_model'))

device = "cpu"

calculator = spk.interfaces.SpkCalculator(
    model=best_model,
    device=device,
    energy=MD17.energy,
    forces=MD17.forces,
    energy_units='eV',
    forces_units='eV/A'
)

atoms = read('/data2/deh/calypso-FexCy-clusters/calypso-Fe5C5-cluste5---36yy/csp-CALYPSO-20190331-011643/POSCAR', format='vasp')

h = Atoms('H', positions=[(9, 10, 9)])

atoms += h

atoms.set_calculator(calculator)

print('Prediction:')
print('energy:', atoms.get_total_energy())
print('forces:', atoms.get_forces())

from ase import io

# Generate a directory for the ASE computations
ase_dir = os.path.join('./', 'model_opti_fech')

if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

# Write a sample molecule
molecule_path = os.path.join(ase_dir, 'fec.xyz')
io.write(molecule_path, atoms, format='xyz')

fec_ase = spk.interfaces.AseInterface(
    molecule_path,
    best_model,
    ase_dir,
    device,
    energy=MD17.energy,
    forces=MD17.forces,
    energy_units='eV',
    forces_units='eV/A'
)

fec_ase.optimize(fmax=1e-4)

