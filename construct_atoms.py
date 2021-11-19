from ase import Atoms

FeC = []

for i in range(300):
    FeC1 = Atoms('FeC', positions=[(0, 0, 0), (i+1, 0, 0)])
    FeC2 = Atoms('FeC', positions=[(0, 0, 0), (i+1, i+1, 0)])
    FeC3 = Atoms('FeC', positions=[(0, 0, 0), (i+1, i+1, i+1)])

    FeC.append(FeC1.get_positions())
    FeC.append(FeC2.get_positions())
    FeC.append(FeC3.get_positions())

print(FeC)

