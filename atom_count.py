import pickle
import ase

f = open('/data2/deh/data/KF6/test_data6.pkl', 'rb')
dataset = pickle.load(f)

sum_cnt = 0

for row in dataset:
    print(row[0])
    sum_cnt += len(row[0])

print(sum_cnt/len(dataset))
