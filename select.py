import pickle
import os.path

def get_long_axis(position):
    long_axis = 0
    for i in range(len(position)):
        for j in range(len(position[i])):
            if position[i][j] > long_axis:
                long_axis = position[i][j]
    return long_axis

def get_short_axis(position):
    short_axis = 10
    for i in range(len(position)):
        for j in range(len(position[i])):
            if 0 < position[i][j] < short_axis:
                short_axis = position[i][j]
    return short_axis

def get_ratio(position):
    short = get_short_axis(position)
    long = get_long_axis(position)
    ratio = long/short
    return ratio

dir="D:\FeC"

dataset = []

for parentdir,dirname,filenames in os.walk(dir):
    for file in filenames:
        f = open(dir + '/' + file, 'rb')
        data = pickle.load(f)

        for cluster in data:
            max, min = 0, 10000
            positions = cluster.get_all_distances()
            ratio = get_ratio(positions)

            if ratio > max:
                max = ratio
                max_cluster = cluster
            if ratio < min:
                min = ratio
                min_cluster = cluster

        dataset.append(max_cluster)
        dataset.append(min_cluster)

print(len(dataset))

with open('./clusters.pkl', 'wb') as f:
    pickle.dump(dataset, f)



