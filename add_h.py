import numpy as np
import pickle
from scipy.spatial import ConvexHull
import math
from ase import Atoms

np.random.seed(123456789)

#求质心
def get_centroid(position):
    centroid = list(position.mean(axis=0)) #列求平均
    return centroid

def get_para(position):
    p, q, r = position[0], position[1], position[2]

    a = ((q[1] - p[1]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[1] - p[1]))
    b = ((q[2] - p[2]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[2] - p[2]))
    c = ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    d = (0 - (a * p[0] + b * p[1] + c * p[2]))

    return [a, b, c, d]

f = open('./clusters.pkl', 'rb')
dataset = pickle.load(f)

for c in range(len(dataset)):

    position = dataset[c].get_positions()
    print(dataset[c])

    if len(dataset[c]) > 3:

        hull = ConvexHull(position, qhull_options='QJ')

        #面方程的参数
        equations = hull.equations
        #构成面的三个点的索引
        sim = hull.simplices

        data_h = []

        print(equations)

        for i in range(len(equations)):
            #质心坐标加一维，方便带入面方程
            centroid = get_centroid(position)

            centroid.append(1)

            p = []
            #得到构成平面的三个点的坐标
            for id in sim[i]:
                p.append(position[id])

            #质心到面的距离
            # print("centroid:", centroid)
            print("equations", i, ':', equations[i])
            d = abs(np.dot(centroid, equations[i]))

            face_para = []

            for para in equations[i]:
                face_para.append(para)

            face_para[3] += math.sqrt(math.pow(equations[i][0], 2) + math.pow(equations[i][1], 2)
                                      + math.pow(equations[i][2], 2))

            y_min = np.array(position).min(axis=0)[1]
            y_max = np.array(position).max(axis=0)[1]
            z_min = np.array(position).min(axis=0)[2]
            z_max = np.array(position).max(axis=0)[2]

            y = np.random.uniform(y_min, y_max, 50)
            z = np.random.uniform(z_min, z_max, 50)

            x = []

            if abs(np.dot(centroid, face_para)) > d:

                for j in range(len(y)):
                    x.append((face_para[1]*y[j] + face_para[2]*z[j] + face_para[3]) / (-face_para[0]))

            else:
                for j in range(len(y)):
                    x.append((equations[i][1]*y[j] + equations[i][2]*z[j] + equations[i][3]
                        - math.sqrt(math.pow(equations[i][0], 2) + math.pow(equations[i][1], 2)
                        + math.pow(equations[i][2], 2))) / (-equations[i][0]))

            for k in range(len(y)):
                cluster = Atoms.copy(dataset[c])

                h = Atoms('H', positions=[(x[k], y[k], z[k])])
                cluster += h
                data_h.append(cluster)

        with open('./hull_data/clusters_with_h_' + str(c+1) + '.pkl', 'wb') as f:
            pickle.dump(data_h, f)

    else:
        equations = get_para(position)

        x_min = np.array(position).min(axis=0)[0]
        x_max = np.array(position).max(axis=0)[0]
        y_min = np.array(position).min(axis=0)[1]
        y_max = np.array(position).max(axis=0)[1]

        x = np.random.uniform(x_min, x_max, 50)
        y = np.random.uniform(y_min, y_max, 50)

        for i in range(2):
            z = []
            if i < 1:
                # 上面
                for j in range(len(x)):
                    z.append((equations[0] * x[j] + equations[1] * y[j] + equations[3]
                    + math.sqrt(math.pow(equations[0], 2) + math.pow(equations[1], 2)
                    + math.pow(equations[2], 2))) / (-equations[2]))
            else:
                # 下面
                for j in range(len(x)):
                    z.append((equations[0] * x[j] + equations[1] * y[j] + equations[3]
                    - math.sqrt(math.pow(equations[0], 2) + math.pow(equations[1], 2)
                    + math.pow(equations[2], 2))) / (-equations[2]))

            data_h = []

            for j in range(len(x)):
                cluster = []
                cluster.append(dataset[c])
                h = Atoms('H', positions=[(x[j], y[j], z[j])])
                cluster += h
                data_h.append(cluster)
            with open('./hull_data/clusters_with_h_' + str(i+1) + '.pkl', 'wb') as f:
                pickle.dump(data_h, f)










