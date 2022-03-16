import numpy as np
import pickle
from scipy.spatial import ConvexHull
import math
from ase import Atoms

#求质心
def get_centroid(position):
    centroid = position.mean(axis=0) #列求平均
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

for cluster in dataset:
    position = cluster.get_positions()

    if len(cluster) > 3:

        centroid = get_centroid(position)

        hull = ConvexHull(position)

        #面方程的参数
        equations = hull.equations
        #构成面的三个点的索引
        sim = hull.simplices

        data_h = []

        for i in range(len(equations)):
            #质心坐标加一维，方便带入面方程
            centroid.append(1)

            p = []
            #得到构成平面的三个点的坐标
            for id in sim[i]:
                p.append(position[id])

            print('p:', p)

            #质心到面的距离
            d = abs(np.dot(centroid, equations[i]))/math.sqrt(math.pow(equations[i][0], 2) + math.pow(equations[i][1], 2)
                    + math.pow(equations[i][2], 2))

            face_para = []

            for para in equations[i]:
                face_para.append(para)

            face_para[3] += math.sqrt(math.pow(equations[i][0], 2) + math.pow(equations[i][1], 2)
                                      + math.pow(equations[i][2], 2))

            x_min, x_max, y_min, y_max = 100, 0, 100, 0

            for j in range(len(p)):
                if p[j][0] < x_min:
                    x_min = p[j][0]
                if p[j][0] > x_max:
                    x_max = p[j][0]
                if p[j][1] < y_min:
                    y_min = p[j][1]
                if p[j][1] > y_max:
                    y_max = p[j][1]

            x = np.random.uniform(x_min, x_max, 50)
            y = np.random.uniform(y_min, y_max, 50)

            if abs(np.dot(centroid,face_para))/math.sqrt(math.pow(face_para[0], 2) + math.pow(face_para[1], 2)
                    + math.pow(face_para[2], 2)) > d:

                z = (face_para[0] * x + face_para[1] * y + face_para[3]) / (-equations[2])

            else:
                z = (equations[i][0] * x + equations[i][1] * y + equations[i][3]
                    - math.sqrt(math.pow(equations[i][0], 2) + math.pow(equations[i][1], 2)
                    + math.pow(equations[i][2], 2))) / (-equations[2])

            for k in range(len(x)):
                h = Atoms('H', positions=[(x[k], y[k], z[k])])
                cluster += h
                data_h.append(cluster)
        with open('./clusters_with_h_' + str(i+3) + '.pkl', 'wb') as f:
            pickle.dump(data_h, f)




    else:
        equations = get_para(position)

        x_min, x_max, y_min, y_max = 100, 0, 100, 0

        for i in range(len(position)):
            if position[i][0] < x_min:
                x_min = position[i][0]
            if position[i][0] > x_max:
                x_max = position[i][0]
            if position[i][1] < y_min:
                y_min = position[i][1]
            if position[i][1] > y_max:
                y_max = position[i][1]

        x = np.random.uniform(x_min, x_max, 50)
        y = np.random.uniform(y_min, y_max, 50)

        for i in range(2):
            if i < 1:
                # 上面
                z = (equations[0] * x + equations[1] * y + equations[3]
                    + math.sqrt(math.pow(equations[0], 2) + math.pow(equations[1], 2)
                    + math.pow(equations[2], 2))) / (-equations[2])
            else:
                # 下面
                z = (equations[0] * x + equations[1] * y + equations[3]
                    - math.sqrt(math.pow(equations[0], 2) + math.pow(equations[1], 2)
                    + math.pow(equations[2], 2))) / (-equations[2])

            data_h = []

            for j in range(len(x)):
                h = Atoms('H', positions=[(x[j], y[j], z[j])])
                cluster += h
                data_h.append(cluster)
            with open('./clusters_with_h_' + str(i+1) + '.pkl', 'wb') as f:
                pickle.dump(data_h, f)










