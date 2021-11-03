import pickle

f = open('FeC_pca.pkl', 'rb')
data = pickle.load(f)

l = []

for k, value in data.items():
    l.append(list(value[0]))

L = l[0] + l[1] + l[2] + l[3] + l[4] + l[5]
L = [list(data) for data in L]

test = l[0]

train = l[1] + l[2] + l[3] + l[4] + l[5]

print(len(test))
print(len(train))

x_min = min(L)[0]
x_max = max(L)[0]
y_min = 0
y_max = -10000

for i in range(len(L)):
    if L[i][1] < y_min:
        y_min = L[i][1]
    if L[i][1] > y_max:
        y_max = L[i][1]

rel_test, rel_train = [], []
for i in range(len(test)):
    rel_test.append([(test[i][0]-x_min)/(x_max-x_min), (test[i][1]-y_min)/(y_max-y_min)])

for i in range(len(train)):
    rel_train.append([(train[i][0]-x_min)/(x_max-x_min), (train[i][1]-y_min)/(y_max-y_min)])


dict = {'test': rel_test, 'train': rel_train}

with open('./pca_data/Fe1_test.pkl', 'wb') as f:
    pickle.dump(dict, f)
