import pickle

f = open('FeC_pca_to_4.pkl', 'rb')
data = pickle.load(f)

l = []

for k, value in data.items():
    l.append(list(value[0]))

for m in range(len(l)):

    test = l[m]
    test_01 = []

    #take the first two dimensions of test
    for i in range(len(test)):
        test_01.append([[test[i][0]], test[i][1]])

    train, train_01 = [], []
    for j in range(len(l)):
        if i == j:
            continue
        train += l[j]

    # take the first two dimensions of train
    for i in range(len(train)):
        train_01.append([[train[i][0]], train[i][1]])

    L = train_01 + test_01

    print('len(test):', len(test))
    print('len(train):', len(train))

    x_min = min(L)[0][0]
    x_max = max(L)[0][0]
    y_min = 0
    y_max = -10000

    for i in range(len(L)):
        if L[i][1] < y_min:
            y_min = L[i][1]
        if L[i][1] > y_max:
            y_max = L[i][1]

    print('x_min:', x_min)
    print('x_max:', x_max)
    print('y_min:', y_min)
    print('y_max:', y_max)

    rel_test, rel_train = [], []
    for i in range(len(test_01)):
        rel_test.append([(test_01[i][0] - x_min) / (x_max - x_min), (test_01[i][1] - y_min) / (y_max - y_min)])

    for i in range(len(train_01)):
        rel_train.append([(train_01[i][0] - x_min) / (x_max - x_min), (train_01[i][1] - y_min) / (y_max - y_min)])

    dict = {'test': rel_test, 'train': rel_train}

    with open('./pca_data/Fe' + str(m+1) + '_test_01.pkl', 'wb') as f:
        pickle.dump(dict, f)



