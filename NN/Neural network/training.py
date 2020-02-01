import numpy as np

# activation
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# derivative of activation
der = lambda x: x * (1 - x)
learning_rate = 0.8
# hidden layer size
h = 54
# number of classes
cl = 10

X = np.zeros((1, 785))
path = '~/training'
print('Reading data...')
for number in '0123456789':
    print('Reading', number)
    cur_path = '{}'.format(path + '/' + number + '_data')
    data = np.loadtxt(cur_path, usecols=range(785), delimiter=',')
    X = np.append(X, data, axis=0)

X = np.delete(X, 0, 0)
np.random.shuffle(X)
Target = -np.ones((X.shape[0], cl))
for i in range(X.shape[0]):
    Target[i, int(X[i, -1])] = 1.0
X = X[:, :-1]
X = np.append(X, np.ones((X.shape[0], 1)), 1)

# weights in->hidden
W1 = 1 / X.shape[1] * np.random.random_sample((X.shape[1], h)) - 1 / (2 * X.shape[1])
# weights hidden->out, hid+"bias unit"
W2 = 1 / h * np.random.random_sample((h + 1, cl)) - 1 / (2 * h)
Q = 0
parts = 10
objects = int(np.floor(X.shape[0] / parts))
print('Training...')
for it in range(1000):
    for i in range(parts):
        x = X[i * objects: i * objects + objects, :]
        t = Target[i * objects:i * objects + objects, :]
        # forward pass
        # dimOut1 = examples*hid
        Out1 = sigmoid(x.dot(W1))
        Out1 = np.append(Out1, np.ones((Out1.shape[0], 1)), 1)
        # output for all objects, dimOut2 = (examples*out)
        Out2 = sigmoid(Out1.dot(W2))

        out_classes = np.argmax(Out2, axis=1)
        target_classes = np.argmax(t, axis=1)
        # all digits are classified correctly
        if np.array_equal(out_classes, target_classes):
            print('Full match!')
            break
        else:
            print('Mismatch count ', np.count_nonzero(np.not_equal(out_classes, target_classes)))

        err = np.sum((0.5 * np.square((t-Out2))), axis=1).mean()
        print('Current error ', err, '; it ', it)

        # backward pass
        # input layer error, hidden<-out, dimDelta2=examples*out
        Delta2 = der(Out2) * (Out2 - t)
        # hidden layer error, in<-hidden, dimDelta1=examples*hidden+1
        Delta1 = (der(Out1) * Delta2.dot(W2.T))[:, :-1]

        Grad1 = x.T.dot(Delta1) / x.shape[0]
        Grad2 = Out1.T.dot(Delta2) / x.shape[0]
        # weights update
        W1 = W1 - learning_rate * Grad1
        W2 = W2 - learning_rate * Grad2

        print('Saving weights...')
        np.savetxt('W1', W1, delimiter=',')
        np.savetxt('W2', W2, delimiter=',')

print('Training completed!')
