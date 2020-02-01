import numpy as np

path = '~/testing'

# activation
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# hidden layer size
h = 54
# number of classes
cl = 10

X = np.zeros((1, 785))
print('Reading data...')
Labels = []
for number in '0123456789':
    print(number)
    cur_path = path + "/" + number + "_data"
    # data for one digit
    # dim(X)=examples*features, features = 784(28*28)
    data = np.loadtxt(cur_path, usecols=range(785), delimiter=',')
    X = np.append(X, data, axis=0)
    Labels += [int(number)] * data.shape[0]

X = np.delete(X, 0, 0)
X = X[:, :-1]

X = np.append(X, np.ones((X.shape[0], 1)), 1)

# weights in->hidden
W1 = np.loadtxt('W1', usecols=range(h), delimiter=',')
# weights hidden->out, hid+"bias unit"
W2 = np.loadtxt('W2', usecols=range(cl), delimiter=',')

print('Testing...')
# forward pass
# dimOut1 = examples*hid
Out1 = sigmoid(X.dot(W1))
# ones column
Out1 = np.append(Out1, np.ones((Out1.shape[0], 1)), 1)
# sample output,  dimOut2=examples*out
Out2 = sigmoid(Out1.dot(W2))

Res = np.argmax(Out2, axis=1)
Labels = np.array(Labels)
print(np.count_nonzero(Res == Labels),  X.shape[0])
accuracy = np.count_nonzero(Res == Labels) * 100 / Labels.size
print(accuracy)
