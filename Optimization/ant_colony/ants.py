import numpy as np


def init_paths(location):
    # cities crossed by ants
    paths = []
    for k in location.keys():
        paths.append([])
        paths[-1].append(location[k])
    return paths

# distances
D = np.array([[float('inf'), 38, 74, 59, 45],
             [38, float('inf'), 46, 61, 72],
             [74, 46, float('inf'), 49, 85],
             [59, 61, 49, float('inf'), 42],
             [45, 72, 85, 42, float('inf')]])

"""D = np.array([[float('inf'), 3, 2, 7, 3],
             [8, float('inf'), 7, 4, 3],
             [3, 5, float('inf'), 2, 6],
             [6, float('inf'), 8, float('inf'), 2],
             [5, 4, 4, 3, float('inf')]])"""


# initial value of pheromones on edges
tau = np.random.rand(D.shape[0], D.shape[1])

# pheromones trace weights
#alpha = np.random.random()
#beta = np.random.random()
#alpha = 0.1412
#beta = 0.666
alpha = 2
beta = 5
# optimal value cost
#Q = np.random.random()
#print(alpha, beta, Q)
Q = 20
# evaporation rate
rho = np.random.random()

# number of ants
m = D.shape[0]

# number of elite ants
e = 2

# colony life (on average, the working ant lives 1-3 years, 3 * 365 days = 1095)
t_max = 1095

# place ants in randomly selected cities (no matches)
places = np.random.permutation(D.shape[0])
# pairs city number:ant number
start_location = {i: places[i] for i in range(m)}
location = {i: places[i] for i in range(m)}
print(location)
allowable = np.ones((m, D.shape[0]))


paths = init_paths(location)
lengths = np.zeros(m)


L_best = float("inf")
T_best = []


for t in range(t_max):
    # deferred amount of pheromone on the edges
    delta_tau = np.zeros(D.shape)
    # loop by ants
    for k in range(m):
        # loop by k-th ant, until he visits all the acceptable cities
        while True:
            # number of the city where the k-ant is located at the current iteration
            start = location[k]
            # mark the city start as visited
            allowable[k, start] = False
            # acceptable cities for the k-th ant
            dest = np.argwhere(allowable[k] == True)[:, 0]
            dest = np.intersect1d(np.argwhere(D[location[k], :] != float('inf'))[:, 0], dest)
            if dest.size == 0:
                break
            # all numerators in the transition probability formula (vector)
            num = np.power(tau[start, dest], alpha) * np.power(1 / D[start, dest], beta)
            # denominator in the transition probability formula
            den = np.sum(num)
            # the transition probabilities of the k-th ant from the city location [k] to the city j (vector)
            P = num / den
            # choose a city randomly (but the city corresponding to the largest p)
            next = np.random.choice(dest, p=P)
            # add this city to the path of the k-th ant
            paths[k].append(next)
            # add path length
            lengths[k] += D[start, next]
            # change the current position of the k-th ant
            location[k] = next
        # close the outline
        lengths[k] += D[location[k], start_location[k]]
        paths[k].append(start_location[k])
        # the ant returned to the same city in which it was at first
        location[k] = start_location[k]
    # best solution check
    cur_len = np.min(lengths)
    cur_path = paths[np.argmin(lengths)]
    if cur_len < L_best: #and len(cur_path) == D.shape[0] + 1:
        L_best = cur_len
        T_best = cur_path

    # change the amount of pheromone
    for p in range(len(paths)):
        for q in range(len(paths[p])-1):
            delta_tau[paths[p][q], paths[p][q + 1]] += Q / lengths[p]

    # for elite ants
    delta_tau_e = np.zeros(D.shape)
    for p in range(len(T_best) - 1):
        delta_tau_e[T_best[p], T_best[p + 1]] = Q / L_best

    # update traces of pheromone on the edges
    tau = (1 - rho) * tau + delta_tau + e * delta_tau_e

    # sequence of cities traversed by ants
    paths = init_paths(location)
    # path length traversed by the k-th ant
    lengths = np.zeros(m)

print(T_best, L_best)
