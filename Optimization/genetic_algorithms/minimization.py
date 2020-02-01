import sys
import numpy as np
import random
from fractions import Fraction

# Genetic algorithm for function minimization
class Minimization:
    def __init__(self, coefs, size, interval, single_point=True, threshold=0.0, n=False):
        self.coefs = np.array(coefs[::-1])
        self.size = size  # population size
        self.lower = interval[0]  # lower bound of search segment
        self.upper = interval[1]  # upper bound
        # number of genes
        self.width = len(np.binary_repr(self.upper))
        # is one-point crossingover
        self.single_point = single_point
        # threshold for mutation probability
        self.threshold = threshold
        # single mutation or until average fitness increase
        print('One-point crossingover') if single_point else print('Crossingover with offset')
        print('Random mutation') if threshold == 0 \
            else print('Random mutation with probability threshold <{}'.format(self.threshold))
        print('Mutation untill average fitness increase') if n \
            else print('Single mutation')
        self.find_min()

    def find_min(self):
        parents = self.parent_generation()
        i = 0
        while True:
            print("Iteration no.: {}".format(i))
            # parent generation in binary format
            bin_parents = self.to_binary(parents)
            print("Parent generation: {}, {}".format(parents, bin_parents))

            fit_values = self.fitness(parents)
            print("Fitness function values: ", fit_values)

            pairs = self.select_pairs(fit_values)
            print("Pairs for crossing: \n", pairs)

            if self.single_point:  # one-point crossingover
                new_generation, decimal = self.crossover1(pairs, bin_parents)
            else:  # crossingover with offset
                new_generation, decimal = self.crossover2(pairs, bin_parents)
            fit_values = self.fitness(decimal)
            mutations = 0
            while True:  # mutation loop
                if self.threshold == 0:  # random mutation
                    self.mutation1(new_generation)
                else: 
                    self.mutation2(new_generation, self.threshold)
                new_decimal = self.to_decimal(new_generation)
                new_fit_values = self.fitness(new_decimal)
                if not self.multiplicity:
                    break
                else:
                    mutations += 1
                    before_avg = np.average(fit_values)
                    after_avg = np.average(new_fit_values)
                    print('Average value of fitness function before mutation {:.3f}, after mutation {:.3f}'.
                          format(before_avg, after_avg))
                    if after_avg <= before_avg:
                        print('Number of mutations ', mutations)
                        break
                    else:
                        print("After mutation:\n", new_generation, new_decimal, new_fit_values)
            print("After mutation:\n", new_generation, new_decimal, new_fit_values)
            print("{:_<100}".format(""))
            if np.array_equal(parents, decimal) and fit_values.min() == new_fit_values.min():
                min_index = fit_values.argmin()
                print('min = {}'.format(decimal[min_index]))
                break
            total = np.concatenate((parents, decimal), axis=0)
            all_fit = self.fitness(total)
            ind = np.argsort(all_fit) 
            parents = total[ind[:self.size]]
            i +=1


    def parent_generation(self):
        parents = random.sample(range(self.lower, self.upper + 1), self.size)
        return parents


    def to_binary(self, parents):
        bin_parents = np.array([np.binary_repr(x, self.width) for x in parents])
        return bin_parents


    def to_decimal(self, generation):
        return np.array([int(g, 2) for g in generation])

    
    def fitness(self, sample):
        fit_values = []
        for x in sample:
            res = 0.0
            for y in self.coefs[:-1]:
                res = (res + y) * x
            res += self.coefs[-1]
            fit_values.append(res)
        return np.array(fit_values)


    def generate_weights(self, fit_val):
        n = sum(np.absolute(fit_val))
        weights = []
        for val in fit_val:
            p = 1 - (val / n)
            weights.append(weights[-1] + p if len(weights) > 0 else p)
        return np.array(weights)

    def find_index(self, rand, weights):
        w = 0
        while weights[w] < rand:
            w += 1
        return w


    def select_pairs(self, fit_val):
        weights = self.generate_weights(fit_val)
        max_val = weights[-1]
        pairs = []
        for i in range(int(np.ceil(self.size / 2))):
            ind1 = 0; ind2 = 0
            pairs.append([ind1, ind2])

            while ind1 == ind2 or [ind1, ind2] in pairs[:-1]:
                rand_weights = max_val * np.random.sample(2)
                ind1 = self.find_index(rand_weights[0], weights)
                ind2 = self.find_index(rand_weights[1], weights)
                pairs[-1] = [ind1, ind2]
        return pairs[:self.size]

    # crossover by point
    def single(self, parent1, parent2, point):
        samples = []
        samples.append(parent1[:point] + parent2[point:])
        samples.append(parent2[:point] + parent1[point:])
        return samples

    def crossover1(self, pairs, parents, single_point=None):
        new_generation = []
        if not single_point:
            single_point = np.random.randint(1, self.width)
        print("One-point crossingover by gene {}".format(single_point))

        for pair in pairs:
            new = self.single(parents[pair[0]], parents[pair[1]], single_point)
            new_generation += new
        decimal = self.to_decimal(new_generation)
        print("Children: ", new_generation[:self.size], decimal[:self.size])
        return new_generation[:self.size], decimal[:self.size]


    def crossover2(self, pairs, parents):
        new_generation = []
        i = 1
        for pair in pairs:
            new = self.single(parents[pair[0]], parents[pair[1]], i)
            new_generation += new
            i = i + 1 if i < len(self.coefs) - 1 else 1
        decimal = self.to_decimal(new_generation)
        print("Children: ", new_generation[:self.size], decimal[:self.size])
        return new_generation[:self.size], decimal[:self.size]


    def mutation1(self, generation):
        n = np.random.random_integers(1, self.size)
        prob = np.random.random_sample(self.size)
        prob = prob / np.sum(prob)
        print("Mutation probabilities: ", prob)
        ind = np.random.choice(self.size, n, replace=False, p=prob)
        print("Individual numbers for mutaion: ", ind)
        for i in range(n):
            pos = np.random.randint(0, self.width)
            generation[ind[i]] = generation[ind[i]][:pos] + \
                                    ('1' if generation[ind[i]][pos] == '0' else '0') + \
                                    generation[ind[i]][pos + 1:]

    def mutation2(self, generation, threshold):
        # mutation probabilities
        prob = np.random.random_sample(self.size)
        prob = prob / np.sum(prob)
        print("Mutation probabilities: ", prob)
        ind = np.argwhere(prob < threshold)[:, 0]
        print("Individual numbers for mutation: ", ind)
        for i in ind:
            # gene position for mutation
            pos = np.random.randint(0, self.width)
            print(i, pos, generation)
            generation[i] = generation[i][:pos] + \
                                    ('1' if generation[i][pos] == '0' else '0') + \
                                    generation[i][pos + 1:]



coefs = []
size = int(input("Number of individuals: "))
print("Equation coefficients: ")
line = sys.stdin.readline()
for num in line.split():
    coefs.append(float(Fraction(num)))

print("Search interval: ")
interval = [int(x) for x in sys.stdin.readline().split()]

Minimization(coefs, size, interval, single_point=True, threshold=0.3, n=False)
