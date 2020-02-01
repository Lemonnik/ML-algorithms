import sys
import random
import numpy as np

"""
Genetic algorithm for diophantine equation solving
"""
class DiophantineEquation:
    def __init__(self, coefs, size):
        # equation coefficients
        self.coefs = np.array(coefs)
        # left value of search segment
        self.lower = 1
        # right value
        self.upper = self.coefs[-1]
        # number of genes that individual has to
        self.genes_quantity = len(coefs) - 1
        # population size
        self.size = size

        self.parents = self.parent_generation()
        print('Parent generation:')
        for i in range(len(self.parents)):
            print('{:2d}. {}'.format(i, self.parents[i]))
        self.solve(self.parents)

    def parent_generation(self):
        parents = np.zeros((self.size, self.genes_quantity))
        for i in range(self.size):
            parents[i, :] = np.random.randint(self.lower, self.upper, self.genes_quantity)
        return parents

    def fitness(self, sample):
        fit_values = np.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            f = 0.0
            for j in range(self.genes_quantity):
                f += self.coefs[j] * sample[i, j]
            f -= self.coefs[-1]
            fit_values[i] = f
        return fit_values


    def generate_weights(self, fit_val):
        n = np.sum(np.absolute(fit_val))
        weights = np.zeros(fit_val.size)
        i = 0
        for val in fit_val:
            t = 1 - (val / n)
            weights[i] = (weights[i - 1] + t if i > 0 else t)
            i += 1
        return weights

    # interval in weights that number rand belongs to
    def find_index(self, rand, weights):
        w = 0
        while w < weights.size and weights[w] < rand:
            w += 1
        return w


    def selection(self, fit_values, parents):
        weights = self.generate_weights(fit_values)
        s = weights[-1]
        new_generation = np.zeros((self.size, self.genes_quantity))
        num = np.zeros((self.size, 2))
        j = 1
        print("Crossing: ")
        for i in range(self.size):
            num1 = 0; num2 = 0
            while num1 == num2 or [num1, num2] in num.tolist():
                rand_weights = s * np.random.sample(2)
                num1 = self.find_index(rand_weights[0], weights)
                num2 = self.find_index(rand_weights[1], weights)
            num[i, :] = [num1, num2]
            new_generation[i, :] = self.crossover(parents[num1], parents[num2], j)
            print('({},{}) by gene {}'.format(num1, num2, j), new_generation[i, :])
            j = 1 if j == self.genes_quantity - 1 else j + 1
        return new_generation

    def crossover(self, a, b, i):
        return np.append(a[:i], b[i:])

    def mutation(self, fit_val, sample):
        before_avg = np.average(fit_val)
        print('Average value of fitness function before mutation: ', before_avg)
        while True:
            # individuals for mutation
            q = random.randint(1, self.size)
            # mutation probability
            """prob = np.random.random_sample(self.size)
            prob = prob / np.sum(prob)"""
            prob = fit_val / np.sum(np.absolute(fit_val))
            ind = np.random.choice(self.size, q, replace=False, p=prob)
            print("Numbers of individuals: ", ind)
            print("Mutations: ")
            for i in ind:
                # random selection of gene position for mutation
                pos = np.random.randint(0, self.genes_quantity)
                #change = sample[i, pos] if sample[i, pos] > self.lower else self.lower + 2
                sample[i, pos] = np.random.randint(self.lower, self.upper + 1)
                print('Individual {}, number of gene: {}, mutation {}'.format(i, pos, sample[i]))
            new_fit = self.fitness(sample)
            after_avg = np.average(new_fit)
            print('Average value of fitness function after mutation: ', after_avg)
            if after_avg <= before_avg:
                break
            fit_val = new_fit
        return sample

    def solve(self, parents):
        it = 1
        while True:
            print("Number of iteration: {}".format(it))
            fit_values_parents = self.fitness(parents)
            print("Fitness function values for parent generation: ", fit_values_parents)
            # individuals selection and crossing according to fitness function values
            new_generation = self.selection(fit_values_parents, parents)
            fit_values_children = self.fitness(new_generation)
            print("Fitness function values for children: ", fit_values_children)
            if not np.all(fit_values_children):
                args = np.where(fit_values_children == 0)[0]
                print("Solutions: \n", new_generation[args])
                break

            self.mutation(fit_values_children, new_generation)
            fit_values_children = self.fitness(new_generation)
            print("Generation after mutations: \n", new_generation)
            print("Fitness function values after mutations: ", fit_values_children)
            if not np.all(fit_values_children):
                args = np.where(fit_values_children == 0)[0]
                print("Solutions: \n", new_generation[args])
                break

            # select individuals from parents and new_generation with lowest values of fitness function
            best_ind = np.argsort(np.append(fit_values_parents, fit_values_children))
            parents = np.concatenate((parents, new_generation), axis=0)[best_ind[:self.size], :]
            print("New parent generation: ")
            for i in range(self.size):
                print('{:2d}. {}'.format(i, parents[i]))
            print("{:_<100}".format(""))
            it += 1


coefs = []
size = int(input("Number of individuals: "))
print("Equation coefficients: ")
line = sys.stdin.readline()
for num in line.split():
    coefs.append(int(num))

d = DiophantineEquation(coefs, size)

