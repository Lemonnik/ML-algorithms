import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    def __init__(self, c, X):
        self.X = X
        # learning rate
        self.c = c
        # weights
        self.w = np.random.random(3)
        # neuron response for x
        self.f = lambda x: np.sign(np.sum(x[:-1] * self.w))
        # value of separating line in x
        self.y = lambda x: (-self.w[2] - self.w[0] * x) / self.w[1]
        # number of iterations
        self.it = 0
        self.x = [np.min(X[:, 0]), np.max(self.X[:, 0])]

        self.fig, self.ax = plt.subplots()
        # enter press
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        self.draw_objects()
        plt.show()

    # draw points and separating line
    def draw_objects(self):
        obj = []
        k1 = 0
        k2 = 0
        for i in range(self.X.shape[0]):
            x1 = self.X[i, 0]
            x2 = self.X[i, 1]
            if self.f(self.X[i]) == -1:
                obj1, = self.ax.plot(x1, x2, 'ro', label=r'$K_2$', markersize=4)
                if k1 == 0:
                    obj.append(obj1)
                    k1 += 1
            else:
                obj2, = self.ax.plot(x1, x2, 'cs', label=r'$K_1$', markersize=4)
                if k2 == 0:
                    obj.append(obj2)
                    k2 += 1
            self.ax.annotate('(%.1f; %.1f)' % (x1, x2), xy=(x1, x2), fontSize=7)
        line, = self.ax.plot(self.x, [self.y(self.x[0]), self.y(self.x[1])],
                             label='%.1f$x_1+$ %.1f$x_2+$ %.1f=0' % (self.w[0], self.w[1], self.w[2]))
        plt.setp(line, color='k', linewidth=1.5)
        plt.legend(handles=obj+[line])

    # enter press event, one pass per sample
    def press(self, event):
        if event.key == 'enter':
            # first object with incorrect answer
            mismatch = self.check()
            if mismatch != -1:
                self.it += 1
                self.ax.clear()
                # weights correction
                self.train(mismatch)
                out = "Number of passes per sample: {}".format(self.it)
                self.draw_objects()
            else:
                # mouse clicks
                self.fig.canvas.mpl_connect('button_press_event', self.click)
                out = "Number of passes per sample: {}.\nLearning completed.".format(self.it)
            self.ax.set_title(out, fontsize=9)
            # redraw
            self.fig.canvas.draw()

    # check neuron answers using weights self.w
    def check(self):
        for i in range(self.X.shape[0]):
            if self.f(self.X[i]) != self.X[i, -1]:
                return i
        return -1

    def train(self, j):
        for i in range(j, self.X.shape[0]):
            cur = self.f(self.X[i])
            true = self.X[i, -1]
            if cur != true:
                self.w = self.w + self.c * (true - cur) * self.X[i, :-1]

    # mouse click processing (for new objects)
    def click(self, event):
        x1 = event.xdata
        x2 = event.ydata
        new_obj = [x1, x2, 1, 0]
        if self.f(new_obj) == -1:
            self.ax.plot(x1, x2, 'ro', markersize=4)
        else:
            self.ax.plot(x1, x2, 'cs', markersize=4)
        self.ax.annotate('(%.1f; %.1f)' % (x1, x2), xy=(x1, x2), fontSize=7)
        self.fig.canvas.draw()


# test sample
X = np.array([[1, 1, 1, 1],
            [9.4, 6.4, 1, -1],
            [2.5, 2.1, 1, 1],
            [8, 7.7, 1, -1],
            [0.5, 2.2, 1, 1],
            [7.9, 8.4, 1, -1],
            [7, 7, 1, -1],
            [2.8, 0.8, 1, 1],
            [1.2, 3, 1, 1],
            [7.8, 6.1, 1, -1]])

# learning rate
c = 0.5
Neuron(c, X)
