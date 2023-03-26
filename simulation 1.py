import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

# speed of light is 1,
# we are using light seconds as a unit for distance.
k = 0.4714  # 2.582 * 10 ** 7
k_ = 1  # 10 ** -15 / 3


def abr(t):
    return ((abs(t) + 1) * np.sqrt(1 + 4 * abs(t)) - 1) / 3


def fun(a, tim, x0, y0):
    return (tim + abr(a)) ** 2 - ((a - x0) ** 2) - (y0 ** 2)


def fprime(a, tim, x0, y0):
    sgn = lambda t: t / abs(t)
    z = (abr(a) + tim) * (np.sqrt(1 + 4 * abs(a)) + 2 * (abs(a) + 1) / np.sqrt(1 + 4 * abs(a))) * sgn(a)
    return 2 * z + 2 * (a - x0)


class Charges(object):
    def __init__(self, t):
        self.t_o = t

    def pos_ret(self, a, b):
        return op.fsolve(fun, np.array([-10.0]), args=(self.t_o, a, b), fprime=fprime)[0]

    def neg_ret(self, a, b):
        return op.fsolve(fun, np.array([10.0]), args=(self.t_o, a, b), fprime=fprime)[0]

    def potential(self, a, b):
        (px, nx) = ([], [])
        for (x, y) in zip(a, b):
            if x ** 2 + y ** 2 <= self.t_o ** 2:
                px += [0]
                nx += [0]
            else:
                z1 = self.pos_ret(x, y)
                z2 = self.neg_ret(x, y)
                px += [1 / np.sqrt((z1 - x) ** 2 + y ** 2)]
                nx += [1 / np.sqrt((z2 - x) ** 2 + y ** 2)]
        return (np.array(px) - np.array(nx)) * k_


x_ = np.linspace(-3, 3.00, 150)
y_ = np.linspace(-3, 3.00, 150)
X, Y = np.meshgrid(x_, y_)

fig, ax = plt.subplots()
chrg = Charges(1.25)

Z = chrg.potential(X.flatten(), Y.flatten())
Z = Z.reshape(X.shape)

CS = ax.contour(X,Y,Z,20, colors='Black')

ax.clabel(CS, CS.levels, inline=True, fontsize=10)
plt.show()