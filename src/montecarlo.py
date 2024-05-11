from random import randrange as r

import matplotlib.pyplot as plt


def monte_carlo_pi_graf2(points=2500):
    X, Y = [], []
    X1, Y1 = [], []
    circle_points = 0
    for i in range(points):
        x, y = r(-100, 100, 1) / 100, r(-100, 100, 1) / 100
        if x ** 2 + y ** 2 <= 1:
            circle_points += 1
            X.append(x)
            Y.append(y)
        else:
            X1.append(x)
            Y1.append(y)

    print(f'Generated Pi ({points} points):', 4 * circle_points / points)
    plt.scatter(X, Y, color='red')
    plt.scatter(X1, Y1, color='blue')
    plt.show()


if __name__ == '__main__':
    monte_carlo_pi_graf2()
