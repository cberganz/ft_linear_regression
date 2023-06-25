import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyLR():
    """
    Implementation of a linear regression class.
    """
    def __init__(self, thetas, alpha=0.001, factor=1):
        self.alpha = alpha
        self.thetas = thetas
        self.factor = factor

    def add_intercept(self, x):
        return np.c_[np.ones(len(x)), x]

    def gradient(self, x, y, theta):
        x_prime = self.add_intercept(x)
        x_T = x_prime.transpose()
        diff = np.matmul(x_prime, theta) - y
        return np.matmul(x_T, diff) / len(x)

    def fit_(self, x, y):
        loss_diff = 1
        save_loss = 1
        i = 0
        x = x / self.factor
        y = y / self.factor
        print('Processing linear ajustments...')
        while loss_diff > 0.1 or loss_diff < -0.1:
            self.thetas = self.thetas - self.alpha * self.gradient(x, y, self.thetas)
            loss = self.loss_(y * self.factor, self.predict_(x))
            loss_diff = loss - save_loss
            save_loss = loss
            i += 1
        print("Linear ajustement done with", i, "iterations. Thetas are :\nt0:", self.thetas[0], "\nt1:", self.thetas[1])
        self.plot(x * self.factor, y * self.factor, self.predict_(x))
        plt.show()

    def predict_(self, x):
        return np.c_[np.ones(len(x)), x].dot(self.thetas).dot(self.factor)

    def loss_elem_(self, y, y_hat):
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        return ((y - y_hat) ** 2).mean()

    def plot(self, x, y, y_hat):
        plt.scatter(x, y, color="green")
        plt.scatter(x, y_hat, color="blue")
        plt.plot(x, y_hat, color="red")

    def save_thetas(self, path):
        f = open(path, "w")
        f.write('theta0,theta1\n')
        f.write(str(self.thetas[0][0]) + ',' + str(self.thetas[1][0]))
        f.close()
        print("Thetas successfully saved to 'thetas.csv'.")

    def read_thetas(self, path):
        try:
            df = pd.read_csv(path)
            self.thetas = df.values[0]
            print("Thetas successfully read from 'thetas.csv'. Thetas are :\nt0:", self.thetas[0], "\nt1:", self.thetas[1])
        except:
            self.thetas = np.array([0.0, 0.0])
            print("File thetas.csv does not exist. Thetas have been set to 0.")

    def get_data_from_csv(self, file):
        df = pd.read_csv('data.csv')
        x = np.array([[x[0]] for x in df.values], dtype=np.float128)
        y = np.array([[y[1]] for y in df.values], dtype=np.float128)
        print("Data successfully read from 'data.csv'.")
        return x, y
