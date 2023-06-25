from ft_linear_regression import MyLR
import numpy as np

if __name__ == '__main__':
    LR = MyLR(np.array([[0.0], [0.0]]), alpha=0.001, factor=10000)
    X, Y = LR.get_data_from_csv('data.csv')
    LR.fit_(X, Y)
    LR.save_thetas("thetas.csv")
    y_hat = LR.predict_(X)
