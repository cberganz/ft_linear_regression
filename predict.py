from ft_linear_regression import MyLR
import numpy as np

if __name__ == '__main__':
    LR = MyLR(np.array([[0.0], [0.0]]), alpha=0.001, factor=10000)
    LR.read_thetas('thetas.csv')
    x = int(input('Enter km for the car you wanna estimate: '))
    if x < 0:
        print("Mileage must be a positive integer !")
    else:
        result = LR.predict_(np.array([[x / LR.factor]]))
        if result < 0:
            print("Mileage too high, this model cannot predict the value of this car !")
        else:
            print("The estimated price of this car is {:0.2f}$ according to this model.".format(*result))
