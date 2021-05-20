#генеция данных, датасет и даталоадер (его доделать надо)

from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging

# logging.basicConfig(level = logging.INFO,
#                     format= '%(asctime)s %(processName) - 10s %(name)s - %(levelname)s: %(message)s')


class Circles(Dataset):
    def __init__(self, n_samples, shuffle, noise, random_state = 0, factor = .8):

        self.X, self.y = datasets.make_circles(n_samples = n_samples, shuffle = shuffle, noise = noise,
        random_state = random_state, factor = factor)


        #preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)

        self.X, self.y = self.X.astype(np.float32), self.y.astype(int)

        # to do feature generation
        # fixme
    def __len__(self):
        return  len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self):
        plt.figure(figsize = (8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], c = self.y)
        plt.show()
        plt.savefig("1.png")

if __name__ == "__main__":
    #пишем логи в файл
    logger = logging.getLogger("dataset")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(processName) - 10s %(name)s - %(levelname)s: %(message)s'))

    file_handler = logging. FileHandler('logs.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(processName) - 10s %(name)s - %(levelname)s: %(message)s'))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(file_handler)


    circles = Circles(n_samples= 5000, shuffle= True, noise= 0.1, random_state= 0, factor= 0.5)
    print(circles.X)
    print(circles.y)
    print(len(circles))
    logger.info(f'element 0 {circles[0]}')
    logger.debug(f'element 10 {circles[10]}')
    logger.info(f'len(circles): {len(circles)}')
    circles.plot_data()







