import numpy as np
from numpy.random import normal
import pandas as pd


def generate_data(n_points, dir):
    data = [t + normal(0, 1) for t in range(n_points)]
    data = np.array(data)
    df = pd.DataFrame({"y": data})
    df.to_csv(dir, index=False)


if __name__ == "__main__":
    generate_data(1000, "../data/data.csv")
