# library imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# project imports
from glpe import GLPE


class Example:
    """
    An example the usage of the algorithm with sample data
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        """
        Single entry method
        """
        # load data
        data = make_swiss_roll(n_samples=1000, noise=0.01)
        # print 3D view
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data.iloc[0, :],
                   data.iloc[1, :],
                   data.iloc[2, :],
                   marker='o',
                   color="black")
        feature_names = list(data)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        plt.savefig("before_reduction_3d.png", dpi=500)
        plt.close()
        # reduce data
        reduced_data = GLPE.run(data=data,
                                k=7,
                                alpha=0.5,
                                target_dim=2)
        # print 2D view after reduction
        plt.scatter(reduced_data.iloc[0, :],
                    reduced_data.iloc[1, :],
                    marker='o',
                    color="black")
        plt.savefig("after_reduction_2d.png", dpi=500)
        plt.close()


if __name__ == '__main__':
    Example.run()
