import numpy as np
import math
import matplotlib.pyplot as plt

from src.unconstrained_min import line_search
from src.utils import plotContour
from tests.examples import funcRosenbrock


def main():
    x =line_search(funcRosenbrock,np.array([2,2]),10**-12, 10**-12,100, "nt")
    plotContour(x[0],x[2])
    print(x[0])


if __name__ == '__main__':
    main()