import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from mpltools import annotation
import itertools

from matplotlib.ticker import ScalarFormatter


def plot_convergence(file):
    df = pd.read_csv(file, skiprows=2)
    dofs = df['No. of dofs'].values
    meshwidth = df['Meshwidth'].values
    #l2_error = df['L2 error'].values

    error_names = [col for col in df.columns if col not in ['No. of dofs', 'Meshwidth']]

    for err_name in error_names:
        err_val = df[err_name].values
        plt.loglog(meshwidth, err_val, 'o-', markersize=5, label=err_name)
        # Add slope triangle
        min_error = np.min(err_val)
        # annotation.slope_marker((meshwidth[1], min_error), 1.0, invert=False)
        # annotation.slope_marker((meshwidth[3], min_error), 2.0, invert=False)
    #     slopes = np.diff(np.log(l2_error)) / np.diff(np.log(meshwidth))
    # average_slope = np.mean(slopes)
        slope = (np.log(err_val[-1]) - np.log(err_val[-2])) / (np.log(meshwidth[-1]) - np.log(meshwidth[-2]))
        plt.plot(meshwidth, slope * np.log(meshwidth) + np.log(err_val[0]), color='yellow', linestyle='--', label='Slope')

    # Add labels and legend

    # Label plot
    plt.legend()
    plt.xlabel('h')
    plt.ylabel('L2 error')
    plt.grid()
    plt.show()


# Convergence plot for experiments


def plot(file, output_name):
    df = pd.read_csv(file, skiprows=2)
    dofs = df['No. of dofs'].values
    meshwidth = df['Meshwidth'].values
    l2_errors = df.iloc[:, 2:]

    error_names = [col for col in df.columns if col not in ['No. of dofs', 'Meshwidth']]
    markers = itertools.cycle(('o', '.', 'v', '^', 's', 'p', '*', 'h', 'd'))
    for i, err_name in enumerate(error_names):
        err_val = l2_errors[err_name].values
        plt.loglog(meshwidth, err_val, linestyle='-', marker=next(markers), markersize=5, label=err_name)

    # Add slope triangle
    min_error = np.min(l2_errors)
    annotation.slope_marker((meshwidth[1], min_error), 1.0, invert=False)
    annotation.slope_marker((meshwidth[3], min_error), 2.0, invert=False)

    # Label plot
    plt.legend()
    plt.xlabel('h')
    plt.ylabel('L2 error')
    plt.grid()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot(str(argv[1]), str(argv[2]))
    #plot_convergence(str(argv[1]))