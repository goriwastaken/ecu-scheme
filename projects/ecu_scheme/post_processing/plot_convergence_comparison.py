import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from mpltools import annotation

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
        annotation.slope_marker((meshwidth[1], min_error), 1.0, invert=False)
        annotation.slope_marker((meshwidth[3], min_error), 2.0, invert=False)

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
    #l2_error = df['L2 error'].values

    error_names = [col for col in df.columns if col not in ['No. of dofs', 'Meshwidth']]

    for err_name in error_names:
        err_val = df[err_name].values
        plt.loglog(meshwidth, err_val, 'o-', markersize=5, label=err_name)
        # Add slope triangle
        min_error = np.min(err_val)
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