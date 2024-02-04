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
    l2_error = df['L2 error'].values

    plt.loglog(meshwidth, l2_error, 'o-', markersize=5, label='L2 error')
    # Add slope triangle
    min_error = np.min(l2_error)
    annotation.slope_marker((meshwidth[0], min_error), 1.0, invert=False)
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
    l2_error = df['L2 error'].values

    # Creating log-log plot
    plt.figure(figsize=(8, 6))
    plt.loglog(meshwidth, l2_error, 'o-', color='red', label='L2 error')

    # Computing and displaying the slope
    slopes = np.diff(np.log(l2_error)) / np.diff(np.log(meshwidth))
    average_slope = np.mean(slopes)
    plt.title(f'Log-Log Convergence Plot\nAverage Slope: {average_slope:.2f}')

    # Labeling the axes
    plt.xlabel('Meshwidth (log scale)')
    plt.ylabel('L2 Error (log scale)')
    # Swap the x direction to get decreasing error
    plt.gca().invert_xaxis()

    # Setting formatter for axes
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Saving the plot
    plt.savefig(output_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    #plot(str(argv[1]), str(argv[2]))
    plot_convergence(str(argv[1]))