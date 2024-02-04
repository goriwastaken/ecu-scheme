#!/bin/bash

# Script to run experiments of the thesis
help(){
  echo -e "Usage: main.sh [OPTIONS] [ARGUMENTS]\n\
  \tWithout options the method runs all the experiments described in
  \tthe experiment section of the thesis.\n\n\
  \t-s | --stationary\t\tRuns only the stationary experiments with lower refinement levels\n\
  \t-i | --instationary\t\tRuns the instationary experiments with lower refinement levels.\n\
  \t-a | --all\t\t\tRuns all the experiments.\n\
  \t-c | --custom\t\t\tRuns the experiments with the refinement levels and epsilon provided.\n\
  \t-h | --help\t\t\tDisplays the help message.\n\
  \t-p | --plot\t\t\tPlots the results of the experiments.\n\
  "
}
# optional arguments: refinement_levels and epsilon to pass to the experiments
ARGUMENTS=()
OPTIONS=[-s | --stationary] [-i | --instationary] [-a | --all] [-c | --custom] [-h | --help] [-p | --plot]

set -e # Suppress errors

# Base directory for experiments
#BASE_DIR="../../build/projects/ecu_scheme"
BASE_DIR="../../cmake-build-release/projects/ecu_scheme"
LIB_DIR="/home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp"

# Function to run Stationary Case Experiments
run_stationary_experiments(){
  echo "Running Stationary Experiments"
  # Add commands for Stationary Experiments

  local refinement_levels="$1"
  local epsilon="$2"

  echo -e "\nExecute manufactered solution experiment (section )\n"
  cd $BASE_DIR/experiments/manufactured_sol || exit
  ./projects.ecu_scheme.experiments.manufactured_solution "$refinement_levels" "$epsilon"
  cd - > /dev/null || return

#  echo -e "\nExecute basic stationary experiment (section )\n"
#  cd $BASE_DIR/experiments/testexp || exit
#  ./projects.ecu_scheme.experiments.testexp.exp1
#  cd - > /dev/null || return

}

# Function to run Instationary Case Experiments
run_instationary_experiments(){
  echo "Running Instationary Experiments"
  # Add commands for Instationary Experiments

  local refinement_levels="$1"
  local epsilon="$2"

  echo -e "\nExecute constant velocity experiment (section )\n"
  cd $BASE_DIR/experiments/const_velo || exit
  ./projects.ecu_scheme.experiments.const_velo "$refinement_levels" "$epsilon"
  cd - > /dev/null || return
}

# Function to run all experiments
run_all_experiments() {
    echo "Running all experiments"
    run_stationary_experiments "$1" "$2"
    run_instationary_experiments "$1" 0.0
    # Add calls to other experiment functions
}

# Display usage information
usage() {
    echo "Usage: main.sh [OPTIONS]"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Process the arguments
while [ "$1" != "" ]; do
    case $1 in
        -a | --all )
            run_all_experiments 5 1e-8
            ;;
        -s | --stationary )
            run_stationary_experiments 2 1e-8
            run_stationary_experiments 3 1.0
            run_stationary_experiments 5 1e-8
            run_stationary_experiments 6 1e-8
            run_stationary_experiments 6 1.0
            ;;
        -i | --instationary )
            run_instationary_experiments 3 0.0
            ;;
        -c | --custom )
            run_all_experiments "$2" "$3"
            ;;
        -h | --help )
            help
            ;;
        -p | --plot )
            echo "Plotting...todo"
            # Add plot commands
            cd $LIB_DIR/projects/ecu_scheme/post_processing || exit
            python3 plot_convergence.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_2_1e-08_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_2_1e-08_plot.eps
            python3 plot_convergence.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_3_1_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_3_1_plot.eps
            python3 plot_convergence.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_5_1e-08_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_5_1e-08_plot.eps
            # plot linear case
            python3 plot_convergence.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_linear_2_1e-08_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_linear_2_1e-08_plot.eps
            python3 plot_convergence.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_linear_3_1_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_linear_3_1_plot.eps
            python3 plot_convergence.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_linear_5_1e-08_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_conv_linear_5_1e-08_plot.eps
            ;;
        * )
            usage
            ;;
    esac
    shift
done
