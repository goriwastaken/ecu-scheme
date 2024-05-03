#!/bin/bash

# Script to run experiments of the thesis
help(){
  echo -e "Usage: main.sh [OPTIONS] [ARGUMENTS]\n\
  \t-s | --scalar\t\tRuns only the scalar(0-forms) experiments with lower refinement levels.\n\
  \t-o | --oneform\t\tRuns the experiments for 1-forms with lower refinement levels.\n\
  \t-a | --all\t\t\tRuns all the experiments with higher refinement levels.\n\
  \t-c | --custom\t\t\tRuns the experiments with the refinement levels and epsilon provided.\n\
  \t-h | --help\t\t\tDisplays the help message.\n\
  \t-p | --plot\t\t\tPlots the results of the experiments.\n\
  "
}
# optional arguments: refinement_levels and epsilon to pass to the experiments
ARGUMENTS=()
#shellcheck disable=SC2215
OPTIONS=("-s | --scalar" "-o | --oneform" "-a | --all" "-c | --custom" "-h | --help" "-p | --plot")

set -e # Suppress errors

# Base directory for experiments
#BASE_DIR="../../build/projects/ecu_scheme"
BASE_DIR="../../cmake-build-release/projects/ecu_scheme"
LIB_DIR="/home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp"

# Function to run Stationary Case Experiments
run_scalar_experiments(){
  echo "Running Stationary Experiments"
  # Add commands for Stationary Experiments

  local refinement_levels="$1"
  local epsilon="$2"

  # Check if the binaries exist
    if [ ! -x "$BASE_DIR/experiments/manufactured_sol/projects.ecu_scheme.experiments.manufactured_solution" ]; then
      echo "Building manufactured solution binary..."
      cd "$BASE_DIR/experiments/manufactured_sol" || return 1
      cmake .
      make
      cd - > /dev/null || return 1
    fi

    if [ ! -x "$BASE_DIR/experiments/concentric_stream/projects.ecu_scheme.experiments.concentric_stream" ]; then
      echo "Building concentric stream binary..."
      cd "$BASE_DIR/experiments/concentric_stream" || return 1
      cmake .
      make
      cd - > /dev/null || return 1
    fi

    if [ ! -x "$BASE_DIR/experiments/const_velo/projects.ecu_scheme.experiments.const_velo" ]; then
      echo "Building constant velocity binary..."
      cd "$BASE_DIR/experiments/const_velo" || return 1
      cmake .
      make
      cd - > /dev/null || return 1
    fi

  echo -e "\nExecute manufactered solution experiment (section 5.1.1)\n"
  cd $BASE_DIR/experiments/manufactured_sol || exit
  ./projects.ecu_scheme.experiments.manufactured_solution "$refinement_levels" "$epsilon"
  cd - > /dev/null || return

  echo -e "\nExectue concentric stream experiment (section 5.1.2)\n"
  cd $BASE_DIR/experiments/concentric_stream || exit
  ./projects.ecu_scheme.experiments.concentric_stream "$refinement_levels" 0.0
  cd - > /dev/null || return

  echo -e "\nExecute constant velocity experiment (section 5.1.3)\n"
  cd $BASE_DIR/experiments/const_velo || exit
  ./projects.ecu_scheme.experiments.const_velo "$refinement_levels" 0.0
  cd - > /dev/null || return
#  echo -e "\nExecute basic stationary experiment (section 5)\n"
#  cd $BASE_DIR/experiments/testexp || exit
#  ./projects.ecu_scheme.experiments.testexp.exp1
#  cd - > /dev/null || return

}

# Function to run Instationary Case Experiments
run_oneform_experiments(){
  echo "Running Instationary Experiments"
  # Add commands for Instationary Experiments

  local refinement_levels="$1"
  local epsilon="$2"

  echo -e "\nExecute constant velocity experiment (section 5.2)\n"
  cd $BASE_DIR/experiments/advection_one_form_experiments/rotating_hump || exit
  ./projects.ecu_scheme.experiments.rotating_hump "$refinement_levels" "$epsilon"
  cd - > /dev/null || return
}

# Function to run all experiments
run_all_experiments() {
    echo "Running all experiments"
    run_scalar_experiments "$1" "$2"
    run_oneform_experiments 4 0.0
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
            run_all_experiments 8 1e-8
            ;;
        -s | --scalar )
            run_scalar_experiments 7 1.0
            run_scalar_experiments 6 1e-8
            ;;
        -o | --oneform )
            run_oneform_experiments 3 0.0
            ;;
        -c | --custom )
            run_all_experiments "$2" "$3"
            ;;
        -h | --help )
            help
            exit 0
            ;;
        -p | --plot )
            echo "Plotting...todo"
            # Add plot commands
            cd $LIB_DIR/projects/ecu_scheme/post_processing || exit
            python3 plot_convergence_comparison.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_quad_comparison_7_1_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_quad_comparison_7_1_plot.eps
            python3 plot_convergence_comparison.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_quad_comparison_6_1e-08_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/manufactured_solution_quad_comparison_6_1e-8_plot.eps
            python3 plot_convergence_comparison.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/concentric_stream_quad_comparison_7_0_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/concentric_stream_quad_comparison_7_0_plot.eps
            python3 plot_convergence_comparison.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/const_velo_quad_comparison_7_0_L2error.csv /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/const_velo_quad_comparison_7_0_plot.eps
            # Plot command for 1-forms experiment
            python3 ./plot_convergence_comparison.py /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/rot_hump_linear_3_0_L2error.csv  /home/gori/Documents/Thesis/thesis-lehrfempp-repo/lehrfempp/cmake-build-release/results/rot_hump_linear_3_0_plot.eps
            ;;
        * )
            usage
            ;;
    esac
    shift
done
