#!/bin/bash

#SBATCH --job-name=rlx_experiment
#SBATCH --output=log/out_and_err.txt
#SBATCH --error=log/out_and_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

commit_hash=""
diff_file_path=""

while getopts c:d: flag
do
    case "${flag}" in
        c) commit_hash=${OPTARG};;
        d) diff_file_path=${OPTARG};;
        \?) echo "Invalid option -$OPTARG" >&2
            exit 1
            ;;
    esac
done

bash run.sh -c $commit_hash -d $diff_file_path -h false
