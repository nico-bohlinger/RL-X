#!/bin/bash

commit_hash=""
diff_file_path=""
headless=true

while getopts "c:d:h:" opt; do
  case $opt in
    c) commit_hash="$OPTARG" ;;
    d) diff_file_path="$OPTARG" ;;
    h) headless="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
  esac
done

docker_run_command="docker run"

if [ ! -z "$commit_hash" ]; then
  docker_run_command+=" -e COMMIT_HASH='$commit_hash'"
fi

if [ ! -z "$diff_file_path" ]; then
  absolute_diff_file_path="$(realpath "$diff_file_path")"
  diff_file_dir=$(dirname "$absolute_diff_file_path")
  diff_file_name=$(basename "$absolute_diff_file_path")

  docker_run_command+=" -e DIFF_PATH='/RL-X_ws/diffs/$diff_file_name' -v '$diff_file_dir:/RL-X_ws/diffs'"
fi

if [ "$headless" = true ]; then
  docker_run_command+=" -d"
fi

docker_run_command+=" --env-file env.config"
docker_run_command+=" --gpus all"
docker_run_command+=" --rm"
docker_run_command+=" rlx_i"

eval $docker_run_command
