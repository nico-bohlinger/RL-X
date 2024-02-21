#!/bin/bash

WORKDIR="/RL-X_ws/RL-X"

cd $WORKDIR
git fetch && git pull

if [ ! -z "$COMMIT_HASH" ]; then
  git checkout $COMMIT_HASH
fi

if [ ! -z "$DIFF_PATH" ]; then
  git apply $DIFF_PATH
fi

cd $WORKDIR/experiments/docker
bash experiment.sh

exec "$@"
