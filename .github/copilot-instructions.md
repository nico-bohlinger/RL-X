# Coding style
1. No unnecessary helper functions. If a function is only used once, it should be inlined.
2. There should be two lines between methods in a class.
3. Sometimes, it is useful for lines to be longer than 79 characters, especially when it just many wrapped calls, like `np.mean(np.abs(np.array([...)))`. In such cases, it is fine to have lines that are longer than 79 characters.
4. There should be no such things as private methods or fields, i.e. no leading underscores.
5. No comments at the start of a file. And no excessive comments in general. If the code is not self-explanatory, minimal comments can be added but should be inline with the rest of the codebase.

# Code base
## Environments
### Custom MuJoCo environments
1. The custom mujoco environments can have MuJoCo, MJX and MJX+Warp versions, and all algorithms with Flax (for MuJoCo) and Flax fully jitted (for MJX and MJX+Warp) should work for these versions. Code always needs to be written in a way such that it works for all environments and algorithms. E.g. This means np.where should be used instead of if statements when the code block needs to be jitted.

# Experiments
## Local development
1. Experiments are started from the experiments folder. Usually you find some dummy .sh scripts there that can be used as a template for new experiments. You can look into the .github/conda-env-name.txt to find the name of the conda environment that has all the dependencies installed.

## Cluster experiments
1. Look into the .github/cluster-experiments.md file for instructions on how to run experiments on the cluster. If the file does not exist, there are not instructions yet.

## Evaluating experiments
1. Look into the .github/evaluate-experiments.md file for instructions on how to evaluate experiments. If the file does not exist, there are not instructions yet.