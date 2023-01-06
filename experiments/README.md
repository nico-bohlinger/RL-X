# Experiments


## Important files

**experiment.py**
- Imports the runner, all algorithms and environments
- Sets the runner mode and runs the experiment

**experiment.sh**
- Works as a template for how to setup an experiment with the command line
- Can be used to run or even schedule multiple experiments with different configs


## Examples

**Use all default configs**
```
python experiment.py
```

**Overwrite configs by setting the command line arguments**
```
python experiment.py --config.algorithm.gamma=0.9
```

**Structure and run experiments with a bash script**
```
bash experiment.sh
```