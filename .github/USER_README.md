# .github


## Important files

**copilot-instructions.md**
- Automatically picked up by GitHub Copilot Chat as [custom repository instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot)
- Injected into every Copilot conversation opened in this repository without any manual action
- Documents coding style conventions, codebase-specific constraints, and pointers to experiment documentation


## Optional files (gitignored)

The following files can be created to improve AI-assisted experiment workflows.
They are listed in ```.gitignore``` because they may contain confidential information (e.g. cluster hostnames, credentials) or are too project-specific to be shared.
Copilot is instructed to look for these files and use their contents when helping with related tasks.

**conda-env-name.txt**
- Name of the local conda environment that has all dependencies installed

**cluster-experiments.md**
- Instructions for submitting and monitoring jobs on a compute cluster

**evaluate-experiments.md**
- Instructions for downloading wandb run data and generating plots and analysis
