# CCM Memory
## Description
This package defines abstract memory objects and various implementations of cognitively plausible memory models. Currently, the following memory models are implemented:
- MINERVA2 (Hintzman, 1989) 
- Modern Hopfield Network (Ramsauer et al., 2021)
- Stateful Attention Memory (SAM; forthcoming)

These models are intended to assist in research settings (e.g. comparison of model behaviours) and/or be used as components in larger models.

## Installation
To install, first clone (or download & unpack) this repo to a folder on your computer. Navigate to that folder using a terminal, then type the command:

    pip install .

This package requires [``numpy``](https://pypi.org/project/numpy/) and [``hrrlib``](https://github.com/ren-oz/hrrlib). These should be installed automatically using the above command. If this does not work for you, please [submit an issue](https://github.com/ren-oz/ccm_memory/issues).
