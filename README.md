# CCM Memory
## Description
This package defines abstract memory objects and various implementations of cognitively plausible memory models. Currently, the following memory models are implemented:
- MINERVA2 (Hintzman, 1984) 
- Modern Hopfield Network (Ramsauer et al., 2021)
- Stateful Attention Memory (SAM; forthcoming)

These models are intended to assist in research settings (e.g. comparison of model behaviours) and/or be used as components in larger models.

## Installation
To install, first clone (or download & unpack) this repo to a folder on your computer. Navigate to that folder using a terminal, then type the command:

    pip install .

This package requires [``numpy``](https://pypi.org/project/numpy/) and [``hrrlib``](https://github.com/ren-oz/hrrlib). These should be installed automatically using the above command. If this does not work for you, please [submit an issue](https://github.com/ren-oz/ccm_memory/issues).

### References
- Hintzman, D. L. (1984). MINERVA 2: A simulation model of human memory. *Behavior Research Methods, Instruments, & Computers, 16*, 96–101. https://doi.org/10.3758/BF03202365
- Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L., Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2021). Hopfield networks is all you need. *arXiv*, 1-10. https://doi.org/10.48550/arXiv.2008.02217 