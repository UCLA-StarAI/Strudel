# Strudel -- Learning Structured-Decomposable Circuits

This repo contains the code and experiments from the paper "[Strudel: Learning Structured-Decomposable Probabilistic Circuits](http://starai.cs.ucla.edu/papers/DangPGM20.pdf)", published in PGM 2020, and it's extended version "[Strudel: A Fast and Accurate Learner of Structured-Decomposable Probabilistic Circuits](http://starai.cs.ucla.edu/papers/DangIJAR22.pdf)", published in IJAR 2022.

To cite this paper, please use
```
@inproceedings{DangPGM20,
  author    = {Dang, Meihua and Vergari, Antonio and Van den Broeck, Guy},
  title     = {Strudel: Learning Structured-Decomposable Probabilistic Circuits},
  booktitle = {Proceedings of the 10th International Conference on Probabilistic Graphical Models (PGM)},
  month     = {sep},
  year      = {2020}}
```
or the extended version
```
@article{DangIJAR22,
    author = {Dang, Meihua and Vergari, Antonio and Van den Broeck, Guy},
    title = {Strudel: A Fast and Accurate Learner of Structured-Decomposable Probabilistic Circuits},
    journal = {International Journal of Approximate Reasoning},
    month = {Jan},
    year = {2022},
    volume = {140},
    pages = {92-115},
    issn = {0888-613X},
    doi = {https://doi.org/10.1016/j.ijar.2021.09.012}}
```

## Files
```
bin/            Runnable julia scripts (see experiments below)
scripts/        Helper files to generate experiments scripts.
src/            The source code for the algorithm.
Project.toml    This file specifies required julia environment.
README.md       This is this file.
```

## Installation

1. Julia version 1.7
2. The following command will download and install all required packages.

```
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile();'
```
    
## Experiments
Please navigate to `bin/README.md` for details.