[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)


<!--
<p align="center">
    <a href = "https://minigrid.farama.org/" target = "_blank" > <img src="https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png" width="500px"/> </a>
</p>
 -->

<!--
<p align="center">
  <img src="figures/door-key-curriculum.gif" width=200 alt="Figure Door Key Curriculum">
</p>
 -->

Hypergrid is an extension of the [Farama Foundation's](https://farama.org)
[Minigrid](https://minigrid.farama.org) library.
Hypergrid takes the two dimensions of Minigrid and rewrites the library to
operate in arbitrary, discrete dimensions.
The framework, environments, and task derived from Hypergrid are intended
to maintain Minigrid's compatibility with the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API standards, while remaining lightweight, fast, and easily customizable.

Hypergrid additionally borrows inspiration from [Multigrid](https://github.com/ini/multigrid)
by integrating support for multiple agents.

# Installation

To install the Hypergrid for use, run the following:

```bash
pip install git+https://github.com/bhosley/hypergrid.git
```

For an editable installation, for further development or research, run the following:

```bash
git clone https://github.com/bhosley/hypergrid.git
cd hypergrid
pip install --upgrade pip setuptools wheel # needed for pip < 21.3
pip install -e .
```

To install required dependencies for development, run:

```bash
pip install -e ".[dev]"
```

This package requires Python 3.9 or later.

# Environments

<!--
## Minigrid
The list of the environments that were included in the original `Minigrid` library can be found in the [documentation](https://minigrid.farama.org/environments/minigrid/). These environments have in common a triangle-like agent with a discrete action space that has to navigate a 2D map with different obstacles (Walls, Lava, Dynamic obstacles) depending on the environment. The task to be accomplished is described by a `mission` string returned by the observation of the agent. These mission tasks include different goal-oriented and hierarchical missions such as picking up boxes, opening doors with keys or navigating a maze to reach a goal location. Each environment provides one or more configurations registered with Gymnasium. Each environment is also programmatically tunable in terms of size/complexity, which is useful for curriculum learning or to fine-tune difficulty.
 -->

<!--
# Training an Agent
The [rl-starter-files](https://github.com/lcswillems/torch-rl) is a repository with examples on how to train `Minigrid` environments with RL algorithms. This code has been tested and is known to work with this environment. The default hyper-parameters are also known to converge.
-->

# Citation
<!--
The original `gym-minigrid` environments were created as part of work done at [Mila](https://mila.quebec). The Dynamic obstacles environment were added as part of work done at [IAS in TU Darmstadt](https://www.ias.informatik.tu-darmstadt.de/) and the University of Genoa for mobile robot navigation with dynamic obstacles.

https://github.com/Farama-Foundation/Minigrid

To cite this project please use:
 -->

<!-- ```bibtex
@inproceedings{Hypergrid,
  author       = {Brandon Hosley},
  title        = {},
  booktitle    = {},
  month        = {},
  year         = {},
}
``` -->
