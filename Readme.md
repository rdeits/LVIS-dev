# LVIS-dev: Experimental development of LVIS (Learning from Value Interval Sampling)

This package includes the experimental implementation and development files associated with the LVIS control design strategy, as presented in "LVIS : Learning from Value Function Intervals for Contact-Aware Robot Controllers" by Robin Deits, Twan Koolen, and Russ Tedrake.

## Installation

* Install Julia v0.6.4 from <https://julialang.org/downloads/oldreleases.html> and add it to your PATH as `julia-0.6`.
* Install the necessary julia packages with:

    ./setup.sh

## Usage

Activate the Julia environment with:

    ./activate.sh

then launch the Jupyter notebook server with:

    julia-0.6 -e "using IJulia; notebook(dir=pwd())"

and check out the `examples` folder.

