# AlgMol1
sandbox repo for the Python code from the paper "Automated Continuous Flow Synthesis and Purification of Algorithmically Designed Ferroelectric Nematogens"

Python functions for:

*	Generating molecules via SELFIES and group-SELFIES. 
*	Interfacing with external software for QM calculations (Gaussian, MOPAC)
*	Genetic algorithms for new molecule generation from an initial starting structure.
*	Functions for filtering algorithm output by score, by sub-structure, or specific features (e.g. ring size, no. rotatable bonds etc.)

A Jupyter notebook is provided and contains worked examples for:

*	Generating new variants of the nematic liquid crystal 5CB through random mutations
*	Genetic algorithm for generating 5CB variants; functions for scoring, searching, filtering

*	Fragmentation of a ferroelectric nematic LC (RM734) through group-Selfies;
*	Generation of RM734-variants through fragment-based mutation
*	Genetic algorithm for generation of RM734-variants via fragment-based mutation

We actually went and synthesised some of these in the lab which is kind of cool, albeit much slower than generating them in the notebook. If you find yourself deliberately making any of the compounds the algorithms generate I would love you to get in touch.

# Prerequisite Packages
rdkit (tested on Q3 2022 release)
numpy
selfies
matplotlib

# Usage
Run all cells and behold!

# Contact
Questions/comments/feedback welcome - r<dot>mandle<at>Leeds<dot>ac<dot>uk
