# Quantum Algorithms for Geologic Fracture Networks
### Last Updated: 4 September 2022
### Corresponding Author: Jessie M. Henderson, jessieh@lanl.gov

## Authors and Description

This repository contains code and data described in the paper, Quantum Algorithms for Flow in Geologic Fracture Networks.  The code was authored and the data was collected by Jessie M. Henderson and Mariana Podzorova.  The code was developed and results analyzed under the guidance of Marco Cerezo.

## Dependencies and Operating System
Use of this codebase requires at least Python 3.0 and use of the following modules:
- Argparse
- Math
- Matplotlib
- Numpy
- Pathlib
- Pylab
- Qiskit
- Scipy
- Seaborn

For each package except Qiskit, any version compatible with Python 3.0 is likely acceptable.  This code was originally developed with a now-outdated version (0.26) of Qiskit, and there have been some changes in recent versions that may cause minor, easily-addressed syntax errors. To the developers' knowledge, those issues have been resolved in the most recent version of the code, but should any have been missed, please don't hestitate to contact the corresponding author (JessieH@lanl.gov).

The code was developed and tested using Windows 10 and has not been tested using other operating systems.

## Installation Instructions
1. Install Python 3.0.
2. Use the 'pip' installer to install each of the above packages and any dependencies requested. (For example, Matplotlib may require installing Pylatexenc, if that is not already available on your machine.
3. Download the scripts in this repository, which can be run straightforwardly using the standard 'python' command.

The only time-consuming part of the installation can be downloading the dependencies, which should take only about twenty minutes with a reasonable internet connection.

## Documentation
Our goal was to include enough documentation within the code such that it's usable without additional documentation.  Should you have any questions or seek further documentation, please contact the corresponding author.

## Brief Description of Contents
1. Uniform Permeability with 6x8 Regions: This code and data is described in both Section III of the main text, and Section V.B.3 of the Methods.
2. Uniform Permeability with Larger Regions: This code and data is described in both Section III of the main text, and Section V.B.4 of the Methods.
3. Preliminary Results with Varying Permeability: This code and data is described in Section VI.C of the Supplementary Information.

## License

quantum-geologic-fracture-networks is provided under a BSD-ish license with a "modifications must be indicated" clause. See LICENSE.md file for the full text.

This package is part of the Hybrid Quantum-Classical Computing suite, known internally as LA-CC-16-032.
