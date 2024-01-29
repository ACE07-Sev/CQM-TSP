# CQM-TSP
CQM solvers for TSP and ESP, using D-Wave's Quantum Annealing hardware.

## Getting Started
<p align="justify"> The CQM solvers take a list of coordinates and output the globally optimal path in $O(n)$ time (due to running on D-Wave HybridCQMSampler QPU), as compared to $O(n^2 * 2^n)$ complexity of its dynamic programming counterpart which is the Held-Karp modelling. The package provides an end-to-end interface, where given a list of coordinates and edges you can run the `QTSP` and `QESP` solvers. </p>

```
# Define the graph
coordinates = [[1, 1], [2, 3], [3, 2], [2, 4], [1, 5], [3, 6]]
edges = [[0, 1], [1, 2], [1, 3], [1, 5], [2, 3], [3, 4], [4, 5]]

# Define the solver
esp = QESP(coordinates=coordinates, edges=edges, source=1, destination=4, time=30, log=False)

# Run the solver
esp(token=token)
```
```
# Define the graph
coordinates = [[1, 1], [2, 3], [3, 2], [2, 4], [1, 5], [3, 6], [5, 7], [4, 1], [9, 11], [10, 10]]

# Define the solver
tsp_model = QTSP_Improved(coordinates=coordinates, time=50, log=False)

# Run the solver
tsp_model(token=token)
```

### Prerequisites
- Python 3.12+

### Installation
```
pip install PACKAGE
```
The default installation of PACKAGE includes `numpy`, `math`, `matplotlib`, `dimod`, and `dwave.system`.

## Usage
<p align="justify"> The notebooks are a good way for understanding how the package works. Depending on your preference, you may use the framework completely end-to-end, or use it in parts for low-level customization. </p>

<p align="justify"> To use the model, simply replace the coordinate list with your own coordinates and run the model (make sure you have D-wave setup on your IDE, if you do not have D-wave 
token, please create a LEAP account, and setup by running "dwave config create", and inputting your token. Then type in "dwave setup" to finish the setup. For further
information visit D-wave's documentation on CLI: </p>
https://docs.ocean.dwavesys.com/en/stable/docs_cli.html

## License
Distributed under the GPL 3.0 license. See LICENSE for more details.

## Issues
If you face any errors/issues kindly create an issue and your issue will be taken care of at the earliest opportunity. 

Thank you for using the CQM-TSP and wish you a blessed day!

<p>© 2024 Amirali Malekani Nezhad, all rights reserved.</p>
