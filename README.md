# CQM-TSP
CQM model for Traveling Salesman Problem using Quantum Annealing

# Model Usage Instructions
The CQM model takes an np.array list of coordinates and outputs the globally optimal path in $O(n)$ time (due to running on D-Wave HybridCQMSampler QPU), as compared to $O(n^2 * 2^n)$ complexity of its dynamic programming counterpart which is the Held-Karp modelling. 

The model takes the list of coordinates, and generates a distance matrix based on Euclidean distance. The model then uses this generated distance matrix in its objective
to find the cost for each taken path. 

The model has been implemented with three constraints :
1) One entry per node 
2) One exit per node
3) No subtours

Current model runs on 5 second sampling intervals, and can scale up well with higher number of nodes (as long as it does not exceed 10,000 variables). Current model uses 
62 constraints, and 30 variables for 6 nodes. 

To use the model, simply replace the coordinate list with your own coordinates and run the model (make sure you have D-wave setup on your IDE, if you do not have D-wave 
token, please create a LEAP account, and setup by running "dwave config create", and inputting your token. Then type in "dwave setup" to finish the setup. For further
information visit D-wave's documentation on CLI:
https://docs.ocean.dwavesys.com/en/stable/docs_cli.html

# Issues
If you face any errors/issues kindly create an issue and your issue will be taken care of at the earliest opportunity. 

Thank you for using the CQM-TSP and wish you a blessed day!