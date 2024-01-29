# Copyright 2023-2024 Amir Ali Malekani Nezhad.
#
# Licensed under the License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/ACE07-Sev/CQM-TSP/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ['QTSP_Improved']

import re
import os
os.chdir('..')

import dimod
from dimod.serialization.format import Formatter
from dimod import ConstrainedQuadraticModel, Binary, quicksum, Real

from cqm.data import Graph
from cqm.solver import CQM
from cqm.backend import CQMBackend

import numpy as np


class QTSP_Improved(CQM):
    """ `QTSP_Improved` is a class for implementing a CQM for the Travelling Salesman Problem using
        Gavish-Graves subtour elimination constraint.
    """
    def __init__(self,
                 coordinates: list[list[float]],
                 time: int,
                 log: bool = True) -> None:
        """ Initializes a `QTSP` model.

        Parameters
        ----------
        `coordinates` (list[list[float]]):
            The coordinates of the nodes of the graph.
        `time` (int):
            The time limit for the problem.
        `log` (bool):
            Whether to log the output or not.
        """
        self.graph = Graph(coordinates=coordinates)
        self.time_limit = time
        self.log = log
        # Construct the CQM
        self.cqm = self.define_CQM()

    def define_CQM(self) -> dimod.ConstrainedQuadraticModel:
        """ Function to define the CQM for the given graph.

        Returns
        -------
        `cqm` (dimod.ConstrainedQuadraticModel):
            The CQM.
        """
        # Define the number of nodes
        n = self.graph.num_nodes

        # Construct the distance matrix
        distance_matrix = self.graph.calculate_distance_matrix()

        # Initialize the CQM
        cqm = ConstrainedQuadraticModel()

        # Initialize the objective
        objective = 0

        # Initialize the constraints
        constraint_1 = 0
        constraint_2 = 0
        constraint_3 = 0

        # Initialize the decision var
        X = np.array([[Binary(f"X_{i+1}_{j+1}") if i!=j else 0 for j in range(n)] for i in range(n)])
        Z = np.array([[Real(f"Z_{i+1}_{j+1}") if i!=j else 0 for j in range(n)] for i in range(n)])

        # Define objective
        objective = quicksum(distance_matrix[i][j] * X[i][j] for i in range(n) for j in range(n))
        cqm.set_objective(objective)

        # Define Constraints (Only one entry and exit per node)
        for i in range(n):
            constraint_1 = quicksum(X[i][j] for j in range(n))
            cqm.add_constraint(constraint_1 == 1, label=f"Constraint 1-{i+1}")

        for j in range(n):
            constraint_2 = quicksum(X[i][j] for i in range(n))
            cqm.add_constraint(constraint_2 == 1, label=f"Constraint 2-{j+1}")

        # Define subtour elimination constraint
        for i in range(n):
            if i >= 1:
                constraint_3 = quicksum(Z[i][j] for j in range(n)) - quicksum(Z[j][i] for j in range(n) if j != 0)
                cqm.add_constraint(constraint_3 == 1, label=f"Constraint 3-{i+1}")

        for i in range(n):
            for j in range(n):
                if i != 0:
                    if X[i][j] == 0:
                        pass
                    else:
                        constraint_4 = Z[i][j] - (n-1) * X[i][j]
                        cqm.add_constraint(constraint_4, sense="<=", rhs=0, label=f"Constraint 4-{i+1}{j+1}")

        # Return the cqm
        return cqm

    def __call__(self,
                 token: str) -> None:
        """ Solves the improved QTSP.

        Parameters
        ----------
        `token` (str):
            The token for the solver.
        """
        # Define the sampler
        cqm_sampler = CQMBackend(token=token,
                                 time=self.time_limit,
                                 label='CQM-TSP')

        # Run the sampler to get the sample set
        sampleset = cqm_sampler(problem=self.cqm)

        # If the log is active, print the constraints' data
        if self.log:
            # Printing the sample set
            for c, cval in self.cqm.constraints.items():
                print(c, cval)

        # Filter the feasible samples
        feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

        # Get the first sample
        sample = feasible_sampleset.first.sample

        # If the log is active, print the sample
        if self.log:
            # Use `Formatter` for better readability
            Formatter(width=1000).fprint(feasible_sampleset)

            # Print the constraints' data
            for constraint in self.cqm.iter_constraint_data(sample):
                print(constraint.label, constraint.violation)

            # Print the constraints' data
            for c, v in self.cqm.constraints.items():
                print('lhs : ' + str(v.lhs.energy(sample)))
                print('rhs : ' + str(v.rhs))
                print('sense  : ' + str(v.sense))
                print("---")

        # Get the sample's solution
        sample_solutions = [key for key, value in sample.items() if value == 1]

        # If the log is active, print the sample's solution
        if self.log:
            print(sample_solutions)

        def containsNumber(value) -> list[int]:
            """ Checks if value contains a number and
            returns a list of numbers in the value.

            Parameters
            ----------
            `value` (str):
                The string to check.

            Returns
            -------
            `num_list` (list[int]):
                The list of numbers in the value.
            """
            # Find the numbers from the string
            return [(int(num)-1) for num in re.findall(r'\d+', value)]

        # Get the sample's coordinate sequence
        sample_coordinate_sequence = [containsNumber(solution) for solution in sample_solutions]

        # Update the graph
        self.graph.set_edges(sample_coordinate_sequence)

        # Plot the graph
        self.graph.plot()