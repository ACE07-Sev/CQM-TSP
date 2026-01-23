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

__all__ = ["QTSP"]

import dimod
from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dimod.serialization.format import Formatter
import numpy as np
from numpy.typing import NDArray

from cqm.backend import CQMBackend
from cqm.data import Graph
from cqm.solver import CQM
from cqm.solver.utils import contains_number


class QTSP(CQM):
    """ `cqm.QTSP` is a class for implementing a CQM for the Travelling Salesman Problem.
    """
    def __init__(
            self,
            coordinates: list[list[float]],
            time: int,
            distance_matrix: NDArray[np.float64] | None = None,
            edges: list[list[int]] | None = None,
            log: bool = True
        ) -> None:
        """ Initializes a `QTSP` instance.

        Parameters
        ----------
        `coordinates` : list[list[float]]
            The coordinates of the nodes of the graph.
        `time` : int
            The time limit for the problem.
        `distance_matrix` : NDArray[np.float64], optional, default=None
            The distance matrix of the graph.
        `edges` : list[list[int]], optional, default=None
            The edges of the graph. If no edges are provided, a
            complete graph is assumed.
        `log` : bool, optional, default=True
            Whether to log the output or not.

        Attributes
        ----------
        `graph` : cqm.data.Graph
            The graph.
        `distance_matrix` : NDArray[np.float64] | None
            The distance matrix of the graph.
        `time_limit` : int
            The time limit for the problem.
        `log` : bool, optional, default=True
            Whether to log the output or not.
        `cqm` : dimod.ConstrainedQuadraticModel
            The CQM.
        `solution` : list[list[int]]
            The solution of the CQM.
        """
        if edges is None or edges == []:
            edges = [
                [i, j] for i in range(len(coordinates)) for j in range(len(coordinates)) if i != j
            ]

        self.graph = Graph(coordinates=coordinates, edges=edges)
        self.distance_matrix = distance_matrix
        self.time_limit = time
        self.log = log
        self.cqm = self.define_CQM()
        self.solution: list[list[int]] = []  # type: ignore

    def define_CQM(self) -> dimod.ConstrainedQuadraticModel:
        """ Function to define the CQM for the given graph.

        Returns
        -------
        `cqm` : dimod.ConstrainedQuadraticModel
            The CQM.
        """
        n = self.graph.num_nodes

        if self.distance_matrix is not None:
            distance_matrix = self.distance_matrix
        else:
            distance_matrix = np.array(self.graph.calculate_distance_matrix())

        subtours = self.graph.generate_subtours()

        S = len(subtours)

        cqm = ConstrainedQuadraticModel()

        # Initialize the decision var
        X = np.zeros((n, n), dtype=object)

        for edge in self.graph.edges:
            i, j = edge
            if i != j:
                X[i][j] = Binary(f"X_{i+1}_{j+1}")

        # Define objective
        objective = quicksum(distance_matrix[i][j] * X[i][j] for i in range(n) for j in range(n))
        cqm.set_objective(objective)

        # Define single visit constraint
        for i in range(n):
            constraint_1 = quicksum(X[i][j] for j in range(n))

            # Safety check for empty constraints
            if isinstance(constraint_1, (int, float)):
                continue

            cqm.add_constraint(constraint_1 == 1, label="Constraint 1-" + str(i+1))

        for j in range(n):
            constraint_2 = quicksum(X[i][j] for i in range(n))

            # Safety check for empty constraints
            if isinstance(constraint_2, (int, float)):
                continue

            cqm.add_constraint(constraint_2 == 1, label="Constraint 2-" + str(j+1))

        # Define subtour elimination constraint
        for s in range(S):
            constraint_3 = quicksum(X[i][j] for i in (subtours[s]) for j in (subtours[s]) if i!=j)

            # Safety check for empty constraints
            if isinstance(constraint_3, (int, float)):
                continue

            cqm.add_constraint(constraint_3 <= len(subtours[s]) - 1, label="Constraint 3-" + str(s + 1))

        return cqm

    def plot(self) -> None:
        """ Plots the solution path.
        """
        if self.solution == []:
            raise ValueError("No solution found. Please solve the CQM first.")

        self.graph.set_edges(self.solution)
        self.graph.plot()

    def __call__(
            self,
            token: str | None = None
        ) -> list[list[int]]:
        """ Solves QTSP.

        Parameters
        ----------
        `token` : str, optional, default=None
            The token for the solver. Currently falls back on `ExactCQMSolver`
            if no token is provided.

        Returns
        -------
        `sample_coordinate_sequence` : list[list[int]]
            The sequence of coordinates representing the solution path.
        """
        cqm_sampler = CQMBackend(
            token=token,
            time=self.time_limit,
            label="CQM-TSP"
        )

        sampleset = cqm_sampler(problem=self.cqm)

        if self.log:
            for c, cval in self.cqm.constraints.items():
                print(c, cval)

        feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

        sample = feasible_sampleset.first.sample # type: ignore

        if self.log:
            Formatter(width=1000).fprint(feasible_sampleset)

            for constraint in self.cqm.iter_constraint_data(sample):
                print(constraint.label, constraint.violation)

            for c, v in self.cqm.constraints.items():
                print("lhs : " + str(v.lhs.energy(sample)))
                print("rhs : " + str(v.rhs))
                print("sense  : " + str(v.sense))
                print("---")

        sample_solutions = [key for key, value in sample.items() if value == 1]

        if self.log:
            print(sample_solutions)

        sample_coordinate_sequence = [contains_number(solution) for solution in sample_solutions]

        self.solution = sample_coordinate_sequence

        return sample_coordinate_sequence