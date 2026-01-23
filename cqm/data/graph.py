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

__all__ = ["Graph"]

from abc import ABC

import itertools
import matplotlib.pyplot as plt # type: ignore
import numpy as np


class Graph(ABC):
    """ `cqm.Graph` class represents graph data, such as maps whcih
    are used in the TSP.
    """
    def __init__(
            self,
            coordinates: list[list[float]],
            edges: list[list[int]] = [],
            distance_matrix: list[list[float]] | None = None
        ) -> None:
        """
        Initializes a graph model.

        Parameters
        ----------
        `coordinates` : list[list[float]]
            The coordinates of the nodes of the graph.
        `edges` : list[list[int]], optional, default=[]
            The edges of the graph.
        `distance_matrix` : list[list[float]], optional, default=None
            The distance matrix of the graph.
        """
        self.coordinates = coordinates
        self.num_nodes = len(coordinates)
        self.edges = edges
        self.distance_matrix = distance_matrix

    def calculate_distance_matrix(self) -> list[list[float]]:
        """ Defines the distance matrix for the given coordinates.

        Returns
        -------
        `distance_matrix` : list[list[float]]
            The distance matrix.
        """
        if self.distance_matrix is not None:
            return self.distance_matrix

        distance_matrix = []

        def distance_between_points(
                point_A: list[float],
                point_B: list[float]
            ) -> float:
            """ Function for calculating the Euclidean distance.

            Parameters
            ----------
            `point_A` : list[float]
                The first point.
            `point_B` : list[float]
                The second point.

            Returns
            -------
            `distance` : float
                The Euclidean distance between the two points.
            """
            return np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2)

        for a in self.coordinates:
            distance_matrix.append([distance_between_points(a, b) for b in self.coordinates])

        return distance_matrix

    def generate_subtours(self) -> list[list[int]]:
        """ Function to generate the subtours for the given graph.

        Returns
        -------
        `subtours` : list[list[int]]
            The subtours.
        """
        subtours: list[list[int]] = []

        def find_subsets(
                s: int,
                n: int
            ) -> list[list[int]]:
            """ Return the list of all subsets of length n in s.

            Parameters
            ----------
            `s` : int
                The number of nodes.
            `n` : int
                The length of the subsets.

            Returns
            -------
            `subsets` : list[list[int]]
                The list of all subsets of length n in s.
            """
            return [list(combo) for combo in itertools.combinations(range(s), n)]

        for i in range(2, self.num_nodes):
            subtours.extend(find_subsets(self.num_nodes, i))

        return subtours

    def set_edges(
            self,
            edges: list[list[int]]
        ) -> None:
        """ Sets the edges of the graph.

        Parameters
        ----------
        `edges` : list[list[int]]
            The edges of the graph.
        """
        self.edges = edges

    def plot(self) -> None:
        """ Plots the graph.
        """
        plt.figure(figsize=(10, 10))
        plt.scatter([i[0] for i in self.coordinates], [i[1] for i in self.coordinates], color="red")

        for i in self.edges:
            plt.plot(
                [self.coordinates[i[0]][0], self.coordinates[i[1]][0]],
                [self.coordinates[i[0]][1], self.coordinates[i[1]][1]],
                color="blue"
            )

        plt.show()