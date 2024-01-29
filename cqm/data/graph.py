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

__all__ = ['Graph']

from abc import ABC
from collections.abc import Iterable
import os
os.chdir('..')

import numpy as np
import itertools
from matplotlib import pyplot as plt


class Graph(ABC):
    """ `CQM.Graph` class represents graph data, such as maps whcih
        are used in the TSP.
    """
    def __init__(self,
                 coordinates: Iterable[Iterable[float]],
                 edges: Iterable[Iterable[int]] | None = None) -> None:
        """ 
        Initializes a graph model.

        Parameters
        ----------
        `coordinates` (Iterable[Iterable[float]]):
            The coordinates of the nodes of the graph.
        `edges` (Iterable[Iterable[int]]):
            The edges of the graph.
        """
        self.coordinates = coordinates
        self.num_nodes = len(coordinates)
        self.edges = edges

    def calculate_distance_matrix(self) -> Iterable[Iterable[float]]:
        """ Defines the distance matrix for the given coordinates.

        Returns
        -------
        `distance_matrix` (Iterable[Iterable[float]]):
            The distance matrix.
        """
        # Initialize the distance matrix
        distance_matrix = []

        def distance_between_points(point_A: Iterable[int],
                                    point_B: Iterable[int]) -> float:
            """ Function for calculating the euclidean distance.

            Parameters
            ----------
            `point_A` (Iterable[int]):
                The first point.
            `point_B` (Iterable[int]):
                The second point.

            Returns
            -------
            `distance` (float):
                The euclidean distance between the two points.
            """
            return np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2)

        # Calculate the distance matrix
        for a in self.coordinates:
            distance_matrix.append([distance_between_points(a, b) for b in self.coordinates])

        # Return the distance matrix
        return distance_matrix

    def generate_subtours(self) -> Iterable[Iterable[int]]:
        """ Function to generate the subtours for the given graph.

        Returns
        -------
        `subtours` : Iterable[Iterable[int]]
            The subtours.
        """
        # Initialize subtours list
        subtours = []

        def find_subsets(s: int,
                         n: int) -> Iterable[Iterable[int]]:
            """ Return the list of all subsets of length n in s.

            Parameters
            ----------
            `s` (int):
                The number of nodes.
            `n` (int):
                The length of the subsets.

            Returns
            -------
            `subsets` (Iterable[Iterable[int]]):
                The list of all subsets of length n in s.
            """
            return list(itertools.combinations(s, n))

        # Generate subtours
        for i in range(2, self.num_nodes):
            subtours.extend(find_subsets(range(self.num_nodes), i))

        # Return the subtours
        return subtours

    def set_edges(self,
                  edges: Iterable[Iterable[int]]) -> None:
        """ Sets the edges of the graph.

        Parameters
        ----------
        `edges` (Iterable[Iterable[int]]):
            The edges of the graph.
        """
        self.edges = edges

    def plot(self) -> None:
        """ Plots the graph.
        """
        # Plot the graph
        plt.figure(figsize=(10, 10))
        plt.scatter([i[0] for i in self.coordinates], [i[1] for i in self.coordinates], color='red')
        for i in self.edges:
            plt.plot([self.coordinates[i[0]][0], self.coordinates[i[1]][0]],
                     [self.coordinates[i[0]][1], self.coordinates[i[1]][1]], color='blue')
        plt.show()