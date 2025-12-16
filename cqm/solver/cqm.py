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

__all__ = ["CQM"]

from abc import ABC, abstractmethod
from typing import Any

import dimod


class CQM(ABC):
    """ `cqm.CQM` is the base class for implementing constrained quadratic models.
    """
    def __init__(
            self,
            time: int,
            log: bool = True
        ) -> None:
        """ Initializes a `CQM` instance.

        Parameters
        ----------
        `time` (int):
            The time limit for the problem.
        `log` (bool):
            Whether to log the output or not.

        Attributes
        ----------
        `time_limit` : int
            The time limit for the problem.
        `log` : bool
            Whether to log the output or not.
        `cqm` : dimod.ConstrainedQuadraticModel
            The CQM.
        `solution` : Any
            The solution of the CQM.
        """
        self.time_limit = time
        self.log = log
        self.cqm = self.define_CQM()
        self.solution: Any = None

    @abstractmethod
    def define_CQM(self) -> dimod.ConstrainedQuadraticModel:
        """ Function to define the CQM for the given graph.

        Returns
        -------
        `cqm` : dimod.ConstrainedQuadraticModel
            The CQM.
        """
        pass

    @abstractmethod
    def __call__(
            self,
            token: str | None = None
        ) -> list[list[int]]:
        """ Solves the TSP CQM.

        Parameters
        ----------
        `token` (str, optional):
            The token for the solver.

        Returns
        -------
        `sample_coordinate_sequence` (list[list[int]]):
            The sequence of coordinates representing the solution path.
        """
        pass