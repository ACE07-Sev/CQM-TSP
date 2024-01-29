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
from abc import ABC, abstractmethod

__all__ = ['Backend', 'CQMBackend']

import dimod
from dwave.cloud import Client

import time


class Backend(ABC):
    """ `Solver` is the base class for implementing solvers.

    Parameters
    ----------
    `token` (str):
        The token for the solver.
    `time` (int):
        The time limit for the problem.
    `label` (str):
        The label for the problem.
    """
    def __init__(self,
                 token: str,
                 time: int,
                 label: str) -> None:
        self.token = token
        self.time = time
        self.label = label

    @abstractmethod
    def __call__(self,
                 problem: dimod.QuadraticModel) -> dimod.SampleSet:
        """ Solves the given problem.

        Parameters
        ----------
        `problem` (dimod.QuadraticModel):
            The problem to solve.

        Returns
        -------
        `result` (dimod.SampleSet):
            The result of the problem.
        """
        pass


class CQMBackend(Backend):
    """ `CQM_Solver` is a class for implementing solvers for constrained quadratic models.

    Parameters
    ----------
    `token` (str):
        The token for the solver.
    `time` (int):
        The time limit for the problem.
    `label` (str):
        The label for the problem.
    """
    def __init__(self,
                 token: str,
                 time: int,
                 label: str) -> None:
        super().__init__(token, time, label)

    def __call__(self,
                 problem: dimod.ConstrainedQuadraticModel) -> dimod.SampleSet:
        """ Solves the given problem.

        Parameters
        ----------
        `problem` (dimod.ConstrainedQuadraticModel):
            The problem to solve.

        Returns
        -------
        `sampleset` (dimod.SampleSet):
            The result of the problem.
        """
        # Connect using the default or environment connection information
        with Client.from_config(token=self.token) as client:
            # Define QPU
            qpu = client.get_solver(name="hybrid_constrained_quadratic_model_version1")

            # Sample the CQM
            sampleset = qpu.sample_cqm(problem, label=self.label, time_limit=self.time)

            # Wait until it finishes
            while not sampleset.done():
                time.sleep(5)

            return sampleset.sampleset