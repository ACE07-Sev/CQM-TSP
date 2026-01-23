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

__all__ = ["Backend", "CQMBackend"]

from abc import ABC, abstractmethod

import dimod
import time


class Backend(ABC):
    """ `cqm.Backend` is the base class for implementing solvers.
    """
    def __init__(
            self,
            time: int,
            label: str,
            token: str | None = None
        ) -> None:
        """ Initializes a `Backend` instance.

        Parameters
        ----------
        `token` : str, optional, default=None
            The token for the solver. If not provided, a default solver will be used.
            If no default solver is available, an error will be raised.
        `time` : int
            The time limit for the problem.
        `label` : str
            The label for the problem.
        """
        self.token = token
        self.time = time
        self.label = label

    @abstractmethod
    def __call__(
            self,
            problem: dimod.QuadraticModel | dimod.ConstrainedQuadraticModel
        ) -> dimod.SampleSet:
        """ Solves the given problem.

        Parameters
        ----------
        `problem` : dimod.QuadraticModel | dimod.ConstrainedQuadraticModel
            The problem to solve.

        Returns
        -------
        `result` : dimod.SampleSet
            The result of the problem.
        """
        pass


class CQMBackend(Backend):
    """ `cqm.CQMBackend` is a class for implementing solvers for constrained
    quadratic models.
    """
    def __init__(
            self,
            time: int,
            label: str,
            token: str | None = None
        ) -> None:
        """ Initializes a `CQMBackend` instance.

        Parameters
        ----------
        `token` : str, optional, default=None
            The token for the solver. If not provided, `dimod.ExactCQMSolver` will be used.
        `time` : int
            The time limit for the problem.
        `label` : str
            The label for the problem.
        """
        super().__init__(time, label, token)

    def __call__(
            self,
            problem: dimod.QuadraticModel | dimod.ConstrainedQuadraticModel
        ) -> dimod.SampleSet:
        """ Solves the given problem.

        Parameters
        ----------
        `problem` : dimod.QuadraticModel | dimod.ConstrainedQuadraticModel
            The problem to solve.

        Returns
        -------
        `sampleset` : dimod.SampleSet
            The result of the problem.

        Raises
        ------
        TypeError:
            - If the problem is not a Constrained Quadratic Model (CQM).
        """
        from dwave.cloud import Client # type: ignore

        if not isinstance(problem, dimod.ConstrainedQuadraticModel):
            raise TypeError(
                "The problem must be a Constrained Quadratic Model (CQM). "
                f"Received {type(problem)} instead."
            )

        if self.token is None:
            qpu = dimod.ExactCQMSolver()
            sampleset = qpu.sample_cqm(
                problem,
                label=self.label,
                time_limit=self.time
            )

            while not sampleset.done():
                time.sleep(5)

            return sampleset

        with Client.from_config(token=self.token) as client:
            qpu = client.get_solver(name="hybrid_constrained_quadratic_model_version1")
            sampleset = qpu.sample_cqm(
                problem,
                label=self.label,
                time_limit=self.time
            )

            while not sampleset.done():
                time.sleep(5)

            return sampleset.sampleset # type: ignore