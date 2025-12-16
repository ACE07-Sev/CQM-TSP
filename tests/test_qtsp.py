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

from cqm import QTSP


def test_qtsp_no_edges() -> None:
    coordinates = [
        [1, 1],
        [2, 3],
        [3, 2],
        [2, 4],
        [1, 5]
    ]

    tsp_model = QTSP(coordinates=coordinates, time=50, log=False)

    assert tsp_model() == [[0, 2], [1, 3], [2, 1], [3, 4], [4, 0]]

def test_qtsp_with_edges() -> None:
    coordinates = [
        [1, 1],
        [2, 3],
        [3, 2],
        [2, 4]
    ]
    edges = [
        [1, 2],
        [3, 0],
        [2, 3],
        [0, 1]
    ]
    
    tsp_model = QTSP(coordinates=coordinates, edges=edges, time=50, log=False)
    
    assert tsp_model() == [[0, 1], [1, 2], [2, 3], [3, 0]]