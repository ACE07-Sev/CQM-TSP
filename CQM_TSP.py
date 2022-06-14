import math
import os

import geopy
from dimod.serialization.format import Formatter

os.chdir('..')
from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler
import numpy as np
import itertools
from matplotlib import pyplot as plt


# Function to generate the subsets of a list given size
def findsubsets(s, n):
    return list(itertools.combinations(s, n))


# Function to generate all the subsets of a list
def find_all_subsets(s):
    subtours = []
    temp_list = []
    for i in range(2, len(s)-1):
        temp_list = findsubsets(s, i)
        for j in range(len(temp_list)):
            subtours.append(temp_list[j])
    return subtours


# Function for calculating the euclidean distance
def distance_between_points(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2)


# Function for calculating the distance using coordinates
def distance_two_coordinates(point_A, point_B):
    lat1 = math.pi * ((point_A[0] // 1) + 5.0 * (point_A[0] % 1) / 3.0) / 180.0
    lon1 = math.pi * ((point_A[1] // 1) + 5.0 * (point_A[1] % 1) / 3.0) / 180.0
    lat2 = math.pi * ((point_B[0] // 1) + 5.0 * (point_B[0] % 1) / 3.0) / 180.0
    lon2 = math.pi * ((point_B[1] // 1) + 5.0 * (point_B[1] % 1) / 3.0) / 180.0

    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    radius = 6378.388

    distance = radius * math.acos(1 / 2 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1

    if distance == 1.0:
        return 0

    return round(geopy.distance.great_circle(point_A, point_B).km)


# dataset for distance matrix
C_ = [[0, 2, 2, 4, 2],
      [2, 0, 2, 2, 3],
      [2, 2, 0, 3, 2],
      [4, 2, 3, 0, 3],
      [2, 2, 2, 3, 0]]

C_1 = [[0.0, 2.23606797749979, 2.23606797749979, 3.1622776601683795, 4.0],
       [2.23606797749979, 0.0, 1.4142135623730951, 1.0, 2.23606797749979],
       [2.23606797749979, 1.4142135623730951, 0.0, 2.23606797749979, 3.605551275463989],
       [3.1622776601683795, 1.0, 2.23606797749979, 0.0, 1.4142135623730951],
       [4.0, 2.23606797749979, 3.605551275463989, 1.4142135623730951, 0.0]]

# Initializing the CQM
cqm = ConstrainedQuadraticModel()
# Initializing the objective
objective = 0
# Initializing the constraints
constraint_1 = 0
constraint_2 = 0
constraint_3 = 0
# number of nodes
n = 5
global_set = [1, 2, 3, 4, 5]
# list of subtours
subtours = find_all_subsets(global_set)
print(subtours)
# number of subsets
S = len(subtours)
print(S)
# # Distance Matrix
# C_in = []
# temp_set = []
# coordinates = []
# # Inputting the coordinates
# for i in range(n):
#     print("Input x coordinate")
#     point_x = float(input())
#     print("Input y coordinate")
#     point_y = float(input())
#     coordinates.append([point_x, point_y])
#     plt.scatter(point_x, point_y)
# # Dot plot
# plt.show()
# # Generating the distance matrix
# for i in range(n):
#     for j in range(n):
#         temp_set.append(distance_between_points(coordinates[i], coordinates[j]))
#     C_in.append(temp_set)
#     temp_set = []
# # Distnace Matrix
# print(C_in)

# Initializing the decision var
X_ = []

# Objective
for i in range(n):
    for j in range(n):
        if i == j:
            X_.append(0)
        else:
            X_.append(Binary('X_' + str(i + 1) + "_" + str(j + 1)))
    objective += quicksum(C_1[j][i] * X_[j] for j in range(n))
    X_.clear()

# Assignment Constraints (Only one entry and exit per node)
for i in range(n):
    for j in range(n):
        if i == j:
            X_.append(0)
        else:
            X_.append(Binary('X_' + str(i + 1) + "_" + str(j + 1)))
    constraint_1 = quicksum(X_[j] for j in range(n))
    cqm.add_constraint(constraint_1 == 1, label="Constraint 1-" + str(i + 1))
    X_.clear()

for j in range(n):
    for i in range(n):
        if i == j:
            X_.append(0)
        else:
            X_.append(Binary('X_' + str(i + 1) + "_" + str(j + 1)))
    constraint_2 = quicksum(X_[i] for i in range(n))
    cqm.add_constraint(constraint_2 == 1, label="Constraint 2-" + str(j + 1))
    X_.clear()

# subtour elimination constraint
for s in range(S):  # s = {1,2}, length of s = 1, so 1 iterations of below
    for i in range(len(subtours[s])):   # len of subtour = 2, so 2 iterations
        for j in range(len(subtours[s])):   # len of subtour = 2 so 2 iterations
            # possible X_ : X1,1 X1,2 X2,1 X2,2
            if i == j:
                continue   # X1,1 and X2,2 are not accepted
            else:
                X_.append(Binary('X_' + str(subtours[s][i]) + "_" + str(subtours[s][j])))
    constraint_3 = quicksum(X_[j] for j in range(len(subtours[s])))
    cqm.add_constraint(constraint_3 <= len(subtours[s]) - 1, label="Constraint 3-" + str(s + 1))
    X_.clear()

# Running the sampler to get the sample set
cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label='CQMAmirali', time_limit=5)

# Printing the sample set
for c, cval in cqm.constraints.items():
    print(c, cval)

feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

sample = feasible_sampleset.first.sample
Formatter(width=1000).fprint(feasible_sampleset)
# print('')
# print('')
# Formatter(width=1000).fprint(sampleset)
# print(feasible_sampleset)

for constraint in cqm.iter_constraint_data(sample):
    print(constraint.label, constraint.violation)

for c, v in cqm.constraints.items():
    print('lhs : ' + str(v.lhs.energy(sample)))
    print('rhs : ' + str(v.rhs))
    print('sense  : ' + str(v.sense))
    print("---")
