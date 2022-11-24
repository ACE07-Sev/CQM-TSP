import os

from dimod.serialization.format import Formatter

os.chdir('..')
from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler
import numpy as np
import itertools
from matplotlib import pyplot as plt


def containsNumber(value):
    num_list = []
    for character in value:
        if character.isdigit():
            num_list.append(int(character) - 1)
    return num_list


# Function to generate the subsets of a list given size
def findsubsets(s, n):
    return list(itertools.combinations(s, n))


# Function to generate all the subsets of a list
def find_all_subsets(s):
    subtours = []
    temp_list = []
    for i in range(2, len(s)):
        temp_list = findsubsets(s, i)
        for j in range(len(temp_list)):
            subtours.append(temp_list[j])
    return subtours


# Function for calculating the euclidean distance
def distance_between_points(point_A, point_B):
    return np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2)


# Initializing the CQM
cqm = ConstrainedQuadraticModel()
# Initializing the objective
objective = 0
# Initializing the constraints
constraint_1 = 0
constraint_2 = 0
constraint_3 = 0
# List of coordinates
# coordinates = np.array([[1, 1], [2, 3], [3, 2], [2, 4], [1, 5], [3, 6], [4, 3]])

coordinates = np.array([[1, 1], [2, 3], [3, 2], [2, 4], [1, 5], [3, 6], [1, 3], [2, 6], [2, 5], [3, 3], [1, 2], [4, 3], [2, 0.5], [3, 5], [2, 2]])
# number of nodes
n = len(coordinates)
global_set = [i for i in range(n)]
# list of subtours
subtours = find_all_subsets(global_set)
print(subtours)
# number of subsets
S = len(subtours)
print(S)

x_vals = coordinates[:, 0]
y_vals = coordinates[:, 1]

# Distance Matrix
Distance_matrix = []
temp_set = []
# Generating the distance matrix
for i in range(n):
    for j in range(n):
        temp_set.append(distance_between_points(coordinates[i], coordinates[j]))
    Distance_matrix.append(temp_set)
    temp_set = []
print(Distance_matrix)

# Initializing the decision var
X_ = []

# Objective
for i in range(n):
    for j in range(n):
        if i == j:
            X_.append(0)
        else:
            X_.append(Binary('X_' + str(i + 1) + "_" + str(j + 1)))
    objective += quicksum(Distance_matrix[j][i] * X_[j] for j in range(n))
    cqm.set_objective(objective)
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
    for i in (subtours[s]):  # len of subtour = 2, so 2 iterations
        for j in (subtours[s]):  # len of subtour = 2 so 2 iterations
            # possible X_ : X1,1 X1,2 X2,1 X2,2
            if i == j:
                continue  # X1,1 and X2,2 are not accepted
            else:
                X_.append(Binary('X_' + str(i + 1) + "_" + str(j + 1)))
    constraint_3 = quicksum(X_[j] for j in range(len(X_)))
    cqm.add_constraint(constraint_3 <= len(subtours[s]) - 1, label="Constraint 3-" + str(s + 1))
    X_.clear()

# Running the sampler to get the sample set
cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label='CQMAmirali', time_limit=15)

# Printing the sample set
for c, cval in cqm.constraints.items():
    print(c, cval)

feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

sample = feasible_sampleset.first.sample
Formatter(width=1000).fprint(feasible_sampleset)

for constraint in cqm.iter_constraint_data(sample):
    print(constraint.label, constraint.violation)

for c, v in cqm.constraints.items():
    print('lhs : ' + str(v.lhs.energy(sample)))
    print('rhs : ' + str(v.rhs))
    print('sense  : ' + str(v.sense))
    print("---")

sample_keys = sample.keys()
sample_solutions = []
for key in sample_keys:
    if sample.get(key) == 1:
        sample_solutions.append(key)
print(sample_solutions)

sample_coordinate_sequence = []
for i in range(len(sample_solutions)):
    res = containsNumber(sample_solutions[i])
    sample_coordinate_sequence.append(res)

plt.scatter(x_vals, y_vals)

for i in sample_coordinate_sequence:
    start = coordinates[i[0]]
    end = coordinates[i[1]]
    plt.plot([start[0], end[0]], [start[1], end[1]])

plt.show()
