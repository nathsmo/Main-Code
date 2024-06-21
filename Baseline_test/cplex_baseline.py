import sys
import numpy as np
from scipy.spatial import distance_matrix
import time
import cplex
from cplex.exceptions import CplexError
import time
# Use venv cplex_env - activate

# Approved functions


def load_coordinates(filename):
    """ Load coordinates from a .npy file. """
    return np.load(filename)

def save_distance_matrix(distances, filename):
    """ Save the distance matrix to a .npy file. """
    np.save(filename, distances)

def compute_distance_matrix(coords):
    """ Calculate the Euclidean distance matrix for a set of coordinates. """
    n_problems, n_nodes, _ = coords.shape
    distance_matrices = np.zeros((n_problems, n_nodes, n_nodes))
    for i in range(n_problems):
        for j in range(n_nodes):
            for k in range(n_nodes):
                if j != k:
                    distance_matrices[i, j, k] = np.linalg.norm(coords[i, j] - coords[i, k])
    return distance_matrices

# # ______________________________________________________________________________________________________________________


def tsp_cplex(distance_matrix):
    # Number of nodes
    n = len(distance_matrix)

    # Create an instance of a Cplex problem
    problem = cplex.Cplex()
    
    # Set up as a minimization problem
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Add binary variables for edges
    edge_vars = []
    for i in range(n):
        for j in range(i + 1, n):
            var_name = f"x_{i}_{j}"
            edge_vars.append(var_name)
            problem.variables.add(obj=[distance_matrix[i][j]],
                                  lb=[0],
                                  ub=[1],
                                  types=["B"],
                                  names=[var_name])

    # Add the constraints that each node must be entered and exited exactly once
    for i in range(n):
        edge_in = [f"x_{min(i, j)}_{max(i, j)}" for j in range(n) if i != j]
        problem.linear_constraints.add(
            lin_expr=[[edge_in, [1] * len(edge_in)]],
            senses=["E"],
            rhs=[2])

    # Solve the problem
    try:
        problem.solve()
    except CplexError as exc:
        print(exc)
        return None

    # Fetch results
    solution = problem.solution
    print("Solution status: ", solution.get_status(), ":", solution.status[solution.get_status()])
    print("Solution value: ", solution.get_objective_value())

    # Retrieve the solution
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if solution.get_values(f"x_{i}_{j}") > 0.5:
                edges.append((i, j))
    return edges, solution.get_objective_value()

# # Example usage
# filename = f"./data/vrp_test_1000_10_seed24601.npy"
# coordinates = load_coordinates(filename)
# distance_matrix = compute_distance_matrix(coordinates)
# distance_matrices = distance_matrix[0]

# # Solve the TSP
# solution_edges = tsp_cplex(distance_matrices)
# print("Edges in the solution:", solution_edges)


def main(data_type, n_problems, n_nodes, seed):
    start_time = time.time()

    file_path = f"./data/vrp_{data_type}_{n_problems}_{n_nodes}_seed{seed}.npy"
    filename_results = f"./data/cplex_solved_tsp_{data_type}_{n_problems}_{n_nodes}_seed{seed}.txt"

    # Load the data and create distance matrices
    coordinates = load_coordinates(file_path)
    distance_matrix = compute_distance_matrix(coordinates)

    # Initialize a list to store the costs
    total_costs = []

    # Open a text file to write the results
    with open(filename_results, 'w') as f:
        for i, distances in enumerate(distance_matrix):
            tour, cost = tsp_cplex(distances)

            total_costs.append(cost)  # Store the cost for later averaging
            result = f'Problem {i+1}: Optimal tour: {tour}, Optimal cost: {cost}\n'
            f.write(result)
        
        # Calculate the average cost
        average_cost = sum(total_costs) / len(total_costs) if total_costs else 0
        
        std_distance = np.std(total_costs)  # Calculating the standard deviation
        stdR_avg = f"Standard deviation of tour distances: {std_distance}\n Average tour distance: {average_cost}\n"
        f.write(stdR_avg)

        end_time = time.time()
        tim = f"Total processing time: {end_time - start_time:.2f} seconds"
        f.write(tim)
        
        print(stdR_avg)
        print(tim)
        print(f'Average optimal cost across all problems: {average_cost}')
        f.write(f'Average optimal cost across all problems: {average_cost}\n')

    print("All TSP results and the average cost have been saved to ", filename_results)


if __name__ == "__main__":
    # main(data_type='test', n_problems=1000, n_nodes=10, seed=24601)
    main(data_type='test', n_problems=1000, n_nodes=20, seed=24601)
    main(data_type='test', n_problems=1000, n_nodes=50, seed=24601)
    main(data_type='test', n_problems=1000, n_nodes=100, seed=24601)
