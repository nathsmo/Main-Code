from gurobipy import Model, GRB, quicksum
import itertools
import numpy as np
import sys

def load_data_and_create_distance_matrix(filepath):
    """
    Load coordinate data from a .npy file and convert it into a distance matrix.
    
    Args:
        filepath (str): The path to the .npy file containing the coordinates.
    
    Returns:
        np.ndarray: A 3D array where each 2D array element represents the distance matrix
                    for a set of problems (nodes).
    """
    coords = np.load(filepath)
    n_problems, n_nodes, _ = coords.shape
    distance_matrices = np.zeros((n_problems, n_nodes, n_nodes))
    for i in range(n_problems):
        for j in range(n_nodes):
            for k in range(n_nodes):
                if j != k:
                    distance_matrices[i, j, k] = np.linalg.norm(coords[i, j] - coords[i, k])
    return distance_matrices

def subtourelim(model, where, n):
    """
    Callback function for the optimization model to eliminate subtours.

    Args:
        model (Model): The Gurobi model.
        where (int): Callback code indicating why the callback was called.
        n (int): The number of nodes in the problem.
    """
    if where == GRB.Callback.MIPSOL:
        selected = [(i, j) for i in range(n) for j in range(n) if i != j and model.cbGetSolution(model._vars[i, j]) > 0.5]
        tour = subtour(selected, n)
        if len(tour) < n:
            model.cbLazy(quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2)) <= len(tour)-1)

def subtour(edges, n):
    """
    Find a subtour in a list of edges.

    Args:
        edges (list of tuples): Selected edges in the current solution.
        n (int): Total number of nodes.

    Returns:
        list: A list of nodes that forms the shortest cycle found.
    """
    unvisited = list(range(n))
    cycle = []
    for edge in edges:
        if edge[0] in unvisited:
            start = edge[0]
            # Perform a depth-first search to find a cycle
            stack = [start]
            path = []
            while stack:
                node = stack.pop()
                if node in unvisited:
                    unvisited.remove(node)
                    path.append(node)
                    stack.extend([j for i, j in edges if i == node and j in unvisited])
            if not cycle or (path and len(path) < len(cycle)):
                cycle = path
    return cycle

def solve_tsp(distances):
    """
    Find a subtour in a list of edges.

    Args:
        edges (list of tuples): Selected edges in the current solution.
        n (int): Total number of nodes.

    Returns:
        list: A list of nodes that forms the shortest cycle found.
    """
    n = len(distances)  # Number of nodes
    model = Model('TSP')

    # Create variables
    vars = {}
    for i in range(n):
        for j in range(i+1):
            if i != j:
                vars[i, j] = model.addVar(obj=distances[i][j], vtype=GRB.BINARY, name='e{}_{}'.format(i, j))
                vars[j, i] = vars[i, j]  # Mirror image

    # Add degree-2 constraint
    for i in range(n):
        model.addConstr(quicksum(vars[i, j] for j in range(n) if i != j) == 2)

    model._vars = vars
    model.Params.lazyConstraints = 1

    # Set the callback using a lambda to avoid scope issues
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            selected = [(i, j) for i in range(n) for j in range(n) if i != j and model.cbGetSolution(model._vars[i, j]) > 0.5]
            tour = subtour(selected, n)
            if len(tour) < n:
                model.cbLazy(quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2)) <= len(tour)-1)

    model.optimize(callback)

    # Retrieve solution
    solution = model.getAttr('X', vars)
    selected = [(i, j) for i in range(n) for j in range(n) if i != j and solution[i, j] > 0.5]
    #print('Optimal tour:', selected)
    #print('Optimal cost:', model.ObjVal)
    return selected, model.ObjVal
 



def main(data_type, n_problems, n_nodes, seed):
    file_path = f"./data/vrp_{data_type}_{n_problems}_{n_nodes}_seed{seed}.npy"
    filename_results = f"./data/gurobi_solved_tsp_{data_type}_{n_problems}_{n_nodes}_seed{seed}.txt"

    # Load the data and create distance matrices
    # file_path = './data/vrp_test_1000_10_seed24601.npy'
    distance_matrices = load_data_and_create_distance_matrix(file_path)
    
    # Solve the TSP for the first dataset in the file
    # solve_tsp(distance_matrices[0])

    # Initialize a list to store the costs
    total_costs = []

    # Open a text file to write the results
    with open(filename_results, 'w') as f:
        for i, distances in enumerate(distance_matrices):
            #print(f'Solving TSP for problem {i+1}...')
            #print(f'Problem {i+1}: Distances:\n{distances}')
            tour, cost = solve_tsp(distances)
            total_costs.append(cost)  # Store the cost for later averaging
            result = f'Problem {i+1}: Optimal tour: {tour}, Optimal cost: {cost}\n'
            #print(result, end='')
            f.write(result)
        
        # Calculate the average cost
        average_cost = sum(total_costs) / len(total_costs) if total_costs else 0
        print(f'Average optimal cost across all problems: {average_cost}')
        f.write(f'Average optimal cost across all problems: {average_cost}\n')

    print("All TSP results and the average cost have been saved to ", filename_results)


if __name__ == "__main__":
    main(data_type = 'test', 
    n_problems = 1000,
    seed = 24601,
    n_nodes = 20)
    main(data_type = 'test', 
    n_problems = 1000,
    seed = 24601,
    n_nodes = 50)
