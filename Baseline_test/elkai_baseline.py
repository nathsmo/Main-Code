import sys
import elkai
import numpy as np
from scipy.spatial import distance_matrix
import time

def create_distance_matrices(data):
    matrices = []
    for problem in data:
        matrices.append(distance_matrix(problem, problem))
    return matrices

def solve_tsp_problems(matrices):
    solutions = []
    for matrix in matrices:
        tour = elkai.solve_int_matrix(np.rint(matrix).astype(int))
        solutions.append(tour)
    return solutions

def calculate_tour_distance(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i+1]]
    total_distance += distance_matrix[tour[-1], tour[0]]
    return total_distance

def main(data_type, n_problems, n_nodes, seed):
    start_time = time.time()
    
    file_path = f"./data/vrp_{data_type}_{n_problems}_{n_nodes}_seed{seed}.npy"
    filename_results = f"./data/lkh3_solved_tsp_{data_type}_{n_problems}_{n_nodes}_seed{seed}.txt"

    data = np.load(file_path)
    print('Data loaded successfully from:', file_path)
    
    all_matrices = create_distance_matrices(data)
    solutions = solve_tsp_problems(all_matrices)
    print('All problems solved successfully!')
    
    tour_distances = [calculate_tour_distance(tour, matrix) for tour, matrix in zip(solutions, all_matrices)]
    
    # Writing results to a file
    with open(filename_results, "w") as file:
        for tour, distance in zip(solutions, tour_distances):
            line = f"Tour: {tour}, Distance: {distance}\n"
            file.write(line)
            # print(line.strip())  # Print to console as well
        
        average_distance = sum(tour_distances) / len(tour_distances)
        std_distance = np.std(tour_distances)  # Calculating the standard deviation

        stdR_avg = f"Standard deviation of tour distances: {std_distance}\n Average tour distance: {average_distance}\n"
        file.write(stdR_avg)

        end_time = time.time()
        tim = f"Total processing time: {end_time - start_time:.2f} seconds"
        file.write(tim)
        
    print(stdR_avg)
    print(tim)
    print('* All done! *')

if __name__ == "__main__":
    main(data_type='test', n_problems=1000, n_nodes=10, seed=24601)
    main(data_type='test', n_problems=1000, n_nodes=20, seed=24601)
    main(data_type='test', n_problems=1000, n_nodes=50, seed=24601)
    main(data_type='test', n_problems=1000, n_nodes=100, seed=24601)
