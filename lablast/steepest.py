import time
import warnings
from time import sleep
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

from utils import shift_2d_to_3d, min_2nd_3rd

n_nodes = 200
sol_size = n_nodes//2
pop_size = 1000

#seeed
torch.manual_seed(738454387)


def perturb_swap(solutions):
    num_rows, num_cols = solutions.shape
    # Generate a random index for each row
    i = torch.randint(num_cols, (num_rows,))

    # Compute the next index for each row (circular)
    next_i = (i + 1) % num_cols

    # Perform the swap for each row
    arangement = torch.arange(num_rows)
    solutions[arangement, i], solutions[arangement, next_i] = \
    solutions[arangement, next_i], solutions[arangement, i]

    return solutions


def perturb_reverse_reverse(solutions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        num_rows, num_cols = solutions.shape

        # Length from 3 to 8
        lengths = torch.randint(3, 9, (1,))
        i_values = torch.randint(0, num_cols - lengths + 1, (1,))
        j_values = i_values + lengths
        flipped = solutions[:, i_values:j_values].clone().flip(1)
        solutions[torch.arange(num_rows), i_values:j_values] = flipped

        # Update indices for second reverse
        k_values = torch.randint(i_values, j_values, (1,))
        h_values = torch.randint(k_values, j_values + 1, (1,))

        flipped = solutions[:, k_values:h_values].clone().flip(1)
        solutions[torch.arange(num_rows), k_values:h_values] = flipped

        return solutions


def perturb_exchange(solutions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        #3 to 5
        how_many = torch.randint(3, 6, (1,))
        num_rows, num_cols = solutions.shape
        # Generate a random index for each row
        i = torch.randint(num_cols, (num_rows,how_many))
        j  = torch.randint(num_cols, (num_rows,how_many))
        # Compute the next index for each row (circular)

        # Perform the swap for each row
        arangement = torch.arange(num_rows)
        solutions = solutions.clone()
        for k in range(how_many):
            solutions[arangement, i[:,k]], solutions[arangement, j[:,k]] = \
                solutions[arangement, j[:,k]], solutions[arangement, i[:,k]]
            solutions = solutions.clone()#TODO inefficient
        return solutions
def perturb_shuffle(solutions):
    num_rows, num_cols = solutions.shape
    length = torch.randint(low=10, high=min(21, num_cols+1), size=(1,)).item()
    start_index = torch.randint(low=0, high=num_cols - length + 1, size=(1,)).item()

    # Generate random indices for the specified subsequence in each row
    random_indices = torch.randperm(length)

    # Shuffle the elements within the specified subsequence
    solutions[:, (start_index):(start_index+length)] = solutions[:, (start_index):(start_index+length)][:, random_indices]

    return solutions


def perturbe(solutions):
    permutation_number = torch.randint(0, 4, (solutions.shape[0],))
    solutions = solutions.clone()
    solutions[permutation_number == 0] = perturb_swap(solutions[permutation_number == 0])
    solutions[permutation_number == 1] = perturb_reverse_reverse(solutions[permutation_number == 1])
    solutions[permutation_number == 2] = perturb_exchange(solutions[permutation_number == 2])
    solutions[permutation_number == 3] = perturb_shuffle(solutions[permutation_number == 3])
    return solutions


def ils(distances, costs, max_seconds=60):
    number_of_local_search_iterations = 0
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    costs = costs.to(device)
    distances = distances.to(device)
    #holder = vector from 0 to n_nodes
    holder = torch.arange(n_nodes, device=device)
    holder = holder.repeat((pop_size, 1))
    bool_marker = torch.zeros(pop_size, n_nodes, device=device, dtype=torch.bool)
    #range_tensor
    range_tensor = torch.arange(sol_size, device=device).repeat((pop_size, 1))

    def  calc_cost_batched(solutions):
        dist_cost = torch.sum(distances[solutions, solutions.roll(-1, dims=1)], dim=-1)
        c = costs[solutions]
        other_costs = torch.sum(c, dim=1)

        # Total cost
        total_cost = dist_cost + other_costs

        return total_cost

    def inter_move(solutions):
        cur_bool_marker = bool_marker.clone()
        cur_bool_marker.scatter_(1, solutions, True)
        unvisited_nodes = holder[~cur_bool_marker].view(pop_size, -1)

        #repeat last dime
        current_nodes = solutions.unsqueeze(-1).expand(-1, -1, solutions.shape[1])
        previous_nodes = torch.roll(current_nodes, shifts=1, dims=1)
        next_nodes = torch.roll(current_nodes, shifts=-1, dims=1)
        cur_prev_distances = distances[previous_nodes, current_nodes]
        cur_next_distances = distances[current_nodes, next_nodes]
        cur_distance_costs = cur_prev_distances + cur_next_distances
        cur_node_costs = costs[current_nodes]
        cur_costs = cur_distance_costs + cur_node_costs

        new_nodes = unvisited_nodes.unsqueeze(-1).expand(-1, -1, unvisited_nodes.shape[1]).transpose(1, 2)
        new_prev_distances = distances[previous_nodes, new_nodes]
        new_next_distances = distances[new_nodes, next_nodes]
        new_distance_costs = new_prev_distances + new_next_distances
        new_node_costs = costs[new_nodes]
        new_costs = new_distance_costs + new_node_costs

        delta = new_costs - cur_costs

        best_move_i, best_move_j = min_2nd_3rd(delta)
        best_move_move = new_nodes[torch.arange(pop_size), best_move_i, best_move_j]
        best_delta = delta[torch.arange(pop_size), best_move_i, best_move_j]

        return best_delta, best_move_i, best_move_j, best_move_move


    def reverse_op(solutions, solutions_mask,  i_indices, j_indices):
        j_indices = j_indices+1
        lengths = j_indices - i_indices
        range_tensor_masked = range_tensor[solutions_mask]
        corrected_indices = i_indices.unsqueeze(1) + lengths.unsqueeze(1) - 1 - (range_tensor_masked - i_indices.unsqueeze(1))
        corrected_indices = torch.clamp(corrected_indices, 0, solutions.size(1) - 1)  # Clamping to avoid out-of-bounds

        mask = (range_tensor_masked >= i_indices.unsqueeze(1)) & (range_tensor_masked < j_indices.unsqueeze(1))

        solutions[mask] = torch.gather(solutions, 1, corrected_indices)[mask]
        return solutions


    def intra_move(solutions):

        edge1_first = solutions.unsqueeze(-1).expand(-1, -1, solutions.shape[1])
        edge1_second = edge1_first.roll(-1, dims=1)

        edge2_first = solutions.unsqueeze(-1).expand(-1, -1, solutions.shape[1]).transpose(1, 2)
        edge2_second = edge2_first.roll(-1, dims=2)

        before_cost = distances[edge1_first, edge1_second] + distances[edge2_first, edge2_second]
        after_cost = distances[edge1_first, edge2_first] + distances[edge1_second, edge2_second]

        deltas = after_cost - before_cost
        #make upper traingular +2 to avoid self loops
        deltas[torch.tril(torch.ones_like(deltas, dtype=torch.bool), diagonal=1)] = 999999


        best_move_i, best_move_j = min_2nd_3rd(deltas)
        best_delta = deltas[torch.arange(pop_size), best_move_i, best_move_j]


        best_move_i+=1
        return best_delta,best_move_i, best_move_j

    population_factory = torch.rand((pop_size, n_nodes), device=device)
    _, population = torch.topk(population_factory, sol_size, dim=1, largest=False)
    population = population
    population_costs = calc_cost_batched(population)
    best_solution = population[torch.argmin(population_costs)]
    best_solution_cost = torch.min(population_costs)
    while (time.time() - start_time) < max_seconds:
        #inter move
        best_inter_delta, best_inter_i, best_inter_j, best_inter_move = inter_move(population)
        #intra move
        best_intra_delta, best_intra_i, best_intra_j = intra_move(population)
        #apply best move
        best_inter_mask = (best_inter_delta <= best_intra_delta) & (best_inter_delta < 0)
        best_intra_mask = (best_intra_delta < best_inter_delta) & (best_intra_delta < 0)
        #apply inter
        population[best_inter_mask, best_inter_i[best_inter_mask]] = best_inter_move[best_inter_mask]
        population_costs[best_inter_mask] += best_inter_delta[best_inter_mask]
        #apply intra
        population[best_intra_mask] = reverse_op(population[best_intra_mask], best_intra_mask, best_intra_i[best_intra_mask], best_intra_j[best_intra_mask])
        population_costs[best_intra_mask] += best_intra_delta[best_intra_mask]
        # #find best solution
        current_best_cost, best_idx = torch.min(population_costs, dim=0)
        if current_best_cost < best_solution_cost:
            best_solution_cost = current_best_cost
            best_solution = population[best_idx].detach().cpu()

        #perturbe not improvable
        best_delta = torch.min(torch.stack([best_intra_delta, best_inter_delta]), dim=0)[0]
        not_improvable = best_delta >= 0
        if torch.any(not_improvable):
            number_of_local_search_iterations += torch.sum(not_improvable).item()
            population[not_improvable] = perturbe(population[not_improvable])
            population_costs[not_improvable] = calc_cost_batched(population)[not_improvable]

    return best_solution, best_solution_cost, number_of_local_search_iterations


DATA_PATH = Path("../data")

def main():
    for file in sorted(DATA_PATH.iterdir()):
        print(file.name)
        #to numpy
        data = pd.read_csv(file, sep=";", header=None).values
        costs = torch.tensor(data[:, 2], dtype=torch.long)
        coordinates = data[:, 0:2]
        distances = torch.cdist(torch.tensor(coordinates, dtype=torch.float32), torch.tensor(coordinates, dtype=torch.float32))
        distances = torch.round(distances).long()
        #solve
        best_solution, best_solution_cost, n_iterations = ils(distances, costs, max_seconds=60)
        calculated_cost = torch.sum(distances[best_solution, best_solution.roll(-1)]) + torch.sum(costs[best_solution])
        print("Compare", calculated_cost, best_solution_cost)
        print("Iterations", n_iterations)
        print("Solution", best_solution)
        #print solution
        normalized_costs = (costs - costs.min())/ (costs.max() - costs.min()).tolist()
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=normalized_costs)
        #plot coordinates with color cost
        plt.plot(coordinates[best_solution, 0].tolist(), coordinates[best_solution, 1].tolist())
        plt.plot([coordinates[best_solution[0], 0], coordinates[best_solution[-1], 0]],
                 [coordinates[best_solution[0], 1], coordinates[best_solution[-1], 1]], 'ro-')
        # print node indices
        for i in range(len(best_solution)):
            plt.text(coordinates[best_solution[i], 0], coordinates[best_solution[i], 1], f"{i}_{best_solution[i]}")
        plt.title(f"Instance {file.name} with cost {best_solution_cost}")
        #show color scale
        plt.colorbar()
        plt.show()
        sleep(1)



if __name__ == "__main__":
    main()
