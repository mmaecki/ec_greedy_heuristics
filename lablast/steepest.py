from time import sleep

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import shift_2d_to_3d, min_2nd_3rd

n_nodes = 200
sol_size = n_nodes//2
pop_size = 1000

#seeed
torch.manual_seed(738454387)


def perturb_swap():
    ...


def perturb_revers_reverse():
    ...


def perturb_exchange():
    ...

def perturb_shuffle():
    ...

def perturbe():
    ...


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coordinates = torch.rand(n_nodes, 2)
    costs = torch.ones(n_nodes, device=device)#torch.rand(n_nodes, device=device)
    distances = torch.cdist(coordinates, coordinates, p=2).to(device)
    #holder = vector from 0 to n_nodes
    holder = torch.arange(n_nodes, device=device)
    holder = holder.repeat((pop_size, 1))
    bool_marker = torch.zeros(pop_size, n_nodes, device=device, dtype=torch.bool)
    #range_tensor
    range_tensor = torch.arange(sol_size, device=device).repeat((pop_size, 1))

    def calc_cost(solution):
        dist_cost = torch.sum(distances[solution, solution.roll(-1)])
        return dist_cost + torch.sum(costs[solution])

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
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    plt.plot(coordinates[population[0].cpu().numpy()][:, 0], coordinates[population[0].cpu().numpy()][:, 1])
    for i in range(len(coordinates)):
        plt.text(coordinates[i, 0], coordinates[i, 1], f"{i}")
    plt.pause(0.1)
    plt.show()
    for iteartion in tqdm(range(1000000)):
        #inter move
        best_inter_delta, best_inter_i, best_inter_j, best_inter_move = inter_move(population)
        #intra move
        best_intra_delta, best_intra_i, best_intra_j = intra_move(population)
        #apply best move
        best_inter_mask = (best_inter_delta <= best_intra_delta) & (best_inter_delta < 0)
        best_intra_mask = (best_intra_delta < best_inter_delta) & (best_intra_delta < 0)
        best_delta = torch.min(best_inter_delta, best_intra_delta)
        if not torch.any(best_inter_mask | best_intra_mask):
            print("no improvement")
            break
        #apply inter
        population[best_inter_mask, best_inter_i[best_inter_mask]] = best_inter_move[best_inter_mask]
        population_costs[best_inter_mask] += best_inter_delta[best_inter_mask]
        #apply intra
        population[best_intra_mask] = reverse_op(population[best_intra_mask], best_intra_mask, best_intra_i[best_intra_mask], best_intra_j[best_intra_mask])
        population_costs[best_intra_mask] += best_intra_delta[best_intra_mask]
        # #find best solution
        current_best_cost, best_idx = torch.min(population_costs, dim=0)
        if False and iteartion%100 == 0 and current_best_cost < best_solution_cost:
            best_solution_cost = current_best_cost
            best_solution = population[best_idx].detach().cpu()
            #plot solution
            plt.scatter(coordinates[:, 0], coordinates[:, 1])
            plt.plot(coordinates[best_solution, 0].tolist(), coordinates[best_solution, 1].tolist())
            plt.plot([coordinates[best_solution[0], 0], coordinates[best_solution[-1], 0]],
                     [coordinates[best_solution[0], 1], coordinates[best_solution[-1], 1]], 'ro-')
            #print node indices
            for i in range(len(best_solution)):
                plt.text(coordinates[best_solution[i], 0], coordinates[best_solution[i], 1], f"{i}_{best_solution[i]}")
            plt.title("IDX: {} Total cost: {}".format(best_idx, best_solution_cost))
            plt.show()
            sleep(1)


