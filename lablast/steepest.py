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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coordinates = torch.rand(n_nodes, 2)
    costs = torch.rand(n_nodes, device=device)
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

        new_nodes = unvisited_nodes.unsqueeze(-1).expand(-1, -1, unvisited_nodes.shape[1]).transpose(1, 2)#shift_2d_to_3d(unvisited_nodes, unvisited_nodes.shape[1])
        new_prev_distances = distances[previous_nodes, new_nodes]
        new_next_distances = distances[new_nodes, next_nodes]
        new_distance_costs = new_prev_distances + new_next_distances
        new_node_costs = costs[new_nodes]
        new_costs = new_distance_costs + new_node_costs

        delta = new_costs - cur_costs

        # gt_delta = torch.zeros((pop_size, sol_size, unvisited_nodes.shape[1]), device=device)
        # besta_mova_i = []
        # besta_mova_j = []
        # besta_mova_move = []
        # besta_mova_delta = []
        # for pop_index in range(pop_size):
        #     besta_i = 0
        #     besta_j = 0
        #     besta_move = 0
        #     besta_delta = 999999
        #     for sol_index in range(sol_size):
        #         for unvisited_index in range(unvisited_nodes.shape[1]):
        #             cur_sol = solutions[pop_index].clone()
        #             # print("Changing", cur_sol[sol_index], "to", unvisited_nodes[pop_index, unvisited_index], "in", cur_sol)
        #             cur_sol[sol_index] = unvisited_nodes[pop_index, unvisited_index]
        #             gt_delta[pop_index, sol_index, unvisited_index] = calc_cost_batched(cur_sol.unsqueeze(0)) - calc_cost_batched(
        #                 solutions[pop_index].unsqueeze(0))[0]
        #             if gt_delta[pop_index, sol_index, unvisited_index] < besta_delta:
        #                 besta_i = sol_index
        #                 besta_j = unvisited_index
        #                 besta_move = unvisited_nodes[pop_index, unvisited_index]
        #                 besta_delta = gt_delta[pop_index, sol_index, unvisited_index]
        #     besta_mova_i.append(besta_i)
        #     besta_mova_j.append(besta_j)
        #     besta_mova_move.append(besta_move)
        #     besta_mova_delta.append(besta_delta)
        #
        # besta_mova_i = torch.tensor(besta_mova_i, device=device)
        # besta_mova_j = torch.tensor(besta_mova_j, device=device)
        # besta_mova_move = torch.tensor(besta_mova_move, device=device)
        # besta_mova_delta = torch.tensor(besta_mova_delta, device=device)


        #assert difference close to 0
        # assert torch.allclose(gt_delta, delta, atol=1e-3)

        # gt_delta8 = gt_delta[26]
        # delta8 = delta[26]

        best_move_i, best_move_j = min_2nd_3rd(delta)
        best_move_move = new_nodes[torch.arange(pop_size), best_move_i, best_move_j]
        best_delta = delta[torch.arange(pop_size), best_move_i, best_move_j]

        # assert torch.all(best_move_i == besta_mova_i)
        # assert torch.all(best_move_j == besta_mova_j)
        # assert torch.all(best_move_move == besta_mova_move)
        # assert torch.torch.allclose(best_delta, besta_mova_delta, 1e-3)


        # solutions_copy = solutions.clone()
        # mask_best = best_delta < 0
        # costs_before = calc_cost_batched(solutions_copy[mask_best])
        # solutions_copy[mask_best, best_move_i[mask_best]] = best_move_move[mask_best]
        # costs_after = calc_cost_batched(solutions_copy[mask_best])
        # deltas = costs_after - costs_before
        # best_deltas_masked = best_delta[mask_best]
        # assert torch.allclose(deltas, best_delta[mask_best], atol=1e-3)
        return best_delta, best_move_i, best_move_j, best_move_move


    def reverse_op(solutions, solutions_mask,  i_indices, j_indices):
        j_indices = j_indices
        lengths = j_indices - i_indices
        range_tensor_masked = range_tensor[solutions_mask]
        corrected_indices = i_indices.unsqueeze(1) + lengths.unsqueeze(1) - 1 - (range_tensor_masked - i_indices.unsqueeze(1))
        corrected_indices = torch.clamp(corrected_indices, 0, solutions.size(1) - 1)  # Clamping to avoid out-of-bounds

        mask = (range_tensor_masked >= i_indices.unsqueeze(1)) & (range_tensor_masked < j_indices.unsqueeze(1))

        solutions[mask] = torch.gather(solutions, 1, corrected_indices)[mask]
        return solutions


    def intra_move(solutions):
        best_cost = calc_cost_batched(solutions)
        best_move_i = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)
        best_move_j = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)

        before_cost = calc_cost_batched(solutions)

        for i in range(sol_size):
            for j in range(i+2, sol_size+1):#TODO maybe +2 i do not know
                #reverse from i to j
                solutions[:, i:j] = solutions[:, i:j].flip(dims=(1,))
                new_cost = calc_cost_batched(solutions)
                #deapply move
                solutions[:, i:j] = solutions[:, i:j].flip(dims=(1,))
                best_move_i[new_cost < best_cost] = i
                best_move_j[new_cost < best_cost] = j
                best_cost[new_cost < best_cost] = new_cost[new_cost < best_cost]

        return best_cost - before_cost,best_move_i, best_move_j

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
        #plot 1st solution
        #plot 1st solution
        #inter move
        best_inter_delta, best_inter_i, best_inter_j, best_inter_move = inter_move(population)
        #intra move
        best_intra_delta, best_intra_i, best_intra_j = intra_move(population)
        #apply best move
        best_inter_mask = (best_inter_delta <= best_intra_delta) & (best_inter_delta < 0)
        best_intra_mask = (best_intra_delta < best_inter_delta) & (best_intra_delta < 0)
        if not torch.any(best_inter_mask | best_intra_mask):
            print("no improvement")
            break
        #apply inter
        population[best_inter_mask, best_inter_i[best_inter_mask]] = best_inter_move[best_inter_mask]
        curret_cost = calc_cost_batched(population[best_inter_mask])
        real_delta = curret_cost - population_costs[best_inter_mask]
        calculated_delta = best_inter_delta[best_inter_mask]
        if not torch.all(torch.isclose(real_delta, calculated_delta, atol=0.01)):
            #print max difference
            max_diff = torch.max(torch.abs(real_delta - calculated_delta))
            index_diff = torch.argmax(torch.abs(real_delta - calculated_delta))
            print(max_diff, index_diff)
        population_costs[best_inter_mask] += best_inter_delta[best_inter_mask]
        #apply intra
        population[best_intra_mask] = reverse_op(population[best_intra_mask], best_intra_mask, best_intra_i[best_intra_mask], best_intra_j[best_intra_mask])
        population_costs[best_intra_mask] += best_intra_delta[best_intra_mask]


        # #find best solution
        for p in population:
            assert len(p) == sol_size, f"{len(p)}_{sol_size}"
            _, counts = p.unique(return_counts=True)
            assert torch.all(counts == 1)
        current_best_cost, best_idx = torch.min(population_costs, dim=0)
        if iteartion%1 == 0 and current_best_cost < best_solution_cost:
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


