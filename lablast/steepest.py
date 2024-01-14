from time import sleep

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

n_nodes = 64
sol_size = n_nodes//2
pop_size = 1

#seeed
torch.manual_seed(738454387)


if __name__ == "__main__":
    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coordinates = torch.rand(n_nodes, 2, device=device)
    costs = torch.rand(n_nodes, device=device)
    distances = torch.cdist(coordinates, coordinates, p=2)
    print(costs)
    print(distances)

    #holder = vector from 0 to n_nodes
    holder = torch.arange(n_nodes, device=device)
    holder = holder.repeat((pop_size, 1))
    #range_tensor
    range_tensor = torch.arange(sol_size, device=device).repeat((pop_size, 1))

    def calc_cost(solution):
        dist_cost = torch.sum(distances[solution[:-1], solution[1:]])
        return dist_cost + torch.sum(costs[solution])

    def  calc_cost_batched(solutions):
        dist_cost = torch.sum(distances[solutions, solutions.roll(-1, dims=1)], dim=-1)
        c = costs[solutions]
        other_costs = torch.sum(c, dim=1)

        # Total cost
        total_cost = dist_cost + other_costs

        return total_cost

    def inter_move(solutions):
        combined = torch.cat((solutions, holder), dim=1)
        uniques, counts = combined.unique(dim=1, return_counts=True)
        difference = uniques[:, counts == 1]
        best_cost = calc_cost_batched(solutions)
        best_move_i = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)
        best_move_j = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)
        best_move_move = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)
        for i in range(sol_size):
            for j in range(difference.shape[1]):
                tmp = solutions[:, i].clone()
                substitution = difference[:, j]
                solutions[:, i] = substitution
                new_cost = calc_cost_batched(solutions)
                #deapply move
                solutions[:, i] = tmp
                best_move_i[new_cost < best_cost] = i
                best_move_j[new_cost < best_cost] = j
                best_move_move[new_cost < best_cost] = substitution[new_cost < best_cost]
                best_cost[new_cost < best_cost] = new_cost[new_cost < best_cost]

        return best_cost, best_move_i, best_move_j, best_move_move


    def reverse_op(solutions, solutions_mask,  i_indices, j_indices):
        # print("before", solutions)
        j_indices = j_indices
        lengths = j_indices - i_indices
        range_tensor_masked = range_tensor[solutions_mask]
        corrected_indices = i_indices.unsqueeze(1) + lengths.unsqueeze(1) - 1 - (range_tensor_masked - i_indices.unsqueeze(1))
        corrected_indices = torch.clamp(corrected_indices, 0, solutions.size(1) - 1)  # Clamping to avoid out-of-bounds

        mask = (range_tensor_masked >= i_indices.unsqueeze(1)) & (range_tensor_masked < j_indices.unsqueeze(1))

        solutions[mask] = torch.gather(solutions, 1, corrected_indices)[mask]
        # print("after", solutions)
        return solutions


    def intra_move(solutions):
        best_cost = calc_cost_batched(solutions)
        print("old cost", best_cost)
        best_move_i = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)
        best_move_j = torch.zeros(solutions.shape[0], device=device, dtype=torch.long)
        for i in range(sol_size):
            for j in range(i+2, sol_size+1):#TODO maybe +2 i do not know
                #reverse from i to j
                solutions[:, i:j] = solutions[:, i:j].flip(dims=(1,))
                new_cost = calc_cost_batched(solutions)
                # print("new_cost", i, j, solutions[0], new_cost)
                #deapply move
                solutions[:, i:j] = solutions[:, i:j].flip(dims=(1,))
                best_move_i[new_cost < best_cost] = i
                best_move_j[new_cost < best_cost] = j
                best_cost[new_cost < best_cost] = new_cost[new_cost < best_cost]

        return best_cost,best_move_i, best_move_j

    best_solution = torch.randperm(n_nodes, device=device)[:sol_size]
    best_solution_cost = calc_cost(best_solution)
    population_factory = torch.rand((pop_size, n_nodes), device=device)
    _, population = torch.topk(population_factory, sol_size, dim=1, largest=False)
    population = population
    population_costs = calc_cost_batched(population)
    for iteartion in tqdm(range(1000000)):
        #inter move
        best_inter_cost, best_inter_i, best_inter_j, best_inter_move = inter_move(population)
        #intra move
        best_intra_cost, best_intra_i, best_intra_j = intra_move(population)
        #apply best move
        best_inter_mask = (best_inter_cost <= best_intra_cost) & (best_inter_cost < population_costs)
        best_intra_mask = (best_intra_cost < best_inter_cost) & (best_intra_cost < population_costs)
        #apply inter
        population[best_inter_mask, best_inter_i[best_inter_mask]] = best_inter_move[best_inter_mask]
        population_costs[best_inter_mask] = best_inter_cost[best_inter_mask]
        #apply intra
        population[best_intra_mask] = reverse_op(population[best_intra_mask], best_intra_mask, best_intra_i[best_intra_mask], best_intra_j[best_intra_mask])
        population_costs[best_intra_mask] = best_intra_cost[best_intra_mask]


        # #find best solution
        for p in population:
            assert len(p) == sol_size, f"{len(p)}_{sol_size}"
            _, counts = p.unique(return_counts=True)
            assert torch.all(counts == 1)
        best_cost, best_idx = torch.min(population_costs, dim=0)
        if iteartion%10 == 0 and best_cost < best_solution_cost:
            best_solution_cost = best_cost
            best_solution = population[best_idx]
            #plot solution
            plt.plot(coordinates[best_solution, 0].tolist(), coordinates[best_solution, 1].tolist())
            plt.plot([coordinates[best_solution[0], 0], coordinates[best_solution[-1], 0]],
                     [coordinates[best_solution[0], 1], coordinates[best_solution[-1], 1]], 'ro-')
            #print node indices
            for i in range(len(best_solution)):
                plt.text(coordinates[best_solution[i], 0], coordinates[best_solution[i], 1], f"{i}_{best_solution[i]}")
            plt.title("Total cost: {}".format(best_solution_cost))
            plt.show()
            sleep(1)


