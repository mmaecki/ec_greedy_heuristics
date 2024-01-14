import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

n_nodes = 200
sol_size = 100
pop_size = 10000


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coordinates = torch.rand(n_nodes, 2, device=device)
    costs = torch.rand(n_nodes, device=device)
    distances = torch.cdist(coordinates, coordinates, p=2)

    #holder = vector from 0 to n_nodes
    holder = torch.arange(n_nodes, device=device)
    holder = holder.repeat((pop_size, 1))
    #sol_size random indeices from 0 to n_nodes

    def calc_cost(solution):
        dist_cost = torch.sum(distances[solution[:-1], solution[1:]]) + distances[solution[-1], solution[0]]
        return dist_cost + torch.sum(costs[solution])

    def calc_cost_batched(solutions):
        dist_cost = torch.sum(distances[solutions, solutions.roll(-1, dims=1)], dim=-1) + distances[solutions[:, -1], solutions[:, 0]]
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
        best_move_i = torch.zeros(solutions.shape[0], device=device)
        best_move_j = torch.zeros(solutions.shape[0], device=device)
        best_move_move = torch.zeros(solutions.shape[0], device=device)
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

        return best_move_i, best_move_j, best_move_move

    def intra_move(solutions):
        best_cost = calc_cost_batched(solutions)
        best_move_i = torch.zeros(solutions.shape[0], device=device)
        best_move_j = torch.zeros(solutions.shape[0], device=device)
        for i in range(sol_size):
            for j in range(sol_size):
                #reverse from i to j
                solutions[:, i:j] = solutions[:, i:j].flip(dims=(1,))
                new_cost = calc_cost_batched(solutions)
                #deapply move
                solutions[:, i:j] = solutions[:, i:j].flip(dims=(1,))
                best_move_i[new_cost < best_cost] = i
                best_move_j[new_cost < best_cost] = j
                best_cost[new_cost < best_cost] = new_cost[new_cost < best_cost]

        return best_move_i, best_move_j

    best_solution = torch.randperm(n_nodes, device=device)[:sol_size]
    best_solution_cost = calc_cost(best_solution)
    population = torch.rand((pop_size, n_nodes), device=device)
    for i in tqdm(range(1000000)):
        #repeat arrange 100 times
        new_solution_factory = torch.rand((pop_size, n_nodes), device=device)
        _, new_solutions = torch.topk(new_solution_factory, sol_size, dim=1, largest=False)
        new_solution_costs = calc_cost_batched(new_solutions)
        best_cost, best_idx = torch.min(new_solution_costs, dim=0)
        if best_cost < best_solution_cost:
            best_solution_cost = best_cost
            best_solution = new_solutions[best_idx]
            #plot solution
            plt.plot(coordinates[best_solution, 0].tolist(), coordinates[best_solution, 1].tolist())
            plt.title("Total cost: {}".format(best_solution_cost))
            plt.show()


