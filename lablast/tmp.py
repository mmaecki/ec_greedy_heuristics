def ils(costs, distances, max_time):
    move_data_to_gpu()
    population = generate_random_initial_population()
    population_costs = calc_cost_batched(population)
    best_solution = population[argmin(population_costs)]
    best_cost = min(population_costs)
    while there_is_time_left():
        best_inter_d, best_inter_mv = calc_inter_moves(population)
        best_intra_d, intra_mv = calc_intra_moves(population)
        best_inter_msk = (best_inter_d <= best_intra_d) & (best_inter_d < 0)
        best_intra_msk = (best_intra_d < best_inter_d) & (best_intra_d < 0)

        population[best_inter_msk] = do_inter_moves()
        population_costs[best_inter_msk] += best_inter_d[best_inter_msk]
        population[best_intra_msk] = do_intra_moves()
        population_costs[best_intra_msk] += best_intra_d[best_intra_msk]

        current_best_cost, best_idx = min(population_costs, dim=0)
        if current_best_cost < best_cost:
            best_solution = population[best_idx]
            best_cost = current_best_cost

        best_deltas = min(best_inter_d, best_intra_d)
        not_improved = best_deltas >= 0
        population[not_improved] = perturbe(population[not_improved])

    return best_solution, best_cost
