import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


RESULTS_DIR = Path('./results')
INSTANCES = ["TSP"+c for c in ["A", "B", "C", "D"]]

def read_data():
    """Reads the data from the file and returns a pandas dataframe."""
    dfs = {}
    for instance in INSTANCES:
        file = RESULTS_DIR/f"{instance}.json"
        with open(file) as f:
            data = json.load(f)['results']
            df = pd.DataFrame(data)
            dfs[instance] = df
    return dfs



def node_smililarity(sol1, sol2):
    set1 = set(sol1)
    set2 = set(sol2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def edge_similarity(sol1, sol2):
    set1 = {frozenset(edge) for edge in zip(sol1, sol1[1:] + [sol1[0]])}
    set2 = {frozenset(edge) for edge in zip(sol2, sol2[1:] + [sol2[0]])}
    return len(set1.intersection(set2)) / len(set1.union(set2))


SMILIARITY_FUNCTIONS = {
    "node": node_smililarity,
    "edge": edge_similarity
}

def calc_for_instance(instance, df, sim_func):
    #select row with lowest cost
    best = df.loc[df['cost'].idxmin()]
    best_solution = best['bestSolution']
    similarities_to_best = []
    for solution in df['bestSolution']:
        similarities_to_best.append(sim_func(best_solution, solution))
    #similarity to all other_solutions
    similarities_to_others = []
    for sol1 in df['bestSolution']:
        similarities = []
        for sol2 in df['bestSolution']:
            similarities.append(sim_func(sol1, sol2))
        similarities_to_others.append(similarities)

    return {
        "instance": instance,
        "costs": np.array(df['cost'].tolist()),
        "similarities_to_best": np.array(similarities_to_best),
        "mean_similarity_to_best": np.mean(similarities_to_best),
        "std_similarity_to_best": np.std(similarities_to_best),
        "similarities_to_others": np.mean(similarities_to_others, axis=1),
        "mean_similarity_to_others": np.mean(similarities_to_others),
        "std_similarity_to_others": np.std(similarities_to_others),
    }
def main():
    dfs = read_data()
    print("data red")
    for sim_func_name, sim_func in SMILIARITY_FUNCTIONS.items():
        print(sim_func_name)
        with ProcessPoolExecutor(max_workers=6) as executor:
            keys = dfs.keys()
            values = [dfs[key] for key in keys]
            results = list(executor.map(calc_for_instance, keys, values, [sim_func]*len(keys)))
        nrows = len(results)
        fig, axs = plt.subplots(nrows, 2, figsize=(9, 12))
        for row, result in enumerate(results):
            idx = np.argsort(result["costs"])
            correlation_coeff = np.corrcoef(result["costs"][idx], result["similarities_to_best"][idx])
            axs[row, 0].scatter(result["costs"][idx], result["similarities_to_best"][idx])
            axs[row, 0].set_xlabel("cost")
            axs[row, 0].set_ylabel("similarity to best")
            axs[row, 0].set_title(f"Sim to Best {result['instance']} - {sim_func_name}\ncorrelation coeff: {correlation_coeff[0,1]}")
            idx = np.argsort(result["costs"])
            correlation_coeff = np.corrcoef(result["costs"][idx], result["similarities_to_others"][idx])
            axs[row, 1].scatter(result["costs"][idx], result["similarities_to_others"][idx])
            axs[row, 1].set_xlabel("cost")
            axs[row, 1].set_ylabel("similarity to others")
            axs[row, 1].set_title(f"Sim to Others {result['instance']} - {sim_func_name}\ncorrelation coeff: {correlation_coeff[0,1]}")
        plt.subplots_adjust(hspace=0.5, wspace=0.5) # Adjust this value as needed
        plt.savefig(RESULTS_DIR/f"sim_to_best_and_others_{sim_func_name}.png")
        plt.close()







if __name__ == "__main__":
    main()
            