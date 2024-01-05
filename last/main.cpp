#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <list>
#include <memory>
#include <map>
#include <coroutine>
#include <algorithm>
#include <random>
#include <cassert>
#include <deque>
#include <unordered_set>

using namespace std;

auto rng = std::default_random_engine(869468469);

struct Result
{
    int bestCost;
    int worstCost;
    int averageCost;
    vector<int> bestSolution;
    vector<int> worstSolution;

    Result(int bc, int wc, int ac, vector<int> bs, vector<int> ws)
        : bestCost(bc), worstCost(wc), averageCost(ac), bestSolution(bs), worstSolution(ws) {}
};

int calculate_cost(vector<int> solution, vector<vector<int>> distances, vector<int> costs)
{
    int cost = 0;
    for (int j = 0; j < solution.size() - 1; j++)
    {
        cost += distances[solution[j]][solution[j + 1]];
        cost += costs[solution[j]];
    }
    cost += distances[solution[solution.size() - 1]][solution[0]];
    return cost;
}

class Algo
{
public:
    vector<vector<int>> distances;
    vector<int> costs;
    string name;
    Algo(vector<vector<int>> distances, vector<int> costs,string name)
        : distances(distances), costs(costs), name(name) {}
    virtual Result solve() = 0;
    int calculate_cost(vector<int>& solution)
    {
        int cost = 0;
        for (int j = 0; j < solution.size() - 1; j++)
        {
            cost += distances[solution[j]][solution[j + 1]];
            cost += costs[solution[j]];
        }
        cost += distances[solution[solution.size() - 1]][solution[0]];
        return cost;
    }
    string get_name()
    {
        return this->name;
    }
};


template <typename T>
struct generator
{
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct promise_type
    {
        T value;
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        generator get_return_object() { return generator{handle_type::from_promise(*this)}; }
        void unhandled_exception() { std::terminate(); }
        std::suspend_always yield_value(T val)
        {
            value = val;
            return {};
        }
    };

    bool move_next() { return coro ? (coro.resume(), !coro.done()) : false; }
    T current_value() { return coro.promise().value; }

    generator(generator const &) = delete;
    generator(generator &&other) : coro(other.coro) { other.coro = {}; }
    ~generator()
    {
        if (coro)
            coro.destroy();
    }

private:
    generator(handle_type h) : coro(h) {}
    handle_type coro;
};

class LocalSearch : public Algo
{
public:
    vector<bool> visited;
    LocalSearch(vector<vector<int>> distances, vector<int> costs)
            : Algo(distances, costs, "LS")
    {
        visited = vector<bool>(distances.size());
    }


    int fixIndex(int index, int solutionSize)
    {
        if (index < 0)
        {
            return solutionSize + index;
        }
        if (index >= solutionSize)
        {
            return index - solutionSize;
        }
        return index;
    }

    Result solve()
    {
        return Result(0, 0, 0, vector<int>(), vector<int>());
    }

    void localSearch(shared_ptr<vector<int>> solution)
    {
        while (true)
        {
            auto neighbourhoodIterator2 = neighbourhoodGenerator(*solution);
            int bestDelta = INT32_MAX;
            vector<int> bestMove;
            while (neighbourhoodIterator2.move_next())
            {
                vector<int> move = neighbourhoodIterator2.current_value();
                int delta = calculateDelta(*solution, move);
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestMove = move;
                }
            }
            if (bestDelta >= 0)
            {
                break;
            }
            applyMove(solution, bestMove);
        }
    }


    int calculateDelta(vector<int>solution, vector<int> &move)
    {
        int delta;
        if (move.size() == 3)
        {
            // exchange nodes
            int i = move[0];
            int new_node = move[1];
            int old_node = solution[i];
            int oldCost = costs[old_node] + distances[old_node][solution[fixIndex(i + 1, solution.size())]] + distances[old_node][solution[fixIndex(i - 1 + solution.size(), solution.size())]];
            int newCost = costs[new_node] + distances[new_node][solution[fixIndex(i + 1, solution.size())]] + distances[new_node][solution[fixIndex(i - 1 + solution.size(), solution.size())]];
            delta = newCost - oldCost;
        }
        else if (move.size() == 4)
        {
            // edge exchnge
            int edge1_first = solution[move[0]];
            int edge1_second = solution[move[1]];
            int edge2_first = solution[move[2]];
            int edge2_second = solution[move[3]];
            int oldCost = distances[edge1_first][edge1_second] + distances[edge2_first][edge2_second];
            int newCost = distances[edge1_first][edge2_first] + distances[edge1_second][edge2_second];
            delta = newCost - oldCost;
        }
        else
        {
            throw runtime_error("Wrong size of best move");
        }
        return delta;
    }

    void applyMove(shared_ptr<vector<int>> solution, const vector<int> &move)
    { // modifies solution i think usuall  poinnter is enuogh
        if (move.size() == 3)
        {
            int i = move[0];
            int j = move[1];
            (*solution)[i] = j;
            visited[move[2]] = false;
            visited[j] = true;
        }
        else if (move.size() == 4)
        {
            int j = move[1];
            int k = move[2];
            reverse(solution->begin() + j, solution->begin() + k + 1);
        }
    }

    generator<vector<int>> neighbourhoodGenerator(vector<int> &currentSolution)
    {
        auto intraNeighbourhoodIterator = intraEdgesNeighbourhoodGenerator(currentSolution);
        while (intraNeighbourhoodIterator.move_next())
        {
            co_yield intraNeighbourhoodIterator.current_value();
        }
        auto interNeighbourhoodIterator = interNeighbourhoodGenerator(currentSolution);
        while (interNeighbourhoodIterator.move_next())
        {
            co_yield interNeighbourhoodIterator.current_value();
        }
    }

    generator<vector<int>> interNeighbourhoodGenerator(vector<int> &currentSolution)
    {
        vector<int> move = {0, 0};
        for (int i = 0; i < currentSolution.size(); i++)
        {
            int currentnode = currentSolution[i];
            for (int j = 0; j < distances.size(); j++)
            {
                if (!visited[j])
                {
                    co_yield makeInterMove(i, j, currentSolution[i]);
                }
            }
        }
    }

    vector<int> makeInterMove(int currentNodeId, int newNode, int currentNode)
    {
        return {currentNodeId, newNode, currentNode};
    }

    generator<vector<int>> intraEdgesNeighbourhoodGenerator(vector<int> &currentSolution)
    {
        vector<int> temp_vec = {0, 0, 0, 0};
        vector<int> move = vector<int>(temp_vec);
        for (int i = 0; i < currentSolution.size(); i++)
        {
            int node1 = currentSolution[i];
            int node1_next = currentSolution[fixIndex(i + 1, currentSolution.size())];
            for (int j = i + 2; j < currentSolution.size(); j++)
            {
                int node2 = currentSolution[j];
                int node2_next = currentSolution[fixIndex(j + 1, currentSolution.size())];
                co_yield makeIntraMove(i, i + 1, j, fixIndex(j + 1, currentSolution.size()));
            }
        }
    }

    vector<int> makeIntraMove(int edge1_first, int edge1_second, int edge2_first, int edge2_second)
    {
        return {edge1_first, edge1_second, edge2_first, edge2_second};
    }
};




class EvolutionaryAlgorithm : public Algo
{
public:
    int solutionSize;
    int populationSize;
    int maxIterations;
    int maxTime;
    int elitismSize;
    LocalSearch ls;
    bool doLocalSearch;
    map<int, shared_ptr<vector<int>>> population;

    EvolutionaryAlgorithm(vector<vector<int>> distances, vector<int> costs, string name, int population_size, int max_iterations, int max_time, int elitism_size)
        : Algo(distances, costs, name), populationSize(population_size), maxIterations(max_iterations), maxTime(max_time),  elitismSize(elitism_size), ls(distances, costs) {
            this->solutionSize = distances.size() / 2;
    }

    Result solve()
    {
        generate_initial_population();
        int iteration = 0;
        clock_t start, end;
        start = clock();
        while (iteration < maxIterations && (double(end - start) / double(CLOCKS_PER_SEC)) < maxTime) {
            iteration++;
            map<int, shared_ptr<vector<int>>> new_population;
            while (new_population.size() < populationSize - elitismSize) {
                pair<shared_ptr<vector<int>>, shared_ptr<vector<int>>> parents = select_parents();
                vector<shared_ptr<vector<int>>> children = crossover(parents.first, parents.second);
                for (auto child: children) {
                    if (doLocalSearch) {
                        ls.localSearch(child);
                    }
                    int cost = calculate_cost(*child);
                    if (new_population.find(cost) == new_population.end() and
                        population.find(cost) == population.end()) {
                        new_population[cost] = child;
                    }
                    if (new_population.size() == populationSize - elitismSize) {
                        break;
                    }
                }
            }
            while (population.size() > elitismSize) {
                population.erase(population.rbegin()->first);
            }
            for (auto it = new_population.begin(); it != new_population.end(); it++) {
                population[it->first] = it->second;
            }
            end = clock();
        }

        //best solution has lowest score
        int bestCost = population.begin()->first;
        vector<int> bestSolution = *population.begin()->second;
        return Result(bestCost, bestCost, bestCost, bestSolution, bestSolution);
    }

    void generate_initial_population() {
        while(population.size() < populationSize) {
            shared_ptr<vector<int>> solution = make_shared<vector<int>>(solutionSize);
            iota(solution->begin(), solution->end(), 0);
            shuffle(solution->begin(), solution->end(), rng);
            solution->resize(solutionSize);
            if(doLocalSearch){
                ls.localSearch(solution);
            }
            int cost = calculate_cost(*solution);
            if (population.find(cost) == population.end()) {
                population[cost] = solution;
            }
        }
    }

    pair<shared_ptr<vector<int>>, shared_ptr<vector<int>>> select_parents() {
        std::uniform_int_distribution<> dist(0, populationSize - 1);
        auto first = std::begin(population);
        std::advance(first, dist(rng));
        auto second = std::begin(population);
        do{
            second = std::begin(population);
            std::advance(second, dist(rng));
        } while(first == second);
        return make_pair(first->second, second->second);
    }


    vector<shared_ptr<vector<int>>> crossover(shared_ptr<vector<int>> first_parent, shared_ptr<vector<int>> second_parent) {
        //random crossover 1 or 2
        if(rand() % 2 == 0) {
            return crossover1(first_parent, second_parent);
        } else {
            return crossover2(first_parent, second_parent);
        }
    }

    //we do not need common edges
    vector<shared_ptr<vector<int>>> crossover1(shared_ptr<vector<int>> first_parent, shared_ptr<vector<int>> second_parent) {
        unordered_set<int> first_nodes;
        unordered_set<int> second_nodes;
        for (int i = 0; i < first_parent->size(); i++)
        {
            first_nodes.insert((*first_parent)[i]);
            second_nodes.insert((*second_parent)[i]);
        }
        unordered_set<int> common_nodes;
        for (int i = 0; i < first_parent->size(); i++)
        {
            if (second_nodes.find((*first_parent)[i]) != second_nodes.end())
            {
                common_nodes.insert((*first_parent)[i]);
            }
        }
        vector<int> firstChild(solutionSize);
        vector<int> secondChild(solutionSize);
        for(int i = 0; i < solutionSize; i++) {
            if(common_nodes.find((*first_parent)[i]) != common_nodes.end()) {
                firstChild[i] = (*first_parent)[i];
            }else{
                firstChild[i] = -1;
            }
            if(common_nodes.find((*second_parent)[i]) != common_nodes.end()) {
                secondChild[i] = (*second_parent)[i];
            }else{
                secondChild[i] = -1;
            }
        }
        vector<int> missingNodes;
        for(int i = 0; i < solutionSize; i++){
            //if i not in common nodes
            if(common_nodes.find(i) == common_nodes.end()) {
                missingNodes.push_back(i);
            }
        }
        vector<int> firstMissingNodes = missingNodes;
        vector<int> secondMissingNodes = missingNodes;
        shuffle(firstMissingNodes.begin(), firstMissingNodes.end(), rng);
        shuffle(secondMissingNodes.begin(), secondMissingNodes.end(), rng);
        for(int i = 0; i < solutionSize; i++) {
            if(firstChild[i] == -1) {
                firstChild[i] = firstMissingNodes.back();
                firstMissingNodes.pop_back();
            }
            if(secondChild[i] == -1) {
                secondChild[i] = secondMissingNodes.back();
                secondMissingNodes.pop_back();
            }
        }
        vector<shared_ptr<vector<int>>> children;
        children.push_back(make_shared<vector<int>>(firstChild));
        children.push_back(make_shared<vector<int>>(secondChild));
        return children;
    }


    //    Operator 2. We choose one of the parents as the starting solution. We remove from this
//            solution all edges and nodes that are not present in the other parent. The solution is
//            repaired using the heuristic method in the same way as in the LNS method. We also test the
//    version of the algorithm without local search after recombination (we still use local search
//    for the initial population).
    vector<shared_ptr<vector<int>>> crossover2(shared_ptr<vector<int>> parent1, shared_ptr<vector<int>> parent2, mt19937 &rng) {
        shared_ptr<vector<int>> child = make_shared<vector<int>>(*parent1); // Start with a copy of parent1

        // Create sets for easier search
        unordered_set<int> parent1Nodes(parent1->begin(), parent1->end());
        unordered_set<int> parent2Nodes(parent2->begin(), parent2->end());

        // Remove nodes from child that are not present in parent2
        child->erase(
                remove_if(child->begin(), child->end(), [&parent2Nodes](int node) {
                    return parent2Nodes.find(node) == parent2Nodes.end();
                }),
                child->end()
        );

        // Identify missing nodes
        unordered_set<int> missingNodes;
        for (int i = 0; i < solutionSize; ++i) { // Assuming solutionSize is the size of the parent solutions
            if (parent1Nodes.find(i) != parent1Nodes.end() && find(child->begin(), child->end(), i) == child->end()) {
                missingNodes.insert(i);
            }
        }

        // Add random missing nodes to the child until it has the same length as the parents
        while (child->size() < parent1->size()) {
            auto it = missingNodes.begin();
            advance(it, rng() % missingNodes.size()); // Random selection
            child->push_back(*it);
            missingNodes.erase(it); // Remove the added node from the set of missing nodes
        }

        // Repair the child using local search
        ls.localSearch(child);

        // Wrap in a vector and return
        vector<shared_ptr<vector<int>>> children;
        children.push_back(child);
        return children;
    }
};



enum ProblemInstance
{
    TSPA,
    TSPB,
    TSPC,
    TSPD
};

std::map<ProblemInstance, std::string> ProblemInstanceStrings = {
    {TSPA, "TSPA"},
    {TSPB, "TSPB"},
    {TSPC, "TSPC"},
    {TSPD, "TSPD"}};

// times
map<ProblemInstance, double> maxTimes = {
    {TSPA, 38.216},
    {TSPB, 40.5296},
    {TSPC, 43.9474},
    {TSPD, 45.5663}};




vector<vector<int>> read_file(string filename)
{
    vector<vector<int>> result;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        vector<int> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ';'))
        {
            row.push_back(stoi(cell));
        }
        result.push_back(row);
    }
    return result;
}

vector<vector<int>> calcDistances(vector<vector<int>> data)
{
    vector<vector<int>> distances;
    for (int i = 0; i < data.size(); i++)
    {
        vector<int> row;
        for (int j = 0; j < data.size(); j++)
        {
            int x = data[i][0] - data[j][0];
            int y = data[i][1] - data[j][1];
            // round to nearest int
            float distance = round(sqrt(x * x + y * y));
            row.push_back(distance);
        }
        distances.push_back(row);
    }
    return distances;
}
int N_TRIES = 20;

int main(){
    return 0;
}
//int main()
//{
//    string root_path = "../data/";
//    vector<ProblemInstance> problemInstances = {TSPC, TSPD};
//
//
//    for (auto problemInstance : problemInstances)
//    {
//        string file = root_path + ProblemInstanceStrings[problemInstance] + ".csv";
//        auto data = read_file(file);
//        auto distances = calcDistances(data);
//        vector<int> costs;
//        for (int i = 0; i < data.size(); i++)
//        {
//            costs.push_back(data[i][2]);
//        }
//        for (auto searchType : searchTypes)
//        {
//            for (auto initialSolutionType : initialSolutionTypes)
//            {
//                for (auto interNeighbourhoodType : interNeighbourhoodTypes)
//                {
//                    for (auto doLocalSearch : doLocalSearch)
//                    {
//                        cout << "Name: " << LSNS(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0, maxTimes[problemInstance], doLocalSearch).get_name() << endl;
//                        cout << "Problem instance: " << ProblemInstanceStrings[problemInstance] << endl;
//                        Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
//                        double averageTime = 0;
//                        double avg_iterations_number = 0;
//
//                        for (int i = 0; i < N_TRIES; i++)
//                        {
//                            cout << "Try: " << i << endl;
//                            LSNS ls = LSNS(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, -1, maxTimes[problemInstance], doLocalSearch);
//                            clock_t start, end;
//                            start = clock();
//                            Result res = ls.solve();
//                            end = clock();
//                            vector<int> solution = res.bestSolution;
//                            avg_iterations_number += res.averageCost; // iterations number
//                            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
//                            int cost = ls.calculate_cost(solution);
//                            if (cost < algoResult.bestCost)
//                            {
//                                algoResult.bestCost = cost;
//                                algoResult.bestSolution = solution;
//                            }
//                            if (cost > algoResult.worstCost)
//                            {
//                                algoResult.worstCost = cost;
//                                algoResult.worstSolution = solution;
//                            }
//                            algoResult.averageCost += cost;
//                            averageTime += time_taken;
//                        }
//                        avg_iterations_number /= N_TRIES;
//                        algoResult.averageCost /= N_TRIES;
//                        cout << "Best cost: " << algoResult.bestCost << endl;
//                        cout << "Worst cost: " << algoResult.worstCost << endl;
//                        cout << "Average cost: " << algoResult.averageCost << endl;
//                        averageTime /= N_TRIES;
//                        cout << "Average time: " << averageTime << endl;
//                        cout << "Best solution: ";
//                        for (int i = 0; i < algoResult.bestSolution.size(); i++)
//                        {
//                            cout << algoResult.bestSolution[i] << " ";
//                        }
//                        cout << endl;
//                        cout << "Average iterations number: " << avg_iterations_number << endl;
//                    }
//                }
//            }
//        }
//    }
//}