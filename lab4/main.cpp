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
#include <typeinfo>

using namespace std;


auto rng = std::default_random_engine(869468469);

struct Result{
    int bestCost;
    int worstCost;
    int averageCost;
    vector<int> bestSolution;
    vector<int> worstSolution;

    Result(int bc, int wc, int ac, vector<int> bs, vector<int> ws)
        : bestCost(bc), worstCost(wc), averageCost(ac), bestSolution(bs), worstSolution(ws) {}
};


class Algo {
public:
    vector<vector<int>> distances;
    vector<int> costs;
    int starting_node;
    string name;
    Algo(vector<vector<int>> distances, vector<int> costs, int i, string name)
        : distances(distances), costs(costs), starting_node(i), name(name) {}
    virtual Result solve() =0;
    int calculate_cost(vector<int> solution, vector<vector<int>> distances, vector<int> costs){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += distances[solution[j]][solution[j+1]];
        }
        cost+=distances[solution[solution.size()-1]][solution[0]];
        for(int j=0;j<solution.size();j++){
            cost+=costs[solution[j]];
        }
        return cost;
    }
    string get_name(){
        return this->name;
    }
};

class RandomSearch : public Algo {
public:
    RandomSearch(vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "RandomSearch") {}
    
    Result solve() {
        vector<int> worstSolution;
        int solution_size = this->distances.size()/2;
        vector<int> current_solution = vector<int>(solution_size);
        vector<int> visited(this->distances.size());

        for(int j=0; j<solution_size; j++){
            if (j==0){
                current_solution[j] = this->starting_node;
                visited[this->starting_node] = true;
                continue;
            }
            int next = rand() % this->distances.size();
            while(visited[next])next = rand() % this->distances.size();
            current_solution[j] = next;
            visited[next]=true;
        }        
        return Result(0, 0, 0, current_solution, worstSolution);
    }
};

template <typename T>
struct generator {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;
    
    struct promise_type {
        T value;
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        generator get_return_object() { return generator{handle_type::from_promise(*this)}; }
        void unhandled_exception() { std::terminate(); }
        std::suspend_always yield_value(T val) {
            value = val;
            return {};
        }
    };

    bool move_next() { return coro ? (coro.resume(), !coro.done()) : false; }
    T current_value() { return coro.promise().value; }

    generator(generator const&) = delete;
    generator(generator&& other) : coro(other.coro) { other.coro = {}; }
    ~generator() { if (coro) coro.destroy(); }

private:
    generator(handle_type h) : coro(h) {}
    handle_type coro;
};

enum SearchType { greedy, steepest };
enum InitialSolutionType {randomAlg, GC, G2Rw};
enum NeighbourhoodType {intra, inter};
enum InterNeighbourhoodType {twoNode, twoEdges};
enum ProblemInstance {TSPA, TSPB, TSPC, TSPD};

std::map<SearchType, std::string> SearchTypeStrings = {
    {greedy, "greedy"},
    {steepest, "steepest"}
};

std::map<InitialSolutionType, std::string> InitialSolutionTypeStrings = {
    {randomAlg, "random"},
};

std::map<NeighbourhoodType, std::string> NeighbourhoodTypeStrings = {
    {intra, "intra"},
    {inter, "inter"}
};

std::map<InterNeighbourhoodType, std::string> InterNeighbourhoodTypeStrings = {
    {twoNode, "twoNode"},
    {twoEdges, "twoEdges"}
};

std::map<ProblemInstance, std::string> ProblemInstanceStrings = {
    {TSPA, "TSPA"},
    {TSPB, "TSPB"},
    {TSPC, "TSPC"},
    {TSPD, "TSPD"}
};

std::map<ProblemInstance, InitialSolutionType> BestInitialForInstance = {
    {TSPA, G2Rw},
    {TSPB, G2Rw},
    {TSPC, GC},
    {TSPD, G2Rw}
};


class LocalSearch: public Algo {
public:
    SearchType searchType;
    InitialSolutionType initialSolutionType;
    InterNeighbourhoodType intraNeighbourhoodType;
    vector<bool> visited;
    int nPoints;
    LocalSearch(SearchType searchType, InitialSolutionType initialSolutionType, InterNeighbourhoodType intraNeighbourhoodType, vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "LocalSearch"), searchType(searchType), initialSolutionType(initialSolutionType), intraNeighbourhoodType(intraNeighbourhoodType) {
            this->name += "_" + SearchTypeStrings[searchType];
            this->name += "_" + InitialSolutionTypeStrings[initialSolutionType];
            this->name += "_" + InterNeighbourhoodTypeStrings[intraNeighbourhoodType];
            visited = vector<bool>(distances.size());
            nPoints = distances.size();
        }

    int calculate_cost(const vector<int>& solution){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += this->distances[solution[j]][solution[j+1]];
            cost+=this->costs[solution[j]];
        }
        cost+=this->distances[solution[solution.size()-1]][solution[0]];
        return cost;
    }

    Result solve() {
        vector<int> solution = getInitialSolution(this->initialSolutionType, starting_node);
        int solutionCost = this->calculate_cost(solution);
        for(int i=0; i<visited.size(); i++){
                visited[i] = false;
        }
        for(int i=0; i<solution.size(); i++){
            visited[solution[i]] = true;
        }
        while(true){
            //update visited
            for(int i=0; i<visited.size(); i++){
                visited[i] = false;
            }
            for(int i=0; i<solution.size(); i++){
                visited[solution[i]] = true;
            }
            
            shared_ptr<vector<int>> newSolution = search(solution, solutionCost);
            int newSolutionCost = calculate_cost(*newSolution);
            for (int i = 0; i < newSolution->size(); i++) {
                cout << (*newSolution)[i] << " ";
            }
            cout << endl;
            // cout << "new solution cost: " << newSolutionCost << endl;
            /////////////////////////////////////////////////////////////////////////////////////////
            //caclulate number of shared numbers in solution
            int shared = 0;
            bool temp_visited[nPoints];
            for(int i=0; i<nPoints; i++){
                temp_visited[i] = false;
            }
            for(int i=0; i<newSolution->size(); i++){
                if(!temp_visited[(*newSolution)[i]]){
                    shared++;
                    temp_visited[(*newSolution)[i]] = true;
                }
            }
            if(shared != distances.size()/2){
                cout << "Solution is not valid" << endl;
                cout << "Solution cost: " << newSolutionCost << endl;
                cout << "Solution size: " << newSolution->size() << endl;
                cout << "shared: " << shared << endl;
                cout << "Solution: ";
                for(int i=0; i<newSolution->size(); i++){
                    cout << (*newSolution)[i] << " ";
                }
                cout << endl;
                throw runtime_error("Solution is not valid");
            }
            /////////////////////////////////////////////////////////////////////////////////////////
            if(newSolutionCost == solutionCost) break;
            solution = *newSolution;

            solutionCost = newSolutionCost;
        }
        return Result(solutionCost, solutionCost, solutionCost, solution, solution);
    }

    vector<int> getInitialSolution(InitialSolutionType ist, int i){
        if(ist == randomAlg){
            RandomSearch rs = RandomSearch(distances, costs, i);
            return rs.solve().bestSolution;
        }
        // if (ist == GC)
        // {
        //     GreedyCycle gc = GreedyCycle(distances, costs, i);
        //     return gc.solve().bestSolution;

        // }
        // if (ist == G2Rw)
        // {
        //     Greedy2RegretWieghted g2rw = Greedy2RegretWieghted(distances, costs, i);
        //     return g2rw.solve().bestSolution;
        // }
    }

    shared_ptr<vector<int>> search(vector<int>& currentSolution, int currentSolutionCost){
        return steepestSearch(currentSolution, currentSolutionCost);
    }

    shared_ptr<vector<int>> steepestSearch(vector<int>& currentSolution, int currentSolutionCost){
        auto neigbourhoodIterator = neighbourhoodGenerator(currentSolution);
        int bestDelta = INT32_MAX;
        int delta;
        shared_ptr<vector<int>> bestMove;
        shared_ptr<vector<int>> move;
        shared_ptr<vector<int>> bestNeighbour = make_shared<vector<int>>(currentSolution);
        while(neigbourhoodIterator.move_next()){
            move = neigbourhoodIterator.current_value();
            if(move->size() == 2){
                // exchange nodes
                int i = (*move)[0];
                int j = (*move)[1];
                int tmp = currentSolution[i];
                delta = distances[currentSolution[i == 0 ? currentSolution.size() - 1 : i - 1]][j] + costs[j] + distances[j][currentSolution[i == currentSolution.size() - 1 ? 0 : i + 1]] - distances[currentSolution[i == 0 ? currentSolution.size() - 1 : i - 1]][tmp] - costs[tmp] - distances[tmp][currentSolution[i == currentSolution.size() - 1 ? 0 : i + 1]];
            }
            else if(move->size() == 4){
                // edge exchnge
                delta = distances[(*move)[1]][(*move)[2]] + distances[(*move)[0]][(*move)[3]] - distances[(*move)[0]][(*move)[1]] - distances[(*move)[2]][(*move)[3]];
            }
            // find the best move
            if(delta < bestDelta){
                bestDelta = delta;
                bestMove = move;
            }
        }

        // make a move - create the neighbour
        if(move->size() == 2){
            // node exchange
            int i = (*move)[0];
            int j = (*move)[1];
            (*bestNeighbour)[i] = j;
        }
        else if(move->size() == 4){
            // edge exchange
            int j = (*move)[1];
            int k = (*move)[2];
            reverse(bestNeighbour->begin() + j, bestNeighbour->begin() + k + 1);
        }
        else{
            throw runtime_error("aaaaaaaaaaa");
        }
        return bestNeighbour;
    }



    generator<shared_ptr<vector<int>>> neighbourhoodGenerator(vector<int>& currentSolution){
        vector<NeighbourhoodType> nTypeOrder = {intra, inter};
        shuffle(nTypeOrder.begin(), nTypeOrder.end(),rng);
        for(auto nType: nTypeOrder){
            if(nType == intra){
                auto intraNeighbourhoodIterator = intraNeighbourhoodGenerator(currentSolution);
                while(intraNeighbourhoodIterator.move_next()){
                    co_yield intraNeighbourhoodIterator.current_value();
                }
            } else {
                auto interNeighbourhoodIterator = interNeighbourhoodGenerator(currentSolution);
                while(interNeighbourhoodIterator.move_next()){
                    co_yield interNeighbourhoodIterator.current_value();
                }
            }
        }

    }

    generator<shared_ptr<vector<int>>> interNeighbourhoodGenerator(vector<int>& currentSolution){
        // shared_ptr<vector<pair<int, int>>> moves;
        // shared_ptr<std::vector<int>> neighbour = make_shared<std::vector<int>>(currentSolution);
        vector<int> temp_vec = {0, 0};
        shared_ptr<vector<int>> move = make_shared<vector<int>>(temp_vec);

        for (int i = 0; i < currentSolution.size(); i++) {
            for (int j = 0; j < costs.size(); j++) {
                if (!visited[j]) {
                    (*move)[0] = i;
                    (*move)[1] = j;
                    co_yield move;
                    // moves.push_back({i, j});
                }
            }
        }

        // std::shuffle(std::begin(moves), std::end(moves), rng);
        // for (const auto& [i, j] : moves) {
        //     auto tmp = (*neighbour)[i];
        //     (*neighbour)[i] = j;
        //     returnPair.first = neighbour;
        //     returnPair.second = distances[(*neighbour)[i == 0 ? neighbour->size() - 1 : i - 1]][j] + costs[j] + distances[j][(*neighbour)[i == neighbour->size() - 1 ? 0 : i + 1]] - distances[(*neighbour)[i == 0 ? neighbour->size() - 1 : i - 1]][tmp] - costs[tmp] - distances[tmp][(*neighbour)[i == neighbour->size() - 1 ? 0 : i + 1]];
        //     // co_yield 1;
        //     co_yield returnPair;
        //     // co_yield pair<shared_ptr<vector<int>>, int>(make_pair(neighbour, delta));
        //     // co_yield pair<shared_ptr<vector<int>>, int>(neighbour, delta);
        //     (*neighbour)[i] = tmp; //reversing change to save time copying memory
        // }
    }

    generator<shared_ptr<vector<int>>> intraNeighbourhoodGenerator(vector<int>& currentSolution){
        return intraEdgesNeighbourhoodGenerator(currentSolution);
    }

    generator<shared_ptr<vector<int>>> intraEdgesNeighbourhoodGenerator(vector<int>& currentSolution){
        // shared_ptr<std::vector<int>> neighbour = make_shared<std::vector<int>>(currentSolution);
        // vector<pair<pair<int, int>, pair<int, int>>> moves;
        vector<int> temp_vec = {0, 0, 0, 0};
        shared_ptr<vector<int>> move = make_shared<vector<int>>(temp_vec);
        for (int i = 0; i < currentSolution.size(); i++) {
            for (int j = i + 2; j < currentSolution.size(); j++) {
                (*move)[0] = i;
                (*move)[1] = i+1;
                (*move)[2] = j;
                (*move)[3] = (j == currentSolution.size() - 1 ? 0 : j + 1);
                co_yield move;
                // moves.push_back({{i, i + 1}, {j, j == currentSolution.size() - 1 ? 0 : j + 1}});
            }
        }
        // std::shuffle(std::begin(moves), std::end(moves), rng);
        // for (const auto& [edge1, edge2] : moves) {
        //     if (edge1.second < edge2.second) {
        //         reverse(neighbour->begin() + edge1.second, neighbour->begin() + edge2.first + 1);
        //         returnPair.first = neighbour;
        //         returnPair.second = distances[edge1.second][edge2.first] + distances[edge1.first][edge2.second] - distances[edge1.first][edge1.second] - distances[edge2.first][edge2.second];
        //         co_yield returnPair;
        //         // co_yield pair<shared_ptr<vector<int>>, int>(make_pair(neighbour, delta));
        //         reverse(neighbour->begin() + edge1.second, neighbour->begin() + edge2.first + 1);
        //         // to save time copying memory
        //     }
        //     else if(edge2.second != 0){
        //         throw runtime_error("Incorrect edge indices: "  + to_string(edge1.second) + " " + to_string(edge2.second));
        //     }
        // }
    }
};

vector<vector<int>> read_file(string filename) {
    vector<vector<int>> result;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ';')) {
            row.push_back(stoi(cell));
        }
        result.push_back(row);
    }
    return result;
}


vector<vector<int>> calcDistances(vector<vector<int>> data){
    vector<vector<int>> distances;
    for (int i = 0; i < data.size(); i++){
        vector<int> row;
        for (int j = 0; j < data.size(); j++){
            int x = data[i][0] - data[j][0];
            int y = data[i][1] - data[j][1];
            //round to nearest int
            float distance = round(sqrt(x*x + y*y));
            row.push_back(distance);
        }
        distances.push_back(row);
    }
    return distances;
}


void write_solution_to_file(vector<int> sol, string algo_name, string data_name){
        cout << "Writing to: " << algo_name + "_"+ data_name + ".csv" << endl;
        string filename = "results/" + algo_name + "_"+ data_name + ".csv";
        ofstream file;
        file.open(filename);
        for(int i=0; i<sol.size(); i++){
            file << sol[i] << endl;
        }
        file.close();
    }


int main(){

    string root_path = "../data/";
    vector<ProblemInstance> problemInstances = {TSPA, TSPB, TSPC, TSPD};
    vector<SearchType> searchTypes = {steepest};
    vector<InitialSolutionType> initialSolutionTypes = {randomAlg};
    vector<InterNeighbourhoodType> interNeighbourhoodTypes = {twoEdges};


    for(auto problemInstance: problemInstances){
        string file = root_path + ProblemInstanceStrings[problemInstance] + ".csv";
        auto data = read_file(file);
        auto distances = calcDistances(data);
        vector<int> costs;
        for(int i=0; i< data.size(); i++){
            costs.push_back(data[i][2]);
        }
        for(auto searchType: searchTypes){
            for(auto initialSolutionType: initialSolutionTypes){
                for(auto interNeighbourhoodType: interNeighbourhoodTypes){
                    cout << "Problem instance: " << ProblemInstanceStrings[problemInstance] << endl;
                    cout << "Search type: " << SearchTypeStrings[searchType] << endl;
                    cout << "Initial solution type: " << InitialSolutionTypeStrings[initialSolutionType] << endl;
                    cout << "Inter neighbourhood type: " << InterNeighbourhoodTypeStrings[interNeighbourhoodType] << endl;
                    Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
                    double averageTime = 0;
                    for(int i=0; i<distances.size(); i++){
                        cout << "Iteration: " << i << endl;
                        LocalSearch ls = LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, i);
                        clock_t start, end;
                        start = clock();
                        vector<int> solution = ls.solve().bestSolution;
                        end = clock();
                        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
                        int cost = ls.calculate_cost(solution);
                        if(cost < algoResult.bestCost){
                            algoResult.bestCost = cost;
                            algoResult.bestSolution = solution;
                        }
                        if(cost > algoResult.worstCost){
                            algoResult.worstCost = cost;
                            algoResult.worstSolution = solution;
                        }
                        algoResult.averageCost += cost;
                        // cout << "Time taken: " << time_taken << endl;
                        averageTime += time_taken;
                    }
                    algoResult.averageCost /= distances.size();
                    cout << "Best cost: " << algoResult.bestCost << endl;
                    cout << "Worst cost: " << algoResult.worstCost << endl;
                    cout << "Average cost: " << algoResult.averageCost << endl;
                    averageTime /= distances.size();
                    cout << "Average time: " << averageTime << endl;
                    cout << "Best solution: ";
                    for(int i=0; i<algoResult.bestSolution.size(); i++){
                        cout << algoResult.bestSolution[i] << " ";
                    }
                    cout << endl;
                    LocalSearch ls = LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0);
                    write_solution_to_file(algoResult.bestSolution, ls.get_name(), ProblemInstanceStrings[problemInstance]);
                }
            }
        }

    }

}