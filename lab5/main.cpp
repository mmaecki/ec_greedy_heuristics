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
#include<unistd.h> 
#include <set>

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



enum MoveEvaluationResult {doMove, removeMove, skipMove};

class LocalSearch: public Algo {
public:
    SearchType searchType;
    InitialSolutionType initialSolutionType;
    InterNeighbourhoodType intraNeighbourhoodType;
    vector<bool> visited;
    int nPoints;
    LocalSearch(SearchType searchType, InitialSolutionType initialSolutionType, InterNeighbourhoodType intraNeighbourhoodType, vector<vector<int>> distances, vector<int> costs, int i)
        : Algo(distances, costs, i, "LS"), searchType(searchType), initialSolutionType(initialSolutionType), intraNeighbourhoodType(intraNeighbourhoodType) {
            this->name += "_" + SearchTypeStrings[searchType];
            this->name += "_" + InitialSolutionTypeStrings[initialSolutionType];
            this->name += "_" + InterNeighbourhoodTypeStrings[intraNeighbourhoodType];
            this->name += "Candidate";
            visited = vector<bool>(distances.size());
            nPoints = distances.size();
        }

    int calculate_cost(const vector<int>& solution){
        int cost = 0;
        for(int j=0; j<solution.size()-1; j++){
            cost += this->distances[solution[j]][solution[j+1]];
        }
        cost+=this->distances[solution[solution.size()-1]][solution[0]];
        for(int j=0;j<solution.size();j++){
            cost+=this->costs[solution[j]];
        }
        return cost;
    }

    int fixIndex(int index, int solutionSize){
        if(index < 0){
            return solutionSize + index;
        }
        if(index >= solutionSize){
            return index - solutionSize;
        }
        return index;
    }

    
    Result solve() {
        vector<int> solution = getInitialSolution(this->initialSolutionType, starting_node);
        for(int i=0; i<visited.size(); i++){
                visited[i] = false;
        }
        for(int i=0; i<solution.size(); i++){
            visited[solution[i]] = true;
        }
        multiset<pair<int, vector<int>>> LM;
        auto neighbourhoodIterator = neighbourhoodGenerator(solution);
        while(neighbourhoodIterator.move_next()){
            vector<int> move = neighbourhoodIterator.current_value();
            int delta = calculateDelta(solution, move);
            if(delta < 0){
                LM.insert(make_pair(delta, move));
            }
        }

        while(LM.size() > 0){
            auto lmIt = LM.begin();
            while(lmIt != LM.end()){
                auto moveEvaluationResult = evaluateMove(solution, lmIt->second);
                if(moveEvaluationResult == skipMove){
                    lmIt++;
                }else if(moveEvaluationResult == removeMove){
                    lmIt = LM.erase(lmIt);
                }else{
                    break;
                }
            }
            if(lmIt == LM.end()){
                break;
            }
            vector<int> beforeSolution = solution;
            int cost_before = calculate_cost(solution);
            applyMove(&solution, lmIt->second);
            int cost_after = calculate_cost(solution);
            if(cost_after-cost_before != lmIt->first){
                cout << "Wrong delta calculation" << endl;
            }
            auto newMoves = generateNewMoves(solution, lmIt->second);
            for(auto newMove: newMoves){
                int delta = calculateDelta(solution, newMove);
                if(delta < 0){
                    LM.insert(make_pair(delta, newMove));
                }
            }
            if(!validateSolution(solution)){
                cout<< "Solution not valid" << endl;
            }
            
        }
        return Result(calculate_cost(solution), 0, 0, solution, vector<int>());
    }

    bool validateSolution(vector<int>& solution){
        bool temp_visited[distances.size()];
        for(int i=0; i<distances.size(); i++){
            temp_visited[i] = false;
        }
        for(int i=0; i<solution.size(); i++){
            if(temp_visited[solution[i]]){
                return false;
            }
            temp_visited[solution[i]] = true;
        }
        return true;
    }

    vector<int> getInitialSolution(InitialSolutionType ist, int i){
        if(ist == randomAlg){
            RandomSearch rs = RandomSearch(distances, costs, i);
            return rs.solve().bestSolution;
        }
    }

    int calculateDelta(vector<int>& solution, vector<int>& move){
        int delta;
        if(move.size() == 5){
            // exchange nodes
            int i = move[0];
            int new_node = move[1];
            int old_node = solution[i];
            int oldCost = costs[old_node] + distances[old_node][solution[(i+1)%solution.size()]] + distances[old_node][solution[(i-1+solution.size())%solution.size()]];
            int newCost = costs[new_node] + distances[new_node][solution[(i+1)%solution.size()]] + distances[new_node][solution[(i-1+solution.size())%solution.size()]];
            delta = newCost - oldCost;
        }
        else if(move.size() == 8){
            // edge exchnge
            int edge1_first = solution[move[0]];
            int edge1_second = solution[move[1]];
            int edge2_first = solution[move[2]];
            int edge2_second = solution[move[3]];
            int oldCost = distances[edge1_first][edge1_second] + distances[edge2_first][edge2_second];
            int newCost = distances[edge1_first][edge2_first] + distances[edge1_second][edge2_second];
            delta = newCost - oldCost;
        }
        else{
            throw runtime_error("Wrong size of best move");
        }
        return delta;
    }

    MoveEvaluationResult evaluateMove(vector<int>& solution, const vector<int>& move){
        vector<int> wantedMove = {0, 181, 3, 157, 49};
        if(move == wantedMove){
            cout << solution.size() << endl;
            cout << (move[0]-1)%solution.size()<< " " << solution[(move[0]-1)%solution.size()] << " " << move[3] << endl;
            cout << "Wanted move" << endl;
        }
        if(move.size() == 5){
            //check if can do and then if delta calculation is still valid
            if(visited[move[1]] || solution[move[0]] != move[2] || solution[(move[0]-1)%solution.size()] != move[3] || solution[(move[0]+1)%solution.size()] != move[4]){
                return removeMove;
            }
            return doMove;
        }else if(move.size() == 8){//edge move
            //if all nodes are the same on the same positions return doMove
            if(solution[move[0]] == move[4] && solution[move[1]] == move[5] && solution[move[2]] == move[6] && solution[move[3]] == move[7]){
                return doMove;
            }
            //if all nodes are the same but reversed skip move
            if(solution[move[0]] == move[5] && solution[move[1]] == move[4] && solution[move[2]] == move[7] && solution[move[3]] == move[6]){
                return skipMove;
            }
            //else removeMove
            return removeMove;

        }else{
            throw runtime_error("Wrong size of move");
        }
    }

    void applyMove(vector<int>* solution, const vector<int>& move){//modifies solution i think usuall  poinnter is enuogh
        if(move.size() == 5){
            int i = move[0];
            int j = move[1];
            (*solution)[i] = j;
            visited[move[2]] = false;
            visited[j] = true;
        }else if(move.size() == 8){
            int j = move[1];
            int k = move[2];
            reverse(solution->begin() + j, solution->begin() + k + 1);
        }
    }

    vector<vector<int>> generateNewMoves(vector<int> & solution, const vector<int>& move){
        vector<vector<int>> newMoves;
        if(move.size() == 5){
//            throw runtime_error("move 5 should not happen");
            int currentNodeId = move[0];
            int currentNode = move[2];
            int previousNode = solution[(move[0]-1)%solution.size()];//could take fro move cause they do not change
            int nextNode = solution[(move[0]+1)%solution.size()];
            int newNode = move[1];
            //generate moves from newly inserte node to all not inserted
            for(int i=0;i<distances.size(); i++){
                if(i!=move[1] && !visited[i]){//if different than node currently being inserted
                    newMoves.push_back(makeInterMove(currentNodeId, i, currentNode, previousNode, nextNode));
                }
            }
            //generate moves from previously inserted node to all already in solution
            for(int i=0; i< solution.size(); i++){
                if(i == move[0]) continue;//skip the node that was just inserted
                newMoves.push_back(makeInterMove(i, currentNode, solution[i], solution[(i-1)%solution.size()], solution[(i+1)%solution.size()]));
            }
        }else if(move.size() == 8){
//            throw runtime_error("move 8 should not happen");
            //gnerate moves from new edges to all already in solution
            //generate for edge1
            //i => j
            //move[0] => move[2]
            //actually after mov it is just i => i+1
            for(int i=0; i<move[0]-1; i++){
                newMoves.push_back(makeIntraMove(i, i+1, move[0], move[1], solution[i], solution[(i+1)%solution.size()], solution[move[0]], solution[move[1]]));
            }
            for(int i=move[0]+2; i< solution.size(); i++){
                newMoves.push_back(makeIntraMove(move[0], move[1], i, i+1, solution[move[0]], solution[move[1]], solution[i], solution[(i+1)%solution.size()]));
            }
            //generate for edge2
            //i+1 => j+1
            //move[1] => move[3]
            for(int i=0; i<move[2]-1; i++){
                newMoves.push_back(makeIntraMove(i, i+1, move[2], move[3], solution[i], solution[(i+1)%solution.size()], solution[move[2]], solution[move[3]]));
            }
            for(int i=move[2]+2; i< solution.size(); i++){
                newMoves.push_back(makeIntraMove(move[2], move[3], i, i+1, solution[move[2]], solution[move[3]], solution[i], solution[(i+1)%solution.size()]));
            }
        }else{
            throw runtime_error("Wrong size of move");
        }
        return newMoves;
    }

    
    void checkDelta(vector<int> currentSolution, vector<int> move){
        vector<int> neighbour = currentSolution;
        int delta;
        if(move.size() == 5){//node exchange
            // exchange nodes
            int i = move[0];
            int new_node = move[1];
            int old_node = currentSolution[i];
            int oldCost = costs[old_node] + distances[old_node][currentSolution[(i+1)%currentSolution.size()]] + distances[old_node][currentSolution[(i-1+currentSolution.size())%currentSolution.size()]];
            int newCost = costs[new_node] + distances[new_node][currentSolution[(i+1)%currentSolution.size()]] + distances[new_node][currentSolution[(i-1+currentSolution.size())%currentSolution.size()]];
            delta = newCost - oldCost;
            neighbour[i] = new_node;
        }
        else if(move.size() == 8){
            // edge exchnge
            int j = move[1];
            int k = move[2];
            reverse(neighbour.begin() + j, neighbour.begin() + k + 1);
            int edge1_first = currentSolution[move[0]];
            int edge1_second = currentSolution[move[1]];
            int edge2_first = currentSolution[move[2]];
            int edge2_second = currentSolution[move[3]];
            int oldCost = distances[edge1_first][edge1_second] + distances[edge2_first][edge2_second];
            int newCost = distances[edge1_first][edge2_first] + distances[edge1_second][edge2_second];
            delta = newCost - oldCost;
        }
        else{
            throw runtime_error("Wrong size of best move");
        }

        if(calculate_cost(currentSolution) + delta != calculate_cost(neighbour)){
            //print current solution
            cout << "Current solution: ";
            for(int i=0; i<currentSolution.size(); i++){
                cout << currentSolution[i] << " ";
            }
            cout << endl;
            cout << "Move: ";
            for(int i=0; i<move.size(); i++){
                cout << move[i] << " ";
            }
            cout << endl;
            cout << "Neighbour: ";
            for(int i=0; i<neighbour.size(); i++){
                cout << neighbour[i] << " ";
            }
            cout << endl;
            cout << "Delta: " << delta << endl;
            cout << "Current cost: " << calculate_cost(currentSolution) << endl;
            cout << "Neighbour cost: " << calculate_cost(neighbour) << endl;
            throw runtime_error("Wrong delta");
        }

    }



    generator<vector<int>> neighbourhoodGenerator(vector<int>& currentSolution){
//        auto intraNeighbourhoodIterator = intraEdgesNeighbourhoodGenerator(currentSolution);
//        while(intraNeighbourhoodIterator.move_next()){
//            co_yield intraNeighbourhoodIterator.current_value();
//        }
        auto interNeighbourhoodIterator = interNeighbourhoodGenerator(currentSolution);
        while(interNeighbourhoodIterator.move_next()){
            co_yield interNeighbourhoodIterator.current_value();
        }

    }

    generator<vector<int>> interNeighbourhoodGenerator(vector<int>& currentSolution){
        vector<int> move = {0, 0};
        for (int i = 0; i < currentSolution.size(); i++) {
            int currentnode = currentSolution[i];
            for (int j = 0; j < distances.size(); j++) {
                if (!visited[j]) {
                    move[0] = i;//solution node index
                    move[1] = j;//new node
                    move[2] = currentSolution[i];//current node
                    move[3] = currentSolution[i-1]%currentSolution.size();//previous node
                    move[4] = currentSolution[(i+1)%currentSolution.size()];//next node
                    //move cost calculation is applicable if previous node and next node are still the same 
                    co_yield makeInterMove(i, j, currentSolution[i], currentSolution[(i-1)%currentSolution.size()], currentSolution[(i+1)%currentSolution.size()]);
                }
            }
        }
    }

    vector<int> makeInterMove(int currentNodeId, int newNode, int currentNode, int previousNode, int nextNode){
        return {currentNodeId, newNode, currentNode, previousNode, nextNode};
    }


    generator<vector<int>> intraEdgesNeighbourhoodGenerator(vector<int>& currentSolution){
        vector<int> temp_vec = {0, 0, 0, 0};
        vector<int> move = vector<int>(temp_vec);
        for (int i = 0; i < currentSolution.size(); i++) {
            int node1 = currentSolution[i];
            int node1_next = currentSolution[(i + 1) % currentSolution.size()];
            for (int j = i + 2; j < currentSolution.size(); j++) {
                int node2 = currentSolution[j];
                int node2_next = currentSolution[(j + 1) % currentSolution.size()];
                co_yield makeIntraMove(i, i+1, j, (j + 1) % currentSolution.size(), node1, node1_next, node2, node2_next);
            }
        }
    }

    vector<int> makeIntraMove(int edge1_first, int edge1_second, int edge2_first, int edge2_second, int edge1_first_node, int edge1_second_node, int edge2_first_node, int edge2_second_node){
        return {edge1_first, edge1_second, edge2_first, edge2_second, edge1_first_node, edge1_second_node, edge2_first_node, edge2_second_node};
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
        // cout << "Name: " << algo_name + "_"+ data_name + ".csv" << endl;
        string filename = "results/" + algo_name + "_"+ data_name + ".csv";
        ofstream file;
        file.open(filename);
        for(int i=0; i<sol.size(); i++){
            file << sol[i] << endl;
        }
        file.close();
    }


int main(){
    //print all files in current folder and current folder path
     system("ls");
     system("pwd");
    string root_path = "../../data/";
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
                    cout << "Name: " << LocalSearch(searchType, initialSolutionType, interNeighbourhoodType, distances, costs, 0).name << endl;
                    Result algoResult = Result(INT32_MAX, 0, 0, vector<int>(), vector<int>());
                    double averageTime = 0;
                    for(int i=0; i<distances.size(); i++){
                        cout << i << endl;
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