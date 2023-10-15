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

using namespace std;


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
    std::string name;
    virtual Result solve(vector<vector<int>> distances, vector<int> costs) =0;
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
};



class RandomSearch: public Algo {
    int n;
public:
    RandomSearch(int n_param) : n(n_param) {
        name = "RandomSearch";
    }
    
    Result solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> bestSolution;
        vector<int> worstSolution;
        int bestCost = INT32_MAX;
        int averageCost = 0;
        int worstCost = 0;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        for(int i=0; i<n; i++){
            //empty vector size solution_size
            current_solution = vector<int>(solution_size);
            vector<int> visited(distances.size());
            //fill with random numbers
            for(int j=0; j<solution_size; j++){
                int next = rand() % distances.size();
                while(visited[next])next = rand() % distances.size();
                current_solution[j] = next;
                visited[next]=true;
            }
            int current_cost = calculate_cost(current_solution, distances, costs);
            if(current_cost < bestCost){
                bestCost = current_cost;
                bestSolution = current_solution;
            }
            if(current_cost > worstCost){
                worstCost = current_cost;
                worstSolution = current_solution;
            }
            averageCost += current_cost;

        }
        return Result(bestCost, worstCost, averageCost/distances.size(), bestSolution, worstSolution);
    }
};


class NearestNeighboursSearch: public Algo {
public:
    NearestNeighboursSearch() {
        name = "NearestNeighboursSearch";
    }
    Result solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> bestSolution;
        vector<int> worstSolution;
        int bestCost = INT32_MAX;
        int averageCost = 0;
        int worstCost = 0;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        for(int i=0;i<distances.size();i++){
            current_solution.push_back(i);
            vector<bool> visited(costs.size());
            visited[i] = true;
            while(current_solution.size() < solution_size){
                int min_cost = INT32_MAX;
                int min_index = -1;
                for(int j=0; j<distances.size(); j++){
                    if(visited[j]) continue;
                    if(distances[current_solution[current_solution.size()-1]][j] < min_cost){
                        min_cost = distances[current_solution[current_solution.size()-1]][j] + costs[j];
                        min_index = j;
                    }
                }
                visited[min_index] = true;
                current_solution.push_back(min_index);
            }
            int current_cost = calculate_cost(current_solution, distances, costs);
            if(current_cost < bestCost){
                bestCost = current_cost;
                bestSolution = current_solution;
            }
            if(current_cost > worstCost){
                worstCost = current_cost;
                worstSolution = current_solution;
            }
            averageCost += current_cost;
            current_solution.clear();
        }
        return Result(bestCost, worstCost, averageCost/distances.size(), bestSolution, worstSolution);
    }
};

class GreedyCycle: public Algo {
public:
    GreedyCycle() {
        name = "GreedyCycle";
    }
    void write_vector_to_file(vector<int> sol){
        string filename = "animation_greedy/" + to_string(sol.size()) + ".csv";
        ofstream file;
        file.open(filename);
        for(int i=0; i<sol.size(); i++){
            file << sol[i] << endl;
        }
        file.close();
    }




    Result solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> bestSolution;
        vector<int> worstSolution;
        int bestCost = INT32_MAX;
        int worstCost = 0;
        int averageCost = 0;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        vector<vector<int>> graph;
        int starting_node;

        for(int i=0;i<distances.size();i++){
            current_solution.push_back(i);
            vector<bool> visited(costs.size());
            visited[i] = true;
            vector<vector<int>> curr_graph;


            while(current_solution.size() < solution_size){
                // if(current_solution.size() == 2){
                //     current_solution.insert(current_solution.begin(), i);
                //     continue;
                // }
                int smallest_increase = INT32_MAX;
                int insert_index = -1;
                int insert_node = -1;


                for(int j=0; j<current_solution.size(); j++){  // Dla każdego nodea z cyklu
                    int min_distance = INT32_MAX;
                    int min_index = -1;
                    for(int k=0; k<distances.size(); k++){ //znajdź najbliższy nieodwiedzony node
                        if(visited[k]) continue;
                        int curr = -distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + distances[k][current_solution[j]] + costs[k];
                        if(curr < min_distance){
                            min_distance = curr;
                            min_index = k;
                        }
                    }
                    if(min_distance < smallest_increase){
                        smallest_increase = min_distance;
                        insert_index = j;
                        insert_node = min_index;
                    }
                } // koniec
                current_solution.insert(current_solution.begin() + insert_index, insert_node);
                visited[insert_node] = true; 
                // curr_graph.push_back(current_solution);
            }
            int current_cost = calculate_cost(current_solution, distances, costs);
            if(current_cost < bestCost){
                bestCost = current_cost;
                bestSolution = current_solution;
                graph = curr_graph;
                starting_node = i;
            }
            if(current_cost > worstCost){
                worstCost = current_cost;
                worstSolution = current_solution;
            }
            averageCost += current_cost;
            current_solution.clear();
        }
        // To save process of creating a graph
        for(int i=0; i<graph.size(); i++){
            write_vector_to_file(graph[i]);
        }
        return Result(bestCost, worstCost, averageCost/distances.size(), bestSolution, worstSolution);
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



int main(){
    // srand(static_cast<unsigned>(time(0)));
    vector<Algo*> algorithms;
    algorithms.push_back(new RandomSearch(200));
    algorithms.push_back(new GreedyCycle());
    algorithms.push_back(new NearestNeighboursSearch());

    string files[] = {"./TSPA.csv", "./TSPB.csv", "./TSPC.csv", "./TSPD.csv"};

    for(auto algo: algorithms){
        cout<<"#Algorithm: "<< algo->name << endl;
        for(string file: files){
            cout<<"##File: "<< file << endl;
            auto data = read_file("./TSPA.csv");
            auto distances = calcDistances(data);
            vector<int> costs;
            for(int i=0; i< data.size(); i++){
                costs.push_back(data[i][2]);
            }
            auto result = algo->solve(distances, costs);
            cout << "Best cost: " << result.bestCost << endl;
            cout << "Worst cost: " << result.worstCost << endl;
            cout << "Average cost: " << result.averageCost << endl;
            cout << "Best solution: " << endl;
            for(int i=0; i<result.bestSolution.size(); i++){
                cout<<result.bestSolution[i]<<" ";
            }
            cout<<endl;
        }
    }

    return 0;
    }