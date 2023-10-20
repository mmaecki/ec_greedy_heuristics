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





class Greedy2Regret: public Algo {
public:
    Greedy2Regret() {
        name = "Greedy2Regret";
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
             
                int smallest_increase = INT32_MAX;
                int insert_index = -1;
                int insert_node = -1;
                int max_regret = -1;


                for(int k=0; k<distances.size(); k++){ // dla wszystkich nieodwiedzonych nodeów
                    if(visited[k]) continue;
                    vector<int> insertion_cost_for_j;
                    for(int j=0; j<current_solution.size(); j++){ // dla każdego nodea z cyklu
                        int curr = -distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + distances[k][current_solution[j]] + costs[k];
                        // int curr = -distances[current_solution[j]][current_solution[j == current_solution.size() - 1 ? 0 : j + 1]] + distances[current_solution[j]][k] + distances[k][current_solution[j == current_solution.size() - 1 ? 0 : j + 1]] + costs[k];
                        insertion_cost_for_j.push_back(curr);
                    }
                    int smallest_index = -1;
                    int smallest_value = INT32_MAX;
                    int second_smallest_value = INT32_MAX;

                    for (int h = 0; h < insertion_cost_for_j.size(); h++) {
                        if (insertion_cost_for_j[h] < smallest_value) {
                            second_smallest_value = smallest_value;
                            smallest_value = insertion_cost_for_j[h];
                            smallest_index = h+1;
                        } else if (insertion_cost_for_j[h] < second_smallest_value) {
                            second_smallest_value = insertion_cost_for_j[h];
                        }
                    }
                    int regret = second_smallest_value - smallest_value;
                    if(regret > max_regret){
                        max_regret = regret;
                        insert_index = smallest_index;
                        insert_node = k;
                    }
                }


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
            if(i == 47){
                for (int a = 0; a < current_solution.size(); a++) {
                    cout << current_solution[a] << " ";
                }
                cout << endl;
            }
            current_solution.clear();
        }
        return Result(bestCost, worstCost, averageCost/distances.size(), bestSolution, worstSolution);
    }
};


class Greedy2RegretWieghted: public Algo {
public:
    Greedy2RegretWieghted() {
        name = "Greedy2RegretWeighted";
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
             
                int smallest_increase = INT32_MAX;
                int insert_index = -1;
                int insert_node = -1;
                int max_score = -INT32_MAX;


                for(int k=0; k<distances.size(); k++){ // dla wszystkich nieodwiedzonych nodeów
                    if(visited[k]) continue;
                    vector<int> insertion_cost_for_j;
                    for(int j=0; j<current_solution.size(); j++){ // dla każdego nodea z cyklu
                        int curr = -distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][current_solution[j]] + distances[current_solution[j == 0 ? current_solution.size() - 1 : j - 1]][k] + distances[k][current_solution[j]] + costs[k];
                        insertion_cost_for_j.push_back(curr);
                    }
                    int smallest_index = -1;
                    int smallest_value = INT32_MAX;
                    int second_smallest_value = INT32_MAX;

                    for (int h = 0; h < insertion_cost_for_j.size(); h++) {
                        if (insertion_cost_for_j[h] < smallest_value) {
                            second_smallest_value = smallest_value;
                            smallest_value = insertion_cost_for_j[h];
                            smallest_index = h;
                        } else if (insertion_cost_for_j[h] < second_smallest_value) {
                            second_smallest_value = insertion_cost_for_j[h];
                        }
                    }
                    int regret = second_smallest_value - smallest_value;
                    int left_node_idx = smallest_index == 0 ? current_solution.size() -1 : smallest_index -1;
                    int insertion_cost = - distances[current_solution[left_node_idx]][current_solution[smallest_index]] + distances[current_solution[left_node_idx]][k] + distances[k][current_solution[smallest_index]] + costs[k];
                    int score = regret - insertion_cost;
                    if(score> max_score){
                        max_score = score;
                        insert_index = smallest_index;
                        insert_node = k;
                    }
                }

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
            if(i == 32){
                for (int a = 0; a < current_solution.size(); a++) {
                    cout << current_solution[a] << " ";
                }
                cout << endl;
            }
            current_solution.clear();
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
    // srand(static_cast<unsigned>(time(0)));
    vector<Algo*> algorithms;
    algorithms.push_back(new Greedy2Regret());
    algorithms.push_back(new Greedy2RegretWieghted());

    string files[] = {"../data/TSPA.csv", "../data/TSPB.csv", "../data/TSPC.csv", "../data/TSPD.csv"};

    for(auto algo: algorithms){
        cout<<"#Algorithm: "<< algo->name << endl;
        for(string file: files){
            cout<<"##File: "<< file << endl;
            auto data = read_file(file);
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
            // write_solution_to_file(result.bestSolution, algo->name, file.substr(file.size() - 8));
        }
    }
    return 0;
}