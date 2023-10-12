#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <list>


using namespace std;



class Algorithm {
public:
    pair<vector<int>,int> solve(vector<vector<int>> distances, vector<int> costs);
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



class RandomSearch: Algorithm {
    int n;
public:
    RandomSearch(int n_param) : n(n_param) { }
    
    pair<vector<int>,int> solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> bestSolution;
        int bestCost = INT32_MAX;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        for(int i=0; i<n; i++){
            //empty vector size solution_size
            current_solution = vector<int>(solution_size);
            //fill with random numbers
            for(int j=0; j<solution_size; j++){
                current_solution[j] = rand() % distances.size();
            }
            int current_cost = calculate_cost(current_solution, distances, costs);
            cout << current_cost << endl;
            if(current_cost < bestCost){
                bestCost = current_cost;
                bestSolution = current_solution;
            }
        }
        return make_pair(bestSolution, bestCost);
    }
};


class NearestNeighboursSearch: Algorithm {
public:
    pair<vector<int>,int> solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> bestSolution;
        int bestCost = INT32_MAX;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        for(int i=0;i<distances.size();i++){
            current_solution.push_back(i);
            vector<bool> visited(costs.size());
            while(current_solution.size() < solution_size){
                int min_distance = INT32_MAX;
                int min_index = -1;
                for(int j=0; j<distances.size(); j++){
                    if(visited[j]) continue;
                    if(distances[current_solution[current_solution.size()-1]][j] < min_distance){
                        min_distance = distances[current_solution[current_solution.size()-1]][j];
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
            current_solution.clear();
        }
        return make_pair(bestSolution, bestCost);
    }
};

class GreedyCycle: Algorithm {
public:
    
    void write_vector_to_file(vector<int> sol){
        string filename = "animation_greedy/" + to_string(sol.size()) + ".csv";
        ofstream file;
        file.open(filename);
        for(int i=0; i<sol.size(); i++){
            file << sol[i] << endl;
        }
        file.close();
    }


    int FindNearestUnvisitedNode(int node, vector<bool> visited, vector<vector<int>> distances){
        int min_distance = INT32_MAX;
        int min_index = -1;
        for(int j=0; j<distances.size(); j++){
            if(visited[j]) continue;
            if(distances[node][j] < min_distance){
                min_distance = distances[node][j];
                min_index = j;
            }
        }
        return min_index;
    }


    pair<int, int> PlacementAndIncrease(int node, int index, int nearest_vertex, vector<int> cycle, vector<vector<int>> distances){
        int inFront = -distances[cycle[index == 0 ? cycle.size() - 1 : index - 1]][node] + distances[cycle[index == 0 ? cycle.size() - 1 : index - 1]][nearest_vertex] + distances[nearest_vertex][node];
        int behind = -distances[node][cycle[(index+1)%cycle.size()]] + distances[nearest_vertex][cycle[(index+1)%cycle.size()]] + distances[node][nearest_vertex];
        if (inFront < behind){
            return make_pair(index, inFront);
        }
        else{
            return make_pair(index+1, behind);
        }
    }


    pair<vector<int>,int> solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> bestSolution;
        int bestCost = INT32_MAX;
        int solution_size = distances.size()/2;
        vector<int> current_solution;
        vector<vector<int>> graph;

        for(int i=0;i<distances.size();i++){
            cout << i << endl;
            current_solution.push_back(i);
            vector<bool> visited(costs.size());
            // visited[i] = true;
            vector<vector<int>> curr_graph;


            while(current_solution.size() < solution_size){
                // int nearest_vertex = INT32_MAX;
                // int index;
                int smallest_increase = INT32_MAX;
                int insert_index = -1;
                int insert_node = -1;


                for(int j=0; j<current_solution.size(); j++){
                    int nearest = FindNearestUnvisitedNode(current_solution[j], visited, distances);
                    pair<int, int> placement = PlacementAndIncrease(current_solution[j], j, nearest, current_solution, distances);
                    int increase = placement.second;
                    if(increase < smallest_increase){
                        smallest_increase = increase;
                        insert_index = placement.first;
                        insert_node = nearest;
                    }
                }
                current_solution.insert(current_solution.begin() + insert_index, insert_node);
                visited[insert_node] = true; 
                // write_vector_to_file(current_solution);
                curr_graph.push_back(current_solution);
            }
            int current_cost = calculate_cost(current_solution, distances, costs);
            if(current_cost < bestCost){
                bestCost = current_cost;
                bestSolution = current_solution;
                graph = curr_graph;
            }
            current_solution.clear();
        }
        // To save process of creating a graph
        // for(int i=0; i<graph.size(); i++){
        //     write_vector_to_file(graph[i]);
        // }
        return make_pair(bestSolution, bestCost);
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
    srand(static_cast<unsigned>(time(0)));
    auto data = read_file("./TSPA.csv");
    // auto data = read_file("./TSPB.csv");
    // auto data = read_file("./TSPC.csv");
    // auto data = read_file("./TSPD.csv");
    auto distances = calcDistances(data);
    GreedyCycle algo;
    NearestNeighboursSearch algo2;
    vector<int> costs;
    for(int i=0; i< data.size(); i++){
        costs.push_back(data[i][2]);
    }
    auto result = algo.solve(distances, costs);
    auto result2 = algo2.solve(distances, costs);
    //print result
    cout << "Greedy cycle: " << endl;
    cout << "Cost " << result.second << endl;
    for(int i=0; i<result.first.size(); i++){
        cout<<result.first[i]<<" ";
    }
    cout << endl;
    cout << "Nearest neighbours: " << endl;
    cout << "Cost " << result2.second << endl;
    for(int i=0; i<result2.first.size(); i++){
        cout<<result2.first[i]<<" ";
    }
}