#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

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
    auto distances = calcDistances(data);
    NearestNeighboursSearch algo;
    vector<int> costs;
    for(int i=0; i< data.size(); i++){
        costs.push_back(data[i][2]);
    }
    auto result = algo.solve(distances, costs);
    //print result
    cout << "Cost " << result.second << endl;
    for(int i=0; i<result.first.size(); i++){
        cout<<result.first[i]<<" ";
    }
}