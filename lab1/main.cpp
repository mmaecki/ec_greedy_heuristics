#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

using namespace std;

class Algorithm {
public:
    vector<int> solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> output;
        return output;
    }
};



class RandomAlgirthm{
    //constructor accepting n as parameter
    
    public:
    vector<int> solve(vector<vector<int>> distances, vector<int> costs) {
        vector<int> output;
        return output;
    }
}

vector<vector<int>> read_file(string filename) {
    vector<vector<int>> result;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        cout<<line<<endl;
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
    auto data = read_file("./TSPA.csv");
    auto distances = calcDistances(data);
    Algorithm algo;
    //select 3rd column from data
    vector<int> costs;
    for(int i=0; i< data.size(); i++){
        costs.push_back(data[i][2]);
    }
    auto result = algo.solve(distances, costs);
}