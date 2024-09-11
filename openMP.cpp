#include <bits/stdc++.h>
#include <omp.h>
using namespace std;


void generateRandomTestCases(int numProblems, int numItems, vector<int>& capacities, vector<vector<int>>& weights, vector<vector<int>>& values, int maxCapacity, int maxWeight, int maxValue) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> capDist(1, maxCapacity); 
    uniform_int_distribution<> weightDist(1, maxWeight);
    uniform_int_distribution<> valueDist(1, maxValue); 

    for (int i = 0; i < numProblems; i++) {
        capacities.push_back(capDist(gen));
    }

    for (int i = 0; i < numProblems; i++) {
        vector<int> tempWeights;
        vector<int> tempValues;
        for (int j = 0; j < numItems; j++) {
            tempWeights.push_back(weightDist(gen));
            tempValues.push_back(valueDist(gen));
        }
        weights.push_back(tempWeights);
        values.push_back(tempValues);
    }
}

int knapSackRec(int W, int wt[], int val[], int index, int** dp) {
    if (index < 0)
        return 0;

    if (dp[index][W] != -1)
        return dp[index][W];

    if (wt[index] > W) {
        dp[index][W] = knapSackRec(W, wt, val, index - 1, dp);
        return dp[index][W];
    } else {
        dp[index][W] = max(val[index] + knapSackRec(W - wt[index], wt, val, index - 1, dp),
                           knapSackRec(W, wt, val, index - 1, dp));
        return dp[index][W];
    }
}

int knapSack(int W, int wt[], int val[], int n) {
    int** dp = new int*[n];
    for (int i = 0; i < n; i++)
        dp[i] = new int[W + 1];

    for (int i = 0; i < n; i++)
        for (int j = 0; j <= W; j++)
            dp[i][j] = -1;

    int result = knapSackRec(W, wt, val, n - 1, dp);

    for (int i = 0; i < n; i++)
        delete[] dp[i];
    delete[] dp;

    return result;
}

void solveMultipleKnapsackProblems(vector<int>& capacities, 
                                    vector<vector<int>>& weights, 
                                   vector<vector<int>>& values, 
                                   vector<int>& results) {
    int numProblems = capacities.size();

#pragma omp parallel
    {
        int numThreads = omp_get_num_threads();  
        int threadID = omp_get_thread_num();

        for (int i = threadID; i < numProblems; i += numThreads) {
            int n = weights[i].size();
            results[i] = knapSack(capacities[i], weights[i].data(), values[i].data(), n);
        }
    }
}

int main() {
    int numProblems = 100;  
    int numItems = 10;   
    int maxCapacity = 100; 
    int maxWeight = 40;   
    int maxValue = 200;   

    vector<int> capacities;
    vector<vector<int>> weights;
    vector<vector<int>> values;
    generateRandomTestCases(numProblems, numItems, capacities, weights, values, maxCapacity, maxWeight, maxValue);

    
    vector<int> results(numProblems, 0);

    solveMultipleKnapsackProblems(capacities, weights, values, results);

    for (int i = 0; i < numProblems; i++) {
        cout << "Knapsack Problem " << i + 1 << ": Max Value = " << results[i] << endl;
    }

    return 0;
}
