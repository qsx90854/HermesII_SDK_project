#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace VisionSDK {

class HungarianAlgorithm {
public:
    // Solve assignment problem (minimize cost)
    // Returns a vector where assignment[i] = assigned_col_for_row_i
    // If assigned_col_for_row_i == -1, row i is unassigned
    static std::vector<int> Solve(const std::vector<std::vector<float>>& costMatrix) {
        int rows = costMatrix.size();
        if (rows == 0) return {};
        int cols = costMatrix[0].size();
        
        // Implementation of Munkres / Hungarian Algorithm is complex.
        // For simplicity and considering N is small (usually < 20 in our case),
        // we can use a simpler greedy approximation or a recursive backtracking if real Hungarian is too big?
        // But true Hungarian is O(N^3).
        // Let's implement a simplified greedy matching if we don't want 500 lines of code.
        // BUT the user asked for Hungarian.
        // I will implement a standard O(N^3) or O(N^4) version.
        
        // Actually, for N < 20, a greedy approach with multiple passes is often "good enough" but fails optimality.
        // I will implement a proper one.
        
        // Step 1: Subtract row minima
        // Step 2: Subtract col minima
        // ...
        
        // To keep code concise, I will use a known compact implementation.
        // Assignment: vector<int> (size=rows, value=col_index or -1)
        
        // Since implementing full Hungarian from scratch is error-prone in one shot without test,
        // and standard library is not available, I will use a greedy approach first if that's acceptable?
        // No, user explicitly asked for "Hungarian".
        
        // Alternative: Use a library-free implementation.
        // Reference: Standard Hungarian Algorithm steps.
        
        // For the sake of this task and ensuring it fits in one file without bugs:
        // I will implement the Greedy Match if N is small?
        // NO. "Mode 2: Global Optimal Matching".
        
        // ... Okay, I will implement a basic version.
        
        // PAD matrix to square if needed
        int n = std::max(rows, cols);
        std::vector<std::vector<float>> cost = costMatrix;
        
        // Pad rows
        while(cost.size() < (size_t)n) {
            cost.push_back(std::vector<float>(cols, 999999.0f)); // Dummy rows
        }
        // Pad cols
        for(auto& row : cost) {
            while(row.size() < (size_t)n) row.push_back(999999.0f); // Dummy cols
        }
        
        std::vector<float> u(n + 1), v(n + 1), p(n + 1), way(n + 1);
        std::vector<int> linked(n + 1);
        
        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<float> minv(n + 1, std::numeric_limits<float>::max());
            std::vector<char> used(n + 1, false);
            
            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                float delta = std::numeric_limits<float>::max();
                
                for (int j = 1; j <= n; ++j) {
                    if (!used[j]) {
                        float cur = cost[i0-1][j-1] - u[i0] - v[j];
                        if (cur < minv[j]) minv[j] = cur, way[j] = j0;
                        if (minv[j] < delta) delta = minv[j], j1 = j;
                    }
                }
                
                for (int j = 0; j <= n; ++j) {
                    if (used[j]) u[p[j]] += delta, v[j] -= delta;
                    else minv[j] -= delta;
                }
                j0 = j1;
            } while (p[j0] != 0);
            
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }
        
        std::vector<int> assignment(rows, -1);
        for (int j = 1; j <= n; ++j) {
            if (p[j] > 0 && p[j] <= rows && j <= cols) {
                 // Cost check to avoid dummy assignments or very high costs
                 // if (costMatrix[p[j]-1][j-1] < 999990.0f)
                 assignment[p[j]-1] = j-1;
            }
        }
        
        return assignment;
    }
};

} // VisionSDK

#endif
