#include <unordered_map>
#include <map>
#include <unordered_set>
#include <vector>
#include <tuple>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class DLinkedNode {
    public:
        int key;
        int val;
        DLinkedNode *next;
        DLinkedNode *prev;

        DLinkedNode(int key, int val) {
            this->key = key;
            this->val = val;
            this->next = nullptr;
            this->prev = nullptr;
        }
};


class LRUCache {
    private:
        int capacity;
        int curr_capacity;
        unordered_map<int, DLinkedNode*> cache;
        DLinkedNode *front = new DLinkedNode(-1, -1);
        DLinkedNode *back = new DLinkedNode(-1, -1);

        void moveToFront(DLinkedNode* node) {
            node->prev->next = (node->next);
            node->next->prev = (node->prev);
            insertAtFront(node);
        }

        void insertAtFront(DLinkedNode* node) {
            node->next = (this->front->next);
            node->prev = (this->front);
            node->next->prev = (node);
            node->prev->next = (node);
        }

        void evictLRU() {
            // store back.prev.prev as a temp
            // delete back.prev
            // temp.next = back
            // back.prev = temp
            // capacity --
            DLinkedNode *temp = this->back->prev->prev;
            this->cache.erase(this->back->prev->key);
            delete this->back->prev;
            temp->next = (this->back);
            this->back->prev = (temp);
            this->curr_capacity -= 1;
        }

    public:
        LRUCache(int capacity) {
            this->front->next = (back);
            this->back->prev = (front);
            this->capacity = capacity;
            this->curr_capacity = 0;
        }
        
        int get(int key) {
            if (this->cache.find(key) == this->cache.end()) {
                return -1;
            } else {
                moveToFront(this->cache.at(key));
                return this->cache.at(key)->val;
            }
        }
        
        void put(int key, int value) {

            if (this->cache.find(key) == this->cache.end() && this->curr_capacity == this->capacity) {
                // create new node, call del on last node (declared on heap), insert at front, insert to unordered map                
                evictLRU();

                DLinkedNode *created_cache = new DLinkedNode(key, value);
                this->cache[key] = created_cache;
                insertAtFront(created_cache);
                this->curr_capacity += 1;


            } else if (this->cache.find(key) == this->cache.end()) {
                // create new node, insert at front, insert to unordered map
                DLinkedNode *created_cache = new DLinkedNode(key, value);
                this->cache[key] = created_cache;
                insertAtFront(created_cache);
                this->curr_capacity += 1;

            } else {
                // update the value only, also move to front
                moveToFront(this->cache.at(key));
                this->cache.at(key)->val = (value);
                
            }
        }
};


class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> seen;
        for (unsigned int i = 0; i < nums.size(); i++) {
            if (seen.find(nums[i]) == seen.end()) {
                seen.insert(nums[i]);
            } else {
                return true;
            }
        }
        return false;
    }

};

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> needed;
        vector<int> out = {-1, -1};
        for (unsigned int i = 0; i < nums.size(); i++) {
            if (needed.find(nums[i]) == needed.end()) {
                needed[target - nums[i]] = i;
            } else {
                out = {needed[nums[i]], (int) i};
                return out;
            }
        }
        return out;
    }
};

class Solution {
public:
    /*
    brute force:
    for each index in nums, loop through and multiply all numbers in nums except this index

    intuition: maintain a running bidirectional "tally" and we only need to loop through forward once
                and reverse once. then we onnly need to multiple array indexes together forward[i-1] * backward[i+1]

                time: O(n)
                space: O(n)

    approach:
    - loop through forward and maintain a vector<int> forward and running multiplcation of all elements up to and including i
    - loop through backward and maintain a vector<int> backwad and running multiplcation of all elements from len(num) to i

    - create vector<int> out, out[i] = forward[i-1] * backward[i+1]
    - some checking logic for i-1 and i+1 being out of bounds

    optimized approach:
    - similar to approach 1, we still carry but this time we use our output array as our caryr
    - we do a forward and then a bakcward loop.
    */
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> out;

        if (nums.size() == 0) {
            return out;
        } else if (nums.size() == 1) {
            out = {0};
            return out;
        }

        int current = 1;
        for (unsigned int i = 0; i < nums.size(); i++) {
            out.push_back(current);
            current *= nums[i];
        }

        int current = 1;
        for (unsigned int i = 0; i < nums.size(); i++) {
            out[out.size() - 1 - i] *= current;
            current *= out[out.size() - 1 - i];
        }

        return out;
    }
};

class Solution {
public:
    // ieee
    /*
    32 bit -> - 2^31 -> 2^31 - 1
    
    
    note: >> 1 divides by 2 
    
    1101
    
    1011
    
    10
    
    101
    
    */
    uint32_t reverseBits(uint32_t n) {
        unsigned int out = 0;
        for (int i = 0; i < 32; i++) {
            unsigned int val = (n << 32) >> 32;
            out = out * 2 + val;
            n = n >> 1;
        }
        return out;
    }
};

class Solution {
public:
    /*
        1. 
        maintain a reference to an integer counter
        we recursively call on down and right if possible
        we update counter + 1 if we reach finish

        2. queue based bfs approach, iterative 
        at each step, we queue the left and bottom addresses to be visited
        when we reach finish, we update our count
        loop until the queue is empty

        3. subproblems:
        intution: the amount of ways to reach grid[i][j] is grid[i-1][j] + grid[i][j-1]
        we can build out our solution in O(mn)

    */
    int uniquePaths(int m, int n) {
        if (m == 0 && n == 0) {
            return 1;
        }
        vector<vector<int>> solution;

        for (unsigned int i = 0; i < m; i++) {
            vector<int> layer;
            for (unsigned int j = 0; j < n; j++) {
                layer.push_back(0);
            }
            solution.push_back(layer);
        }

        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    solution[i][j] = 1;
                } else if (i - 1 < 0) {
                    solution[i][j] = solution[i][j-1];
                } else if (j - 1 < 0) {
                    solution[i][j] = solution[i-1][j];
                } else {
                    solution[i][j] = solution[i-1][j] + solution[i][j-1];
                }
            }
        }

        return solution[m-1][n-1];
    }
};

class Solution {
public:
    tuple<double, double> getLinearEquation(tuple<int, int>& ref1, tuple<int, int> ref2) {
        double slope = (double)(get<1>(ref2)- get<1>(ref1)) / (double)(get<0>(ref2) - get<0>(ref1));
        double y_int = (double)get<1>(ref2) - (double)(get<0>(ref2) * slope);
        tuple <double, double> out = make_tuple(slope, y_int);
        return out;
    }

    bool isCollinear(tuple<int, int>& ref1, tuple<int, int>& ref2, tuple<int, int>& new_point) {
        double slope = (double)(get<1>(ref2) - get<1>(ref1)) / (double)(get<0>(ref2) - get<0>(ref1));
        return (get<1>(new_point) == (get<0>(new_point) - get<0>(ref1)) * slope + get<1>(ref1));
    }

    int maxPoints(vector<vector<int>>& points) {
        if (points.size() == 1) {
            return 1;
        }
        unordered_map<tuple<double, double>, unordered_set<tuple<int, int>>> equations;

        for (unsigned int i = 0; i < points.size(); i++) {
            for (unsigned int j = i + 1; j < points.size(); j++) {
                tuple<int, int> ref_i = make_tuple(points[i][0], points[i][1]);
                tuple<int, int> ref_j = make_tuple(points[j][0], points[j][1]);
                tuple<double, double> lin_equation = getLinearEquation(ref_i, ref_j);
                if (equations.find(lin_equation) == equations.end()) {
                    unordered_set<tuple<int, int>> points_on_line;
                    points_on_line.insert(ref_i);
                    points_on_line.insert(ref_j);
                    equations[lin_equation] = points_on_line;
                } else {
                    if (equations[lin_equation].find(ref_i) != equations[lin_equation].end()) {
                        equations[lin_equation].insert(ref_i);
                    }
                    if (equations[lin_equation].find(ref_j) != equations[lin_equation].end()) {
                        equations[lin_equation].insert(ref_j);
                    }
                }
            }
        }

        int most = 0;
        for (auto& it: equations) {
            most = max(most, (int)it.second.size());
        }
        return most;
    }
};

class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        if (board.size() != 9 || board[0].size() != 9) {
            return false;
        }

        for (unsigned int i = 0; i < board.size(); i++) {
            unordered_set<char> seen;
            for (unsigned int j = 0; j < board[0].size(); j++) {
                if (seen.find(board[i][j]) == seen.end()) {
                    seen.insert(board[i][j]);
                } else if (board[i][j] != '.') {
                    return false;
                }
            }
        }
        for (unsigned int i = 0; i < board.size(); i++) {
            unordered_set<char> seen;
            for (unsigned int j = 0; j < board[0].size(); j++) {
                if (seen.find(board[j][i]) == seen.end()) {
                    seen.insert(board[j][i]);
                } else if (board[j][i] != '.') {
                    return false;
                }
            }
        }
        for (unsigned int i = 0; i < 3; i++) {
            for (unsigned int j = 0; j < 3; j++) {
                unordered_set<char> seen;
                for (unsigned int r = 0; r < 3; r++) {
                    for (unsigned int c = 0; c < 3; c++) {
                        if (seen.find(board[i * 3 + r][j * 3 + c]) == seen.end()) {
                            seen.insert(board[i * 3 + r][j * 3 + c]);
                        } else if (board[i * 3 + r][j * 3 + c] != '.') {
                            cout << i * 3 + r << " " << j * 3 + c << endl;
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }
};


class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* out = nullptr;
        while (list1 && list2) {

        }
    }
};




int main() {
    return 0;
}


