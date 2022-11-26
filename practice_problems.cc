#include <unordered_map>
#include <map>
#include <unordered_set>
#include <vector>

using namespace std;


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


int main() {
    return 0;
}


