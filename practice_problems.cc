#include <unordered_map>
#include <map>

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





int main() {
    return 0;
}


