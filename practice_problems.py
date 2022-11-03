"""

CodeSignal Practice Problems!

"""


from distutils.command.build import build
import math
from tarfile import _Bz2ReadableFileobj

class ListNode:
    def __init__(self, x):
        self.value = x
        self.next = None

class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = None
        self.right = None


def firstDuplicate(a):
    vals = set()
    
    for i in a:
        if i in vals:
            return i
        else:
            vals.add(i)
            
    return -1


def firstNotRepeatingCharacter(s):
    from queue import Queue
    char_order = Queue()
    distinct_char = set()
    repeat_char = set()
    for i in s:
        if i in distinct_char:
            distinct_char.remove(i)
            repeat_char.add(i)
        elif i in repeat_char:
            continue
        else:
            distinct_char.add(i)
            char_order.put(i)
            
    while char_order.qsize() > 0:
        temp = char_order.get()
        if temp in distinct_char:
            return temp
    
    return '_'



def rotateImage(a):
    """
    1. reverse every col
    2. flip along the 1st determinant line
    """
    for col_num in range(0, len(a)):
        lo = 0
        hi = len(a) - 1
        while lo < hi:
            temp = a[lo][col_num]
            a[lo][col_num] = a[hi][col_num]
            a[hi][col_num] = temp
            lo += 1
            hi -= 1
    
    for row_num in range(0, len(a)):
        for col_num in range(row_num+1, len(a)):
            temp = a[row_num][col_num]
            a[row_num][col_num] = a[col_num][row_num]
            a[col_num][row_num] = temp
            
    return a


def sudoku2(grid):
    for row in grid:
        vals = set()
        for i in row:
            if i == '.':
                continue
                
            if i in vals:
                return False
            elif i not in vals:
                vals.add(i)
    
    for col_num in range(0, 9):
        vals = set()
        for row_num in range(0, 9):
            if grid[row_num][col_num] == '.':
                continue
            
            if grid[row_num][col_num] in vals:
                return False
            else:
                vals.add(grid[row_num][col_num])
                
    
    for i in range(0, 3):
        for j in range(0, 3):
            vals = set()
            for a in range(0, 3):
                for b in range(0, 3):
                    if grid[i * 3 + a][j * 3 + b] == '.':
                        continue
                    
                    if grid[i * 3 + a][j * 3 + b] in vals:
                        return False
                    else:
                        vals.add(grid[i * 3 + a][j * 3 + b])
    
    return True
            

def isCryptSolution(crypt, solution):
    def find_mapping(k):
        for i in solution:
            if i[0] == k:
                return int(i[1])
    
    def word_to_val(word):
        out = 0
        for i in range(0, len(word)):
            if i == 0 and len(word) != 1and find_mapping(word[i]) == 0:
                return -1
            
            out += find_mapping(word[i]) * (10 ** (len(word) - 1 - i))
        
        return out
        
    word1 = word_to_val(crypt[0])
    word2 = word_to_val(crypt[1])
    word3 = word_to_val(crypt[2])
    
    if word1 == -1 or word2 == -1 or word3 == -1:
        return False
    
    return (word1 + word2) == word3


def removeKFromList(l, k):
    # have pointers to 2 nodes, curr and next
    # if curr is null or next is null, break
    # if next value is k, set curr.next to next.next
    
    if l == None:
        return l
    
    if l.next == None and l.value == k:
        return None
    elif l.next == None:
        return l
        
    curr = l
    over = curr.next
    
    while True:
        if curr == None and over == None:
            break
        elif over == None:
            if curr.value == k:
                return None
            break
        
        if curr.value == k:
            l = over
            curr = over
            over = over.next
            continue
        
        if over.value == k:
            curr.next = over.next
            over = over.next
        else:
            curr = curr.next
            over = over.next
    
    return l



def isListPalindrome(l):
    length = 0
    
    if l == None or l.next == None:
        return True
        
    temp = l
    while True:
        if temp == None:
            break
        else:
            length += 1
            temp = temp.next
    
    temp = l
    
    for i in range(0, math.ceil(length / 2)):
        temp = temp.next
    
    next_node = temp.next
    
    while next_node and next_node.next:
        place_holder = next_node.next
        next_node.next = temp
        temp = next_node
        next_node = place_holder
        
    if next_node:
        next_node.next = temp
        temp = next_node

    for i in range(0, math.floor(length / 2)):
        if l.value != temp.value:
            print (l.value, temp.value)
            return False
        else:
            l = l.nexe
            temp = temp.nexe

    return True

def reverseLL(l):
    temp = l.next
    l.next = None

    while temp:
        place_holder = temp.nexe
        temp.next = l
        l = temp
        temp = place_holder
    
    return l
    
def findLength(l):
    length = l
    while l:
        length += l
        l = l.next
    
    return length

def addTwoHugeNumbers(a, b):
    """
    1. reverse a and b
    2. add with carry
    3. reverse the result list
    """
    a = reverseLL(a)
    b = reverseLL(b)

    a_length = findLength(a)
    b_length = findLength(b)

    prev = None
    carry = 0

    for i in range(0, min(a_length, b_length)):
        out = ListNode(0)
        if a.value + b.value + carry > 9999:
            out.value = a.value + b.value + carry - 10000
            carry = 1
        else:
            out.value = a.value + b.value + carry
            carry = 0
            
        a = a.next
        b = b.next
        if prev:
            prev.next = out
        else:
            first = out
        
        prev = out
    
    for i in range(0, max(a_length, b_length) - min(a_length, b_length)):
        out = ListNode(0)
        if a_length > b_length:
            if a.value + carry > 9999:
                out.value = a.value + carry - 10000
                carry = 1
            else:
                out.value = a.value + carry
                carry = 0
            
            a = a.next
        else:
            if b.value + carry > 9999:
                out.value = b.value + carry - 10000
                carry = 1
            else:
                out.value = b.value + carry
                carry = 0
            
            b = b.next
            
        if prev:
            prev.next = out
        else:
            first = out
        
        prev = out
    
    if carry:
        out = ListNode(1)
        prev.next = out
    
    res = reverseLL(first)
    return res


def mergeTwoLinkedLists(l1, l2):
    out = None
    temp = None
    while l1 and l2:
        if l1.value <= l2.value:
            if temp == None:
                out = l1
                temp = l1
                l1 = l1.next
            else:
                temp.next = l1
                temp = l1
                l1 = l1.next
        else:
            if temp == None:
                out = l2
                temp = l2
                l2 = l2.next
            else:
                temp.next = l2
                temp = l2
                l2 = l2.next

    if l1 and temp:
        temp.next = l1
    elif l1:
        out = l1
    elif l2 and temp:
        temp.next = l2
    elif l2:
        out = l2
        
    return out


def reverseNodesInKGroups(l, k):
    if k == 1:
        return l
    
    temp = l
    count = 0
    while temp:
        count += 1
        temp = temp.next
    
    l_start = None
    temp = l
    last_first = None
    for _ in range(0, math.floor(count / k)):
        first = temp
        next_node = temp.next
        temp.next = None
        for _ in range(0, k - 1):
            placeholder = next_node.next
            next_node.next = temp
            temp = next_node
            next_node = placeholder
        
        if l_start == None:
            l_start = temp
        
        if last_first:
            last_first.next = temp
        
        last_first = first
        temp = next_node
    
    if last_first:
        last_first.next = temp
    return l_start

def rearrangeLastN(l, n):
    """
    1. keep reference to first node
    2. keep 2 pointers of size n + 1
    3. if right.next == nullptr (None), set temp = left.next, left.next to None
    4. set right.next to first, and return temp
    """
    if n == 0:
        return l
    
    first = l
    left = l
    right = l
    for _ in range(0, n):
        if right.next == None:
           return first
        right = right.next

    while right.next:
        left = left.next
        right = right.next
    
    temp = left.next
    left.next = None
    right.next = first
    
    return temp


def areFollowingPatterns(strings, patterns):
    patternmap = dict()
    mappedpatterns = set()
    for i in range(0, len(strings)):
        if strings[i] not in patternmap:
            if patterns[i] in mappedpatterns:
                return False
            else:
                patternmap[strings[i]] = patterns[i]
                mappedpatterns.add(patterns[i])
        elif patterns[i] != patternmap[strings[i]]:
            return False
    
    return True
    
    for i in range(0, len(strings)):
        if strings[i] != patternmap[patterns[i]]:
            return False
            
    return True


def twoSumSorted(self, numbers, target):
    left = 1
    right = len(numbers)
    
    while True:
        if left == right:
            return [-1, -1]
        elif numbers[left - 1] + numbers[right - 1] == target:
            return [left, right]
        elif numbers[left - 1] + numbers[right - 1] > target:
            right -= 1
        elif numbers[left - 1] + numbers[right - 1] < target:
            left += 1

class TwoSum:

    def __init__(self):
        self.arr = []
        self.vals = set()
        self.doubles = set()

    def add(self, number: int) -> None:
        self.arr.append(number)
        if number in self.vals:
            self.doubles.add(number + number)
        else:
            self.vals.add(number)

    def find(self, value: int) -> bool:
        if value in self.doubles:
            return True
        
        for val in self.arr:
            if value - val == val:
                continue
            elif value - val in self.vals:
                return True
            
        return False


def twoSumBT(self, root, k):
    if root == None:
        return False
    values = set()
    def inOrderRecurse(node):
        if k - node.val in values:
            return True
        else:
            values.add(node.val)
        
        if node.left == None and node.right == None:
            return False
        elif node.left == None:
            return inOrderRecurse(node.right)
        elif node.right == None:
            return inOrderRecurse(node.left)
        else:
            return inOrderRecurse(node.right) or inOrderRecurse(node.left)

    return inOrderRecurse(root)


def threeSum(self, nums):
    nums.sort()
    #seen = set()
    out = []
    for l in range(0, len(nums)):
        if l != 0 and nums[l] == nums[l-1]:
            continue
        
        lo = l + 1
        hi = len(nums) - 1
        while lo < hi:
            if lo != l+1 and nums[lo] == nums[lo - 1]:
                lo += 1
                continue
            
            if nums[lo] + nums[hi] < -1* nums[l]:
                lo += 1
            elif nums[lo] + nums[hi] > -1*nums[l]:
                hi -= 1
            else:
                # if (nums[l], nums[lo]) not in seen:
                #     seen.add((nums[l], nums[lo]))
                out.append([nums[l], nums[lo], nums[hi]])  
                lo += 1
                
    return out

def findSubArraysWithEqualSum(self, nums):
    seen = set()
    for i in range (0, len(nums)):
        if i + 1 >= len(nums):
            continue
        
        if nums[i] + nums[i+1] not in seen:
            seen.add(nums[i] + nums[i+1])
        else:
            return True
    
    return False


def letterCombinations(self, digits):
    def generateWords(curr_str, ind, digits, outputs):
        if ind == len(digits):
            if curr_str != "":
                outputs.append(curr_str)
            return

        if int(digits[ind]) == 2:
            generateWords(curr_str + "a", ind + 1, digits, outputs)
            generateWords(curr_str + "b", ind + 1, digits, outputs)
            generateWords(curr_str + "c", ind + 1, digits, outputs)
        elif int(digits[ind]) == 3:
            generateWords(curr_str + "d", ind + 1, digits, outputs)
            generateWords(curr_str + "e", ind + 1, digits, outputs)
            generateWords(curr_str + "f", ind + 1, digits, outputs)
        elif int(digits[ind]) == 4:
            generateWords(curr_str + "g", ind + 1, digits, outputs)
            generateWords(curr_str + "h", ind + 1, digits, outputs)
            generateWords(curr_str + "i", ind + 1, digits, outputs)
        elif int(digits[ind]) == 5:
            generateWords(curr_str + "j", ind + 1, digits, outputs)
            generateWords(curr_str + "k", ind + 1, digits, outputs)
            generateWords(curr_str + "l", ind + 1, digits, outputs)
        elif int(digits[ind]) == 6:
            generateWords(curr_str + "m", ind + 1, digits, outputs)
            generateWords(curr_str + "n", ind + 1, digits, outputs)
            generateWords(curr_str + "o", ind + 1, digits, outputs)
        elif int(digits[ind]) == 7:
            generateWords(curr_str + "p", ind + 1, digits, outputs)
            generateWords(curr_str + "q", ind + 1, digits, outputs)
            generateWords(curr_str + "r", ind + 1, digits, outputs)
            generateWords(curr_str + "s", ind + 1, digits, outputs)
        elif int(digits[ind]) == 8:
            generateWords(curr_str + "t", ind + 1, digits, outputs)
            generateWords(curr_str + "u", ind + 1, digits, outputs)
            generateWords(curr_str + "v", ind + 1, digits, outputs)
        elif int(digits[ind]) == 9:
            generateWords(curr_str + "w", ind + 1, digits, outputs)
            generateWords(curr_str + "x", ind + 1, digits, outputs)
            generateWords(curr_str + "y", ind + 1, digits, outputs)
            generateWords(curr_str + "z", ind + 1, digits, outputs)

    out = []
    curr_str = ""
    generateWords(curr_str, 0, digits, out)
    return out

def generateParenthesis(n):
    out = []


def binarySearch(nums, target):
    lo = 0
    hi = len(nums)
    while lo <= hi:
        mid = (lo + hi) // 2 + 1
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    
    return -1




import queue

class peekingQueue(queue.Queue):
    def peek(self):
        with self.mutex:
            return self.queue[0]

class HitCounter:

    def __init__(self):
        self.hits = peekingQueue()
        self.duration = 5 * 60 # 5 minutes

    def hit(self, timestamp: int) -> None:
        self.hits.put(timestamp)

    def getHits(self, timestamp: int) -> int:
        start = timestamp - self.duration + 1
        while True:
            if self.hits.qsize() == 0:
                break
            
            if self.hits.peek() < start:
                self.hits.get()
            else:
                break
        
        return self.hits.qsize()

from collections import OrderedDict
class LRUCache:
    """
    Requirements:
        1. be able to cache the values of certain keys and retrive these quickly (constant time)
        2. update or put values very quickly
        3. boot out the oldest key (move to front heuristic)

    Implementation:
        1. dict
        2. key in dict
        3. linked list? array (O(n))?
    """
    class DoublyNode:
        def __init__(self, key, val) -> None:
            self.value = val
            self.key = key

            self.next = None
            self.prev = None


    def __init__(self, capacity: int):
        self.stored = dict()
        self.list_front = None
        self.list_back = None
        self.capacity = capacity
        self.list_len = 0


    def get(self, key: int) -> int:
        self.stored.get(key)
        if key in self.stored:
            temp = self.stored.pop(key)
            self.stored[key] = temp
            return self.stored[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.stored:
            self.stored.pop(key)
            self.stored[key] = value
        else:
            if len(self.stored) == self.capacity:
                self.stored.popitem(last = False)

            self.stored[key] = value    


class Node(object):
    def __init__(self, val=None, children=[]):
        self.val = val
        self.children = children
class Codec:

    def serialize(self, root: Node) -> str:
        import queue
        """Encodes a tree to a single string.
        
        :type root: Node
        :rtype: str
        """
        to_visit = queue.Queue()
        output = []
        to_visit.put(root)

        while True:
            if to_visit.qsize() == 0:
                return " ".join(output)
            
            n = to_visit.get()
            if n == None:
                output.append(-1)
            else:
                output.append(n.val)
                to_visit.put(output.left)
                to_visit.put(output.right)

    # def recursive_build(node):
    #     if node == None:
    #         return None
        

	
    def deserialize(self, data: str) -> Node:
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: Node
        """
        nodes = data.split(" ")
        if len(nodes) == 1 and nodes[0] == -1:
            return None

        building_nodes_q = queue.Queue()
        root = Node(val = nodes[0])
        building_nodes_q.put(nodes[0])
        index = 0

        while building_nodes_q.qsize() != 0:
            n = building_nodes_q.get()
            if n.val == -1:
                continue

            left = Node(nodes[index+1])
            right = Node(nodes[index+2])

            n.children()




            

            
                

        return root



        