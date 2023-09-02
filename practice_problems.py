"""

CodeSignal Practice Problems!

"""
from typing import List, Optional
import typing

from distutils.command.build import build
import math
from tarfile import _Bz2ReadableFileobj
from xmlrpc.server import list_public_methods

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

class DoublyNode:
        def __init__(self) -> None:
            self.value = None
            self.key = None

            self.next = None
            self.prev = None
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


    def __init__(self, capacity: int):
        self.stored = dict() # stores key -> doublynode
        self.list_front = DoublyNode()
        self.list_back = DoublyNode()
        self.list_front.next = self.list_back
        self.list_back.prev = self.list_front

        self.capacity = capacity
        self.list_len = 0


    def get(self, key: int) -> int:
        if key in self.stored:
            # print(self.stored[key].prev.key)
            p = self.stored[key].prev
            n = self.stored[key].next
            p.next = n
            n.prev = p
            self.stored[key].next = self.list_front.next
            self.stored[key].next.prev = self.stored[key]
            self.stored[key].prev = self.list_front
            self.list_front.next = self.stored[key]
            return self.stored[key].value
        else:
            return -1
        

    def put(self, key: int, value: int) -> None:
        if key not in self.stored:
            self.list_len += 1
            new_node = DoublyNode()
            new_node.value = value
            new_node.key = key
            new_node.next = self.list_front.next
            self.list_front.next = new_node
            new_node.prev = self.list_front
            new_node.next.prev = new_node

            self.stored[key] = new_node

            if self.list_len > self.capacity:
                remove_key = self.list_back.prev.key
                self.list_back.prev = self.list_back.prev.prev
                self.list_back.prev.next = self.list_back
                self.list_len -= 1
                del self.stored[remove_key]

        else:
            self.stored[key].prev.next = self.stored[key].next
            self.stored[key].next.prev = self.stored[key].prev
            self.stored[key].next = self.list_front.next
            self.list_front.next = self.stored[key]
            self.stored[key].prev = self.list_front
            self.stored[key].next.prev = self.stored[key]
            self.stored[key].value = value
            




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

            n.children
            

            
                

        return root



class Solution:
    def isLetterLog(self, log: List) -> bool:
        words = log.split(" ")
        if  (words[1][0] >= "a" and words[1][0] <= "z") or (words[1][0] >= "A" and words[1][0] <= "Z"):
            return True
        
        return False

    def getContents(self, log: List) -> List:
        words = log.split(" ")

        return " ".join(words[1:len(words)])
    
    def getIdentifier(self, log: List) -> List:
        return log.split(" ")[0]

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        
        count_letters = 0
        count_digits = 0
        
        digits = []
        letters = []
            
        for i in range(0, len(logs)):
            if not self.isLetterLog(logs[i]):
                digits.append(logs[i])
            else:
                letters.append(logs[i])
    
        first_digit = len(letters)
                
        logs[0:first_digit] = sorted(letters[0:first_digit], key = lambda x: (self.getContents(x), self.getIdentifier(x)))

        logs[first_digit: len(logs)] = digits

        return logs

class Solution:
    def recurseParen(self, n: int, curr_string: str, count_opens: int, outputs: list):
        if n == 0:
            for _ in range(count_opens):
                curr_string += ")"
                
            outputs.append(curr_string)
        
        else:
            self.recurseParen(n-1, curr_string + "(", count_opens + 1, outputs)
            
            if count_opens != 0:
                self.recurseParen(n, curr_string + ")", count_opens - 1, outputs)
    
    
    def generateParenthesis(self, n: int) -> List[str]:
        output = []
        
        self.recurseParen(n, "", 0, output)
        
        return output

class Solution:
    def romanToInt(self, s: str) -> int:
        total = 0
        mapped_vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

        i = 0
        while i < len(s):
            if i == len(s) - 1:
                total += mapped_vals[s[i]]
                i += 1
            elif s[i] == "I" and (s[i+1] == "V" or s[i+1] == "X"):
                total += mapped_vals[s[i+1]]
                total -= 1
                i += 2
            elif s[i] == "X" and (s[i+1] == "L" or s[i+1] == "C"):
                total += mapped_vals[s[i+1]]
                total -= 10
                i += 2 
            elif s[i] == "C" and (s[i+1] == "D" or s[i+1] == "M"):
                total += mapped_vals[s[i+1]]
                total -= 100
                i += 2 
            else:
                total += mapped_vals[s[i]]
                i += 1
        
        return total

class Solution:
    def findCombinations(self, candidates: List, curr_candidates: List, remaining: int, output: List,
                        found: set(), start: int) -> None:
        if remaining < 0:
            return
        elif remaining == 0:
            to_insert = sorted(curr_candidates)
            if tuple(to_insert) not in found:
                output.append(to_insert)
                found.add(tuple(to_insert))
        else:
            for i in range(start, len(candidates)):
                if len(curr_candidates) == 0:
                    self.findCombinations(candidates, [candidates[i]], remaining - candidates[i], output, found, i)
                else:
                    curr_candidates.append(candidates[i])
                    self.findCombinations(candidates, curr_candidates, remaining - candidates[i], output, found, i)
                    curr_candidates.pop()
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        output = []
        curr_candidate = []
        found = set()
        self.findCombinations(candidates, curr_candidate, target, output, found, 0)
        
        return output

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        total = 0
        curr_max = -1e4 - 1
        
        for r in range(len(nums)):
            if total > 0:
                total += nums[r]
                curr_max = max(curr_max, total)
            else:
                total = nums[r]
                curr_max = max(curr_max, total)
        
        return curr_max

class Solution:
    """
    intuition 1:
    - we can either choose to take or not take each element in nums
    - this generates 2^n possibilities where n is at most 10
    - 10 * 2 ^ 10 32 * 32 = 16 * 16 * 4 = 1024   2^n
    
    - 2^n combinations
    - n bits to store this information
    - [] -> 000
    - [1] -> 001
    - [2] -> 010
    
    - 111 -> 7 

    
    """
#     def findSubsets(self, nums: List[int], start, curr: List[int], output: List[List[int]]):
#         to_insert = curr.copy()
#         output.append(to_insert)
        
        
#         for i in range(start, len(nums)):
#             curr.append(nums[i])
#             self.findSubsets(nums, i + 1, curr, output)
#             curr.pop()
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        output = []
        
        # self.findSubsets(nums, 0, [], output)
        
        for i in range(0, 2**len(nums)):
            arr = []
            for j in range(0, len(nums)):
                if (i >> j) % 2 == 1:
                    arr.append(nums[j])
            
            output.append(arr)
        
        return output


class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.it = iterator
        self.peeked_val = None
        self.next_exists = self.it.hasNext()
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if not self.peeked_val:
            self.peeked_val = self.it.next()
        
        return self.peeked_val
        

    def next(self):
        """
        :rtype: int
        """
        if self.peeked_val != None:
            self.next_exists = self.it.hasNext()
            ret_val = self.peeked_val
            self.peeked_val = None
            return ret_val
        else:
            ret_val = self.it.next()
            self.next_exists = self.it.hasNext()
            return ret_val
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.next_exists


class Solution:
    """
    amount = 11
    coins = 1 2 5

    0 1 2 3 4 5 6 7 8 9 10 11
    0 1 1 0 0 1 0 0 0 0 0  0
    0 1 1 

    """

    def coinChange(self, coins: List[int], amount: int) -> int:
        minimums = [None] * (len(amount) + 1)
        for i in range(len(minimums)):
            for coin_val in coins:
                minimums[i + coin_val] = min(minimums[i + coin_val], minimums[i] + 1)
        
        return -1 if minimums[-1] is None else minimums[-1]



class Solution:
    """
    0 1 2 3 4 5

    1: 1
    2: 1
    3: 1, 2
    4: 1, 2
    5: 
    """

    def canCross(self, stones: List[int]) -> bool:
        if stones[1] != 1:
            return False
    
        reachable = [0] * len(stones)
        reachable[0] = 1
        reachable[1] = 1
        valid_stones = set(stones)
        prev_jumps = dict() # stores stone value -> set of jumps leading to stone
        prev_jumps[1] = set()
        prev_jumps[1].add(1)

        # for i in range(0, len(stones)):
        #     stones_to_index[stones[i]] = i

        for stone in stones:
            if stone not in valid_stones: # this is the water
                continue
            elif stone not in prev_jumps: # this is not a valid stone
                continue
            
            for jump in prev_jumps[stone]:
                if stone + jump - 1 in valid_stones and jump - 1 != 0:
                    if stone + jump - 1 not in prev_jumps:
                        prev_jumps[stone + jump - 1] = set()
                    prev_jumps[stone + jump - 1].add(jump - 1)
                
                if stone + jump in valid_stones:
                    if stone + jump not in prev_jumps:
                        prev_jumps[stone + jump] = set()
                    prev_jumps[stone + jump].add(jump)

                if stone + jump + 1 in valid_stones:
                    if stone + jump + 1 not in prev_jumps:
                        prev_jumps[stone + jump + 1] = set()
                    prev_jumps[stone + jump + 1].add(jump + 1)

        return stones[-1] in prev_jumps 


"""
babad

bbcd



"""
class Solution:
    def longestPalindrome(self, s: str) -> str:
        out = ""         
        for i in range(0, len(s)):
            l1, r1 = self.expandOut(s, i, i)
            l2, r2 = self.expandOut(s, i, i + 1)
            if (max(r1 - l1, r2 - l2) > len(out)):
                if r1 - l1 > r2 - l2:
                    out = s[l1 : r1 + 1]
                else:
                    out = s[l2: r2 + 1]
        
        return out

    def expandOut(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        
        return l, r

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def recurse_tree(self, node: TreeNode):
        if not node:
            return True, 0
        
        left_balanced, left_height = self.recurse_tree(node.left)
        right_balanced, right_height = self.recurse_tree(node.right)

        if left_balanced and right_balanced and abs(left_height - right_height) <= 1:
            return 0
            

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        balanced, height = self.recurse_tree(root)
        return balanced
    


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums
        self.output = None

    def add(self, val: int) -> int:
        # since we are guaranteed at leas k elements in array upon search for kth element
        # we can then directly compute kth element
        # method 1: calculate each time
        # method 2: if already cached, simply do some logic to output the new kth largest
        self.nums.append(val)
        
        self.nums = sorted(self.nums)

        for i in range(len(self.nums)):
            if i == self.k:
                self.output = self.nums[i]
                return self.nums[i]
    


def first_duplicate(a):
    vals = set()
    for val in a:
        if val not in vals:
            vals.add(val)
        else:
            return val
            
    return -1

def first_not_repeating(s):
    unique = dict()
    repeat = set()
    for ind in range(len(s)):
        if s[ind] in repeat:
            continue
        elif s[ind] in unique:
            unique.pop(s[ind])
            repeat.add(s[ind])
        else:
            unique[s[ind]] = ind
    
    min_ind = 10e5
    out = "_"
    for c, ind in unique.items():
        if ind < min_ind:
            out = c
            min_ind = ind
    
    return out


def first_not_repeating_2(s):
    
    # backwards iterate through s and find last occurences of chars
    last_seen = dict()
    for ind in range(len(s) - 1, -1, -1):
        if s[ind] not in last_seen:
            last_seen[s[ind]] = ind
        else:
            last_seen[s[ind]] = -1
    
    # go forwards and see if we have
    for ind in range(len(s)):
        if last_seen[s[ind]] == ind:
            return s[ind]
            
    return "_"
