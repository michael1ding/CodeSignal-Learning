"""

CodeSignal Practice Problems!

"""


import math

class ListNode:
    def __init__(self, x):
        self.value = x
        self.next = None


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
            l = l.next
            temp = temp.next
    
    return True
    
def reverseLL(l):
    temp = l.next
    l.next = None
    
    while temp:
        place_holder = temp.next
        temp.next = l
        l = temp
        temp = place_holder
        
    return l
        
def findLength(l):
    length = 0
    while l:
        length += 1
        l = l.next
        
    return length

def addTwoHugeNumbers(a, b):
    """
    1. reverse a and b
    2. add with carries
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