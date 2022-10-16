"""

CodeSignal Practice Problems!

"""


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
