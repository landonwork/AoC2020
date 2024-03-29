3# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:34:20 2020

@author: lando
"""

# Advent of Code
# Day 1

# Reading the file and cleaning the data
file_reader = open('Day 1.txt','r')
nums = file_reader.readlines()
for i in range(len(nums)):
    nums[i] = int(nums[i].replace('\n',''))

# Part 1
for i in range(len(nums)):
    for j in range(i,len(nums)):
        if nums[i]+nums[j] == 2020:
            print(nums[i]*nums[j])
            
# Part 2
for i in range(len(nums)):
    for j in range(i+1,len(nums)):
        for k in range(j+1,len(nums)):
            if nums[i]+nums[j]+nums[k] == 2020:
                print(nums[i]*nums[j]*nums[k])

# Day 2
# Reading the file and cleaning the data
file_reader = open('Day 2.txt','r')
lines = file_reader.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].replace('\n','')

import re

# Part 1
def break_apart(text):
    dissolve = re.compile('(?P<lo>\d*)-(?P<hi>\d*)\s(?P<letter>\w):\s(?P<passcode>\w*)')
    parts = dissolve.search(text)
    return (int(parts.group('lo')), int(parts.group('hi')), parts.group('letter'), parts.group('passcode'))

def is_correct(lo,hi,letter,passcode):
    count = 0
    for l in passcode:
        count += (l == letter)
    return (lo <= count) & (count <= hi)

count = 0
for line in lines:
    lo, hi, letter, passcode = break_apart(line)
    result = is_correct(lo,hi,letter,passcode)
    print(line, result)
    count += result
print(count)

# Part 2
def is_valid(lo,hi,letter,passcode):
    if hi-1 >= len(passcode):
        return False
    else:
        return (passcode[lo-1] == letter) ^ (passcode[hi-1] == letter)

count = 0
for line in lines:
    lo, hi, letter, passcode = break_apart(line)
    count += is_valid(lo,hi,letter,passcode)
    
print(count)

# Day 3
# Reading the file and cleaning the data
file_reader = open('Day 3.txt','r')
lines = file_reader.readlines()
for i in range(len(lines)):
    lines[i] = list(lines[i].replace('\n',''))
    for j in range(len(lines[i])):
        lines[i][j] = 0 if lines[i][j] == '.' else 1

# Part 1
import numpy as np
grid = np.array(lines)
slope_x = 1
slope_y = 2
x, y = slope_x, slope_y

count = 0
while y < 323:
    count += grid[y,x]
    x = (x+slope_x)%31
    y += slope_y
print(count)

86*232*90*71*31

# Day 4
# Reading the file and cleaning the data
file_reader = open('Day 4.txt','r')
lines = file_reader.readlines()
file_reader.close()
passports = []
info = ''
for line in lines:
    if line == '\n':
        passports.append(info)
        info = ''
    else:
        info = info + line
passports.append(info)

#Part 1
import re
def get_keys(passport:str) -> list:
    keys = []
    index = 0
    while True:
        grab = re.compile('(?P<key>\w{3}):.*?\s?')
        key_match = grab.search(passport,index)
        if key_match is None:
            break
        else:
            keys.append(key_match.group('key'))
            index = key_match.span()[1]
    return set(keys)

def check_keys(keys:set) -> bool:
    check_set = {'byr','iyr','eyr','hgt','hcl','ecl','pid'}
    # print(check_set.difference(keys))
    return check_set.issubset(keys)

count = 0
for i, passport in enumerate(passports):
    keys = get_keys(passport)
    count += check_keys(keys)
print(count) 

# Part 2
def get_info(passport:str) -> dict:
    check_set = ['byr','iyr','eyr','hgt','hcl','ecl','pid']
    reqs = ['\d*','\d*','\d*','\d{2,3}((in)|(cm))','\#[0-9a-f]*',
            '((amb)|(blu)|(brn)|(gry)|(grn)|(hzl)|(oth))\s','\d*']
    info = {}
    for i, key in enumerate(check_set):
        # print(key,reqs[i])
        grab = re.compile('(?P<key>'+key+'):(?P<value>'+reqs[i]+')\s?')
        key_match = grab.search(passport)
        if key_match is None:
            continue
        else:
            info.update({key_match.group('key'):key_match.group('value')})
    return info

def validate_info(info:dict) -> bool:
    if check_keys(set(info.keys())):
        years = [('byr',1920,2002),('iyr',2010,2020),('eyr',2020,2030)]
        so_far = True
        for tup in years:
            so_far = so_far & between(int(info[tup[0]]),tup[1],tup[2])
            if not so_far:
                print('Invalid ' + tup[0],info['byr'])
                return False
        heights = {'in':(59,76),'cm':(150,193)}
        if not between(int(info['hgt'][:-2]),heights[info['hgt'][-2:]][0],heights[info['hgt'][-2:]][1]):
            print('Invalid height',info['hgt'])
            return False
        return True
        if len(info['hcl']) != 7:
            print('Invalid hcl',info['hcl'])
            return False
        if len(info['pid']) != 9:
            print('Invalid pid',info['pid'])
            return False
        return True
    else:
        print('Missing keys')
        return False
    
import numpy as np
import pandas as pd
def checklist(info:dict) -> np.ndarray:
    arr = np.zeros(7)
    check_set = ['byr','iyr','eyr','hgt','hcl','ecl','pid']
    for i, name in enumerate(check_set):
        arr[i] = 0 if info.get(name) is None else 1
    return arr

def between(value,lo,hi):
    return (lo <= value) & (value <= hi)

df = pd.DataFrame([get_info(p) for p in passports])
df = df.dropna(axis='rows')

for row in df.index:
    if not between(int(df['byr'].loc[row]),1920,2002):
        df = df.drop(row,axis='index')
        continue
    if not between(int(df['iyr'].loc[row]),2010,2020):
        df = df.drop(row,axis='index')
        continue
    if not between(int(df['eyr'].loc[row]),2020,2030):
        df = df.drop(row,axis='index')
        continue
    hgts = {'cm':(150,193),'in':(59,76)}
    h = int(df['hgt'].loc[row][:-2])
    unit = df['hgt'].loc[row][-2:]
    if not between(h,hgts[unit][0],hgts[unit][1]):
        df = df.drop(row,axis='index')
        continue
    if len(df['hcl'].loc[row]) != 7:
        df = df.drop(row,axis='index')
        continue
    # print(len(df['ecl'].loc[row]))
    if len(df['pid'].loc[row]) != 9:
        df = df.drop(row,axis='index')
        continue

print(len(df))

# Day 5
# Reading the file and cleaning the data
file_reader = open('Day 5.txt','r')
lines = file_reader.readlines()
for i in range(len(lines)):
    lines[i] = list(lines[i].replace('\n','').replace('B','1').replace('F','0').replace('R','1').replace('L','0'))

import numpy as np
arr = np.array(lines).astype(int)

# Part 1
front_to_back = np.zeros(len(lines))
for i in range(7):
    front_to_back = front_to_back + (2**(6-i))*arr[:,i]
left_to_right = np.zeros(len(lines))
for i in range(7,10):
    left_to_right = left_to_right + (2**(9-i))*arr[:,i]

ids = front_to_back * 8 + left_to_right
print(ids.max())

# Part 2
for i in range(ids.shape[0]-1):
    if ids[i+1] == ids[i] + 2:
        print(ids[i]+1)
        
# Day 6
# read and clean data
file_reader = open('Day 6.txt','r')
lines = file_reader.readlines()
groups = []
info = set()
for line in lines:
    if line == '\n':
        groups.append(info)
        info = set()
    else:
        info.update(set(line.replace('\n','')))
groups.append(info)

# Part 1
import numpy as np
count = np.array([len(i) for i in groups]).sum()

# Part 2
groups = []
li = []
for line in lines:
    if line == '\n':
        groups.append(li)
        li = []
    else:
        li.append(line.replace('\n',''))
groups.append(li)

arr = np.zeros(len(groups))    
for i, li in enumerate(groups):
    # print(i,len(li))
    if len(li) == 1:
        arr[i] = len(li[0])
    else:
        info = set(li[0])
        for j in range(1,len(li)):
            info = info.intersection(set(li[j]))
        arr[i] = len(info)
        
# Day 7
# This one's a doozie
# I will use a dict with other dicts as objects
# Each bag contains other bags. Their required contents are stored in a
# dictionary using the colors of the bags as keys
# Then I would use a list to hold all of the bags in memory
# Then comes solving the problem

# Reading and cleaning
file_reader = open('Day 7.txt','r')
lines = file_reader.readlines()
rules = {}

import re
get_bag = re.compile('(?P<color>\w*?\s\w*?) bags')
get_contents = re.compile('(?P<num>\d+?)\s(?P<color>\w*?\s\w*?) bag')

contents = {}
for i, line in enumerate(lines):
    bag = get_bag.match(line)
    color = bag.group('color')
    rules.update({color:{}})
    con = get_contents.search(line,bag.span()[1])
    while con is not None:
        # print(color,con.group('color'))
        # print(i)
        rules[color].update({con.group('color'):int(con.group('num'))})
        con = get_contents.search(line,con.span()[1])
        
def find_bag(rules:dict,color:str,target:str) -> bool:
    # First, we see if target is in color
    if rules.get(color).get(target) is not None:
        return True
    else: # If it's not in this one, we check all the other bags
        for key in rules.get(color).keys():
            if find_bag(rules,key,target):
                return True
        return False

count = 0
for rule in rules.keys():
    count += find_bag(rules,rule,'shiny gold')
print(count)

def count_bags(rules:dict,color:str):
    # Inside this bag is
    count = 0
    for bag in rules.get(color).keys():
        count += rules.get(color).get(bag)*(1+count_bags(rules,bag))
    return count

print(count_bags(rules,'shiny gold'))

# Day 8
# Read and clean data
file_reader = open('Day 8.txt','r')
lines = file_reader.readlines()

import re
commands = []
split = re.compile('(?P<command>((nop)|(acc)|(jmp)))\s(?P<num>[+-]\d+)')
for line in lines:
    match = split.match(line)
    commands.append([match.group('command'),int(match.group('num')),False])
    
# Part 1
def jmp(state,n):
    state.update({'i':state['i']+n})
def acc(state,n):
    state.update({'accum':state['accum']+n,'i':state['i']+1})
def nop(state,n):
    state.update({'i':state['i']+1})
state = {
    'accum':0,
    'i':0
    }    
defs = {
    'jmp':jmp,
    'acc':acc,
    'nop':nop
    }

FN = 0
N = 1
while not commands[state['i']][2]:
    commands[state['i']][2] = True
    fn = defs[commands[state['i']][FN]]
    fn(state,commands[state['i']][N])

print(state['accum'])

# Part 2
li = []
for i, item in enumerate(commands):
    if (item[0] == 'nop') | (item[0] == 'jmp'):
        li.append(i)
        
# Need to reinitialize the commands so they're "un-run"
def reinitialize(coms,state):
    for i in range(len(coms)):
        coms[i][2] = False
    state.update({'accum':0,'i':0})

for j in li:
    reinitialize(commands,state)
    while not commands[state['i']][2]:
        commands[state['i']][2] = True
        fn = defs[commands[state['i']][FN]]
        if state['i'] == j:
            if fn is jmp:
                fn = nop
            elif fn is nop:
                fn = jmp
            else:
                print('error')
        fn(state,commands[state['i']][N])
        if state['i'] == len(commands):
            print('Hurrah!')
            print(state['accum'])
            break

# Day 9
file_reader = open('Day 9.txt','r')
lines = file_reader.readlines()
nums = [int(line.replace('\n','')) for line in lines]

# Part 1
def combos(li:list) -> list:
    s = set()
    for i in range(len(li)):
        for j in range(i+1,len(li)):
            s.add(li[i]+li[j])
    return s

misfits = []
for i in range(25,len(nums)):
    lo = i-25
    # print(lo,i)
    preamble = nums[lo:i]
    if nums[i] not in combos(preamble):
        misfits.append(nums[i])
        
misfits[0]

# Part 2

import numpy as np

def search(target,li,depth):
    print('Depth:',depth)
    first = True
    for i in range(depth,len(li)+1):
        if li[i-1] >= target:
            return None
        lo = i-depth
        if np.sum(li[lo:i]) == target:
            return np.min(li[lo:i])+np.max(li[lo:i])
    return None

n = 1
ans = None
target = misfits[0]
while ans is None:
    n += 1
    ans = search(target,nums,n)
print(ans)

# Day 10
file_reader = open('Day 10.txt','r')
lines = file_reader.readlines()
nums = [int(line.replace('\n','')) for line in lines]

# Part 1
nums.sort()
nums.append(nums[-1]+3)
nums.insert(0,0)
# print(nums)

d = {1:0,2:0,3:0}
for i in range(1,len(nums)):
    print(i)
    diff = abs(nums[i-1] - nums[i])
    if diff > 3:
        print('Error')
        break
    # print(diff)
    d.update({diff:d[diff]+1})
print(d)

# Part 2 - Counting all different arrangement
# I need a way to put usable adapters into a list
# And I need a way to recurse paths and return the count of arrangements

# I can build a dictionary with the values of nums as keys and possible next
# adapters in a list of values for each key
# This should make it easy for the recursive function to parse possible arrangements

nexts = {}
for i, num in enumerate(nums):
    li = []
    for j in range(i+1,i+4):
        if j == len(nums):
            break
        if abs(num - nums[j]) <= 3:
            li.append(nums[j])
    nexts.update({num:li})
    
# So instead of recursing the entire thing (because there are literally
# trillions of arrangements), I counted backwards and recorded the number of
# arrangements possible from each point so I wouldn't have to do them over
# again.
def count_path(nexts,n):
    if n == 166:
        return 1
    if isinstance(nexts[n], int):
        return nexts[n]
    else:
        count = 0
        for num in nexts[n]:
            c = count_path(nexts,num)
            nexts.update({num:c})
            count += c
        return count
    
i = len(nexts.keys())-1
# count_path(nexts,list(nexts.keys())[-3])
while i >= 0:
    count_path(nexts,list(nexts.keys())[i])
    i -= 1
count_path(nexts,0)

# Day 11
file_reader = open('Day 11.txt','r')
lines = file_reader.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].replace('\n','')
    # lines[i] = list(lines[i])
grid = lines.copy()

# Part 1
# for row in range(len(grid)):
#     for col in range(len(grid[row])):
#         if grid[row][col] == 'L':
#             grid[row][col] = {'chair':True,'occupied':False}
#         elif grid[row][col] == '.':
#             grid[row][col] = {'chair':False,'occupied':False}
#         else:
#             print('error')

# grid = np.array(grid)

def get_neighbors(grid,row,col):
    height = len(grid)
    width = len(grid[0])
    # print(height,width)
    row_lo, row_hi = -1,2
    col_lo, col_hi = -1,2
    if row == 0:
        row_lo = 0
    elif row == height-1:
        row_hi = 1
    if col == 0:
        col_lo = 0
    elif col == width-1:
        col_hi = 1
    neighbors = 0
    for r in range(row_lo,row_hi):
        for c in range(col_lo,col_hi):
            # print(row,col)
            if (r == 0) & (c == 0):
                continue
            # if (col >= width):
            #     print(col,c)
            else:
                neighbors += (grid[row+r][col+c] == '#')
    return neighbors

def change_state(grid):
    new_grid = []
    altered = False
    for row in range(len(grid)):
        line = ''
        for col in range(len(grid[row])):
            if grid[row][col] == '.':
                line = line + '.'
                continue
            else:
                num_occupied = get_neighbors(grid,row,col)
                if (grid[row][col] == '#') & (num_occupied >= 4):
                    line = line + 'L'
                    altered = True
                elif (grid[row][col] == 'L') & (num_occupied == 0):
                    line = line + '#'
                    altered = True
                else:
                    line = line + grid[row][col]
        new_grid.append(line)
    return new_grid, altered

t = 0
altered = True
while altered:
    print(t, altered)
    grid, altered = change_state(grid)
    t += 1
    
count = 0
for row in grid:
    for col in row:
        if col == '#':
            count += 1
print(count)

# Part 2 - requires modification to the prior functions

def get_neighbors(grid,row,col):
    height = len(grid)
    width = len(grid[0])
    dirs = [(1,0),(0,1),(-1,0),(0,-1),(-1,1),(1,1),(1,-1),(-1,-1)]
    neighbors = 0
    for i in range(len(dirs)):
        neighbors += sight(grid,row,col,dirs[i])
    return neighbors

def sight(grid,row,col,tup):
    n = 1
    height = len(grid)
    width = len(grid[0])
    while True:
        r = tup[0]*n
        c = tup[1]*n
        bound = (row+r == -1) | (row+r == height) | (col+c == -1) | (col+c == width)
        if bound:
            return False
        else:
            sq = grid[row+r][col+c]
            if sq == 'L':
                return False
            elif sq == '#':
                return True
            elif sq == '.':
                n += 1
                continue
            else:
                print('Incorrect character')
                break

def change_state(grid):
    new_grid = []
    altered = False
    for row in range(len(grid)):
        line = ''
        for col in range(len(grid[row])):
            if grid[row][col] == '.':
                line = line + '.'
                continue
            else:
                num_occupied = get_neighbors(grid,row,col)
                if (grid[row][col] == '#') & (num_occupied >= 5):
                    line = line + 'L'
                    altered = True
                elif (grid[row][col] == 'L') & (num_occupied == 0):
                    line = line + '#'
                    altered = True
                else:
                    line = line + grid[row][col]
        new_grid.append(line)
    return new_grid, altered

grid = lines.copy()
t = 0
altered = True
while altered:
    # print(t, altered)
    grid, altered = change_state(grid)
    t += 1
    
count = 0
for row in grid:
    for col in row:
        if col == '#':
            count += 1
print(count)

# Day 12
# Read and clean
file_reader = open('Day 12.txt','r')
lines = file_reader.readlines()

# import re
for i in range(len(lines)):
    lines[i] = lines[i].replace('\n','')

# Part 1
# Initialize the ship
ship = {'dir': 1,'x':0,'y':0}

# Define the movement of the ship
def move_ship(ship,text):
    c_moves = {'N':(0,1),'E':(1,0),'S':(0,-1),'W':(-1,0)}
    if text[0] in c_moves.keys():
        x, y = c_moves[text[0]]
        n = int(text[1:])
        ship.update({'x':ship['x']+x*n,'y':ship['y']+y*n})
    elif (text[0] == 'R') | (text[0] == 'L'):
        rot = int(text[1:])/90
        if text[0] == 'L':
            rot *= -1
        ship.update({'dir':int((ship['dir']+rot) % 4)})
    elif text[0] == 'F':
        x, y = c_moves.get(list(c_moves.keys())[ship['dir']])
        n = int(text[1:])
        ship.update({'x':ship['x']+x*n,'y':ship['y']+y*n})
    else:
        print('Error: incorrect first letter')
        
for line in lines:
    move_ship(ship,line)

print(abs(ship['x']) + abs(ship['y']))

# Part 2
ship = {'x':0,'y':0}
way = {'x':10,'y':1}

# Define the movement of the ship
def move_ship_way(ship,way,text):
    c_moves = {'N':(0,1),'E':(1,0),'S':(0,-1),'W':(-1,0)}
    if text[0] in c_moves.keys():
        x, y = c_moves[text[0]]
        n = int(text[1:])
        way.update({'x':way['x']+x*n,'y':way['y']+y*n})
    elif (text[0] == 'R') | (text[0] == 'L'):
        for i in range(int(int(text[1:])/90)):
            way.update({'x':way['y'],'y':-way['x']})
            if text[0] == 'L':
                way.update({'x':-way['x'],'y':-way['y']})
    elif text[0] == 'F':
        x, y = way['x'], way['y']
        n = int(text[1:])
        ship.update({'x':ship['x']+x*n,'y':ship['y']+y*n})
    else:
        print('Error: incorrect first letter')
        
for line in lines:
    print(ship,way)
    input(line)
    move_ship_way(ship,way,line)

print(abs(ship['x']) + abs(ship['y']))

# Day 13
# Read and clean
file_reader = open('Day 13.txt','r')
lines = file_reader.readlines()
timestamp = int(lines[0].replace('\n',''))
import re
mnum = re.compile('(?P<num>(\d+|x))')
buses = []
m = mnum.search(lines[1],0)
li = [m.group('num')]
count = 0
while m is not None:
    m = mnum.search(lines[1],m.span()[1])
    if m is None:
        break
    if m.group('num') == 'x':
        count += 1
    else:
        li.append(count)
        buses.append(li)
        li = [m.group('num')]
        count = 0
buses.append([lines[1][-3:-1],0])
import numpy as np
buses = np.array(buses).astype(int)

        
# Part 1
wait_time = [abs((timestamp % buses[i,0])-buses[i,0]) for i in range(len(buses))]
ind = wait_time.index(min(wait_time))
print(buses[ind,0]*wait_time[ind])

# Part 2 - Brute force takes way too long

    # turn_knob:
        # buses and coefs
        # tumbler loop counter and a coef variable
            # tumbler starts at zero, goes to len(buses-2) (because the last tumbler doesn't matter')
            # then inside the tumbler loop, we loop through possible values for the coef
    # check_knob
        # I need the current and past tumblers and the next tumbler and the time gap between the two
        # I will also need a recursive function to check the value correctly
        # I guess I'll call it tumbler_value

buses[:,1] += 1
# buses = np.array([[67,1],[7,1],[59,1],[61,0]])
coefs = np.zeros(buses[:,0].shape)

def turn_knob(buses,coefs) -> int:
    for tumbler in range(len(buses)-1):
        check = False
        while not check:
            check = check_tumblers(buses[:1+tumbler,:],coefs[:1+tumbler],buses[tumbler+1,0])
            if not check:
                coefs[tumbler] += 1
        print(coefs[tumbler])
    return coefs

def check_tumblers(tumblers, coefs, nxt) -> bool:
    return (tumbler_value(tumblers[:,0],coefs) + tumblers[:,1].sum()) % nxt == 0

def tumbler_value(tumblers,coefs) -> int:
    if len(tumblers) == 1:
        return tumblers[0]*coefs[0]
    else:
        return tumblers[0]*(coefs[0]+tumbler_value(tumblers[1:],coefs[1:]))

coefs = turn_knob(buses,coefs)
print(tumbler_value(buses[:,0],coefs))

# Day 14
# Holy cow these problems get involved

# Read and clean
file_reader = open('Day 14.txt','r')
lines = file_reader.readlines()

# Part 1
import re
find_nums = re.compile('mem\[(?P<address>\d+)\] = (?P<num>\d+)\n') # Check
find_mask = re.compile('mask = (?P<mask>[01X]+)\n')

def concat(li:list) -> str:
    text = ''
    for s in li:
        text = text + s
    return text

def to_binary(num:int, mask:str = None) -> str:
    text = bin(num)[2:]
    leading = '0'*(36-len(text))
    b = leading + text
    if mask is not None:
        b = list(b)
        mask = list(mask)
        # if len(mask) != len(b):
        #     print('Error')
        for i in range(len(mask)):
            if mask[i] != 'X':
                b[i] = mask[i]
        b = concat(b)
    return b

def to_decimal(num:str) -> int:
    return int(num,2)

mem = {}
for line in lines:
    if line.startswith('mask'):
        mask = find_mask.search(line).group('mask')
    else:
        address = find_nums.search(line).group('address')
        num = find_nums.search(line).group('num')
        num = to_decimal(to_binary(int(num),mask))
        mem.update({address:num})

print(sum(mem.values()))

# Part 2
def to_binary_2(num:int, mask:str = None) -> str:
    text = bin(num)[2:]
    leading = '0'*(36-len(text))
    b = leading + text
    if mask is not None:
        b = list(b)
        mask = list(mask)
    #     if len(mask) != len(b):
    #         print('Error')
        for i in range(len(mask)):
            if mask[i] != '0':
                b[i] = mask[i]
        b = concat(b)
    return b

def floating(bits:str,n,li):
    b = list(bits)
    arr = bin(n)[2:]
    arr = '0'*(bits.count('X')-len(arr)) + arr
    # input(arr)
    count = 0
    for i in range(len(b)):
        if b[i] == 'X':
            b[i] = arr[count]
            count += 1
    li.append(concat(b))
    return li if n == (2**bits.count('X')-1) else floating(bits,n+1,li)

mem = {}
for line in lines:
    if line.startswith('mask'):
        mask = find_mask.search(line).group('mask')
    else:
        address = find_nums.search(line).group('address')
        num = find_nums.search(line).group('num')
        address = to_binary_2(int(address),mask)
        addresses = floating(address,0,[])
        for address in addresses:
            mem.update({address:int(num)})
            
print(sum(mem.values()))

# Day 15
# Nothing to clean
starters = [0,6,1,7,2,19,20]

# Parts 1 and 2

# Part 2
# rounds = 2020
rounds = 3*10**7

def play_game(starters: list, rounds: int) -> int:
    
    lasts = {}
    n = 1
    for num in starters:
        lasts.update({num:n})
        n += 1
    # print(lasts)
    def play_round(prev,current):
        lasts.update({prev:n-1})
        nxt = lasts.get(current) # The last time the current number was spoken
        if nxt == None:
            nxt = 0
        else:
            nxt = n - nxt
        return current, nxt
    
    current = starters[-1]
    nxt = 0
    while n <= rounds:
        current, nxt = play_round(current,nxt)
        n += 1
    return current

ans = play_game(starters,rounds)

# Day 16
# Read and clean
file_reader = open('Day 16.txt','r')
lines = file_reader.readlines()
fields = []
my_ticket = []
nearby = []
li = fields
for i in range(len(lines)):
    lines[i] = lines[i].replace('\n','')
    if lines[i] == '':
        continue
    if lines[i].startswith('your'):
        li = my_ticket
        continue
    if lines[i].startswith('nearby'):
        li = nearby
        continue
    li.append(lines[i])
my_ticket = my_ticket[0]

my_ticket = my_ticket.split(',')
nearby = [ticket.split(',') for ticket in nearby]
import numpy as np
my_ticket = np.array(my_ticket).astype(int)
nearby = np.array(nearby).astype(int)

import re
dfields = {}
get_fields = re.compile('(?P<field>[\w\s]+):\s(?P<range_1>\d+-\d+)\sor\s(?P<range_2>\d+-\d+)')
for line in fields:
    m = get_fields.match(line)
    dfields.update({m.group('field'):[m.group('range_1'),m.group('range_2')]})

# Part 1
def make_range(s:str) -> set:
    li = s.split('-')
    # print(li[0],li[1])
    return set(range(int(li[0]),int(li[1])+1))

def list_union(s,li:list) -> set: # Returns the union of all sets in the list
    s = s.union(li[0])
    if len(li) == 1:
        return s
    else:
        return list_union(s,li[1:])

all_fields = []
for li in dfields.values():
    all_fields.extend(li)
ranges = [make_range(s) for s in all_fields]
all_fields = list_union(ranges[0],ranges[1:])

error_rate = 0
for row in range(nearby.shape[0]):
    for col in range(nearby.shape[1]):
        if {nearby[row,col]}.isdisjoint(all_fields):
            error_rate += nearby[row,col]
print(error_rate)

# Part 2
for key in dfields.keys():
    li = [make_range(s) for s in dfields[key]]
    dfields[key] = list_union(li[0],li[1:])
    
valid = []
invalid = []
for row in range(nearby.shape[0]):
    error = False
    for col in range(nearby.shape[1]):
        if {nearby[row,col]}.isdisjoint(all_fields):
            error = True
    if not error:
        valid.append(nearby[row,:])
    else:
        invalid.append(nearby[row,:])
valid = np.array(valid)
invalid = np.array(valid)

match = []
for col in range(valid.shape[1]):
    s = set(valid[:,col])
    li = []
    for key in dfields.keys():
        if s.issubset(dfields[key]):
            li.append(key)
    match.append(li)
    
lens = [len(li) for li in match]
hist = []

for n in range(1,21):
    ind = lens.index(1)
    while ind in hist:
        ind = lens.index(1,ind+1)
    hist.append(ind)
    name = match[ind][0]
    for i in range(len(match)):
        if i == ind:
            continue
        if name in match[i]:
            print(' removing...')
            match[i].remove(name)
    lens = [len(li) for li in match]
    
ans = 1.
for i, name in enumerate(match):
    name = name[0]
    if name.startswith('departure'):
        ans *= my_ticket[i]
        
# Day 17
# Read and clean
file_reader = open('Day 17.txt','r')
lines = file_reader.readlines()
lines = [list(line.replace('#','1').replace('.','0').replace('\n','')) for line in lines]

# Part 1
import numpy as np
lines = np.array(lines).astype(int)
arr = np.zeros((lines.shape[0],lines.shape[1],1))
arr[:,:,0] = lines

def expand_pocket(arr):
    expansion = np.zeros((arr.shape[0]+2,arr.shape[1]+2,arr.shape[2]+2))
    expansion[1:-1,1:-1,1:-1] = arr
    return expansion
    
def get_neighbors(arr,x,y,z):
    return arr[x-1:x+2,y-1:y+2,z-1:z+2].sum() - arr[x,y,z]

def change_state(arr,shape):
    new_arr = np.zeros(arr.shape)
    for i in range(1,shape[0]+1):
        for j in range(1, shape[1]+1):
            for k in range(1, shape[2]+1):
                num = get_neighbors(arr,i,j,k)
                if num == 3:
                    new_arr[i,j,k] = 1
                    continue
                if (num == 2) & (arr[i,j,k] == 1):
                    new_arr[i,j,k] = 1
    return new_arr

def print_z(arr):
    for z in range(0,arr.shape[2]):
        print(f'z={z}:')
        print(arr[:,:,z])

# Giving arr some extra padding
arr = expand_pocket(arr)

# Time starts at t=0
# hist = [arr]
for t in range(1,7):
    new_arr = expand_pocket(arr)
    new_arr = change_state(new_arr,arr.shape)
    # hist.append(new_arr)
    arr = new_arr
print(arr.sum())
    
# Part 2 (It's literally the same thing with 4 dimensions)
# import numpy as np
# lines = np.array(lines).astype(int)
arr = np.zeros((lines.shape[0],lines.shape[1],1,1))
arr[:,:,0,0] = lines

def expand_hyper(arr):
    expansion = np.zeros((arr.shape[0]+2,arr.shape[1]+2,arr.shape[2]+2,arr.shape[3]+2))
    expansion[1:-1,1:-1,1:-1,1:-1] = arr
    return expansion
    
def hyper_neighbors(arr,x,y,z,w):
    return arr[x-1:x+2,y-1:y+2,z-1:z+2,w-1:w+2].sum() - arr[x,y,z,w]

def change_hyper(arr,shape):
    new_arr = np.zeros(arr.shape)
    for i in range(1,shape[0]+1):
        for j in range(1, shape[1]+1):
            for k in range(1, shape[2]+1):
                for l in range(1, shape[3]+1):
                    num = hyper_neighbors(arr,i,j,k,l)
                    if num == 3:
                        new_arr[i,j,k,l] = 1
                        continue
                    if (num == 2) & (arr[i,j,k,l] == 1):
                        new_arr[i,j,k,l] = 1
    return new_arr

# def print_hyperz(arr):
#     for z in range(0,arr.shape[2]):
#         print(f'z={z}:')
#         print(arr[:,:,z,:])

# Giving arr some extra padding
arr = expand_hyper(arr)

# Time starts at t=0
# hist = [arr]
for t in range(1,7):
    new_arr = expand_hyper(arr)
    new_arr = change_hyper(new_arr,arr.shape)
    # hist.append(new_arr)
    arr = new_arr
print(arr.sum())

# Day 18
file_reader = open('Day 18.txt','r')
lines = file_reader.readlines()
lines = [line.replace('\n','').replace(' ','') for line in lines]

# Parts 1 and 2
def add(a,b):
    return a + b

def mult(a,b):
    return a * b

def find_expr(s,start):
    closed = False
    nxt_opn = start
    nxt_cls = start
    while not closed:
        nxt_opn = s.find('(',nxt_opn+1)
        nxt_cls = s.find(')',nxt_cls+1)
        if (nxt_opn == -1) | (nxt_opn > nxt_cls):
            closed = True
    return s[start+1:nxt_cls]

def new_math(s):
    i = 0
    if s[0] == '(':
        expr = find_expr(s,0)
        # print(expr)
        a = new_math(expr)
        i += len(expr)+1
    else:
        a = int(s[0])
    # print(s[0])
    oper = None # short for operator
    i += 1
    while i < len(s):
        # print(s[i])
        switcher = {'+':add,'*':mult}
        if (s[i] == '+') | (s[i] ==  '*'):
            oper = switcher[s[i]] # Part 1 version
            # Addition for Part 2
            if s[i] == '*':
                expr = s[i+1:]
                b = new_math(expr)
                a = oper(a,b)
                i += len(expr)
        else:
            if s[i] == '(':
                expr = find_expr(s,i)
                b = new_math(expr)
                i += len(expr) + 1 # Gotta skip ahead
            else:
                b = int(s[i])
            a = oper(a,b)
            # print(a)
        i += 1
    return a

results = []
for i, line in enumerate(lines):
    # print(i)
    results.append(new_math(line))

# Day 19
# Read and clean
file_reader = open('Day 19.txt')
lines = file_reader.readlines()

# Part 1
rule_lines = []
messages = []
li = rule_lines
for line in lines:
    if line == '\n':
        li = messages
        continue
    li.append(line.replace('\n',''))
del li

def build_rule(s: str):
    
    def rule(rules):
        li = s.split(' ')
        line = '('
        for item in li:
            if item == '|':
                line = line + '|'
            elif item.startswith('"'):
                line = line + item[1]
            else:
                line = line + rules[item](rules)
        line = line + ')'
        return line
    
    return rule

import regex
rules = {}
get_num = regex.compile('(?P<num>\d+):')
for line in rule_lines:
    m = get_num.match(line)
    num =  m.group('num')
    rules.update({num:build_rule(line[m.span()[1]+1:])})

def execute_rule(rules, num: str, message: str):
    matcher = regex.compile(rules[num](rules))
    m = matcher.fullmatch(message)
    return m is not None

ans = 0
for message in messages:
    ans += execute_rule(rules, '0', message)
print(ans)
    
# Part 2
def rule_eight(rules):
    return '(' + rules['42'](rules) + ')+'
    
def rule_eleven(rules):
    return '(?P<one>' + rules['42'](rules) + rules['31'](rules) + '|' + rules['42'](rules) + '(?P>one)' + rules['31'](rules) + ')'

rules.update({'8':rule_eight,'11':rule_eleven})
# regex.fullmatch('c(?P<one>ab|a(?P>one)b)','caaabbb') is not None

ans = 0
for message in messages:
    ans += execute_rule(rules, '0', message)
print(ans)

# Day 20
# Create a list of the tiles
# Identify the four edges of each tile
# Create a list of edge counters (counting which edges don't have matches)
# Sum the IDs of the tiles with 2 edges facing out

# Read and clean
lines = ([line.replace('\n','') for line in open('Day 20.txt','r').readlines()])

import re
import numpy as np
tiles = {}
tile = []
for line in lines:
    if line.startswith('Tile'):
        tile_id = re.search('\d+',line).group()
    elif line == '':
        tiles.update({int(tile_id):np.array(tile).astype(int)})
        tile = []
    else:
        tile.append(list(line.replace('#','1').replace('.','0')))
del tile_id, tile, line

# Part 1
def get_edges(arr):
    return [arr[0,:],arr[-1,:],arr[:,0],arr[:,-1]]
edge_types = {0:'top',1:'bottom',2:'left',3:'right'}

inside_edges = dict(zip(list(tiles.keys()),[0 for n in tiles]))
edge_matches = []
for i in range(len(tiles)):
    my_id = list(tiles.keys())[i]
    t = tiles[my_id]
    my_edges = get_edges(t)
    for j in range(i+1,len(tiles)):
        their_id = list(tiles.keys())[j]
        u = tiles[their_id]
        their_edges = get_edges(u)
        for k in range(len(my_edges)):
            for l in range(len(their_edges)):
                if np.array_equal(my_edges[k], their_edges[l]) or np.array_equal(np.flip(my_edges[k]), their_edges[l]):
                    inside_edges[my_id] += 1
                    inside_edges[their_id] += 1
                    edge_matches.append({my_id,their_id})
                    # if 1439 == my_id:
                    #     print(edge_types[k])
                    #     if np.array_equal(np.flip(my_edges[k]),their_edges[l]):
                    #         print('flipped')
                    # elif 1439 == their_id:
                    #     print(edge_types[l])
                    #     if np.array_equal(np.flip(my_edges[k]),their_edges[l]):
                    #         print('flipped')
                    

ans = 1.
for key in inside_edges:
    if inside_edges[key] == 2:
        ans *= key
del key
print(ans)

# Part 2

corners = [key for key in tiles.keys() if inside_edges[key] == 2]
# borders = [key for key in tiles.keys() if inside_edges[key] == 3]
# middle  = [key for key in tiles.keys() if inside_edges[key] == 4]

def find_matches(*tile_ids): # Receives either one or two ids
    sets = []
    for tile_id in tile_ids:
        s = set()
        for pair in edge_matches:
            if tile_id in pair:
                s = s.union(pair)
        sets.append(s.difference({tile_id}))
    if len(sets) == 1:
        return sets[0]
    else:
        return sets[0].intersection(sets[1])
        
# K. So I'm using tile 1439 (corners[1]) as the top left corner of the image
image = np.zeros((12*10,12*10))
image[0:10,0:10] = tiles[1439]
grid = np.zeros((12,12))
grid[0,0] = 1439

left = False
for row in range(12):
    for col in range(12):
        if (row == 0) & (col == 0):
            continue
        else:
            if row == 0:
                matches = find_matches(grid[row,col-1])
            elif col == 0:
                matches = find_matches(grid[row-1,col])
                left = True
            else:
                matches = find_matches(grid[row-1,col],grid[row,col-1])
            fit = False
            for match in matches:
                for flip in [False,True]:
                    for rot in range(4):
                        if left:
                            my_edge = tiles[match][0,:]
                            their_edge = image[row*10-1,col*10:(col+1)*10]
                        else:
                            my_edge = tiles[match][:,0]
                            their_edge = image[row*10:(row+1)*10,col*10-1]
                        if np.array_equal(my_edge,their_edge):
                            fit = True
                            image[row*10:(row+1)*10,col*10:(col+1)*10] = tiles[match]
                            grid[row,col] = match
                            break
                        else:
                            tiles[match] = np.rot90(tiles[match])
                    if fit:
                        break
                    else:
                        tiles[match] = np.flip(tiles[match],axis=0)
                if fit:
                    break
        left = False
                
for i in range(12):
    for j in range(12):
        if int(grid[i,j]) not in tiles.keys():
            print('uhoh')
            #Sooooo, I think it actually worked.

# Now to find the sea monsters
# First, remove the unnecessary columns and rows
# Then write a function that can find sea monsters
# Then remove the '1' pixels with sea monsters
# and count the remaining '1' pixels

steps = (9,1)
for axis in [0,1]:
    ind = 119
    for i in range(24):
        image = np.delete(image,ind,axis)
        ind -= steps[i%2]
image = image.astype(int)

def find_pattern(arr,pattern):
    h, w = arr.shape[0], arr.shape[1]
    h_p, w_p = pattern.shape[0], pattern.shape[1]
    li = []
    for row in range(h-h_p+1):
        for col in range(w-w_p+1):
             if np.array_equal(pattern,arr[row:row+h_p,col:col+w_p]*pattern):
                 li.append((row,col))
    return li

sea_monster = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                        [1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1],
                        [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0]])

# Not the most elegant to orient the image, but it worked
while not find_pattern(image,sea_monster):
    image = np.rot90(image)
image = np.flip(image,axis = 0)

for coor in find_pattern(image,sea_monster):
    image[coor[0]:coor[0]+sea_monster.shape[0],coor[1]:coor[1]+sea_monster.shape[1]] -= sea_monster
print(image.sum())
# I did it XD

# Day 21
# Read and clean
lines = ([line.replace('\n','') for line in open('Day 21.txt','r').readlines()])

# Part 1
foods = [line.split(' ') for line in lines]
ingredients = set()
allergens = set()
allergen_locs = {}
for food in foods:
    s = ingredients
    a = set()
    for item in food:
        if item == '(contains':
            s = allergens
            continue
        item = item.replace(')','').replace(',','')
        if s is ingredients:
            a.add(item)
        else:
            if item not in allergens:
                allergen_locs.update({item:a})
            else:
                allergen_locs[item] = allergen_locs[item].intersection(a)
        s.add(item)

maybe_allergen = set()
for locs in allergen_locs.values():
    maybe_allergen = maybe_allergen.union(locs)
no_allergen = ingredients.difference(maybe_allergen)

ans = 0
for food in foods:
    for item in food:
        if item == '(contains':
            break
        if item in no_allergen:
            ans += 1
print(ans)

# Part 2
keys = list(allergen_locs.keys())
lens = [len(allergen_locs[key]) for key in keys]
hist = []

for n in range(len(allergens)):
    ind = lens.index(1)
    while ind in hist:
        ind = lens.index(1,ind+1)
    hist.append(ind)
    ingredient = next(iter(allergen_locs[keys[ind]]))
    for i in range(len(allergen_locs)):
        if i == ind:
            continue
        if ingredient in allergen_locs[keys[i]]:
            # print('Removing...')
            allergen_locs[keys[i]].remove(ingredient)
    lens = [len(allergen_locs[key]) for key in keys]
    
l = ''
keys.sort()
for key in keys:
    l = l + next(iter(allergen_locs[key])) + ','
l = l[:-1]
print(l)

# Day 22
# Read and clean
lines = ([line.replace('\n','') for line in open('Day 22.txt','r').readlines()])
example =  ['Player 1:',
            '9',
            '2',
            '6',
            '3',
            '1',
            '',
            'Player 2:',
            '5',
            '8',
            '4',
            '7',
            '10']

def initialize(lines): # Made restarting and testing examples much easier
    players = [[],[]]
    n = 0
    for line in lines:
        if line.startswith('P'):
            continue
        if line == '':
            n += 1
            continue
        players[n].append(int(line))
    return players

# Part 1
def play_game(players):
    win = False
    while not win:
        winner = int(players[0][0] < players[1][0]) # 0 if player 0 wins, 1 if player 1 wins
        players[winner].append(players[winner].pop(0))
        players[winner].append(players[1-winner].pop(0))
        win = (not bool(players[0])) | (not bool(players[1]))
    return players

play_game(players)

ans = 0
for i, card in enumerate(players[0]):
    ans += card * (len(players[0])-i)
print(ans)

# Part 2
def recursive_combat(players):
    win = False
    hist = []
    while not win:
        if players in hist:
            return 0
        hist.append([players[0].copy(),players[1].copy()])
        draws = [players[0].pop(0),players[1].pop(0)]
        if (draws[0] <= len(players[0])) & (draws[1] <= len(players[1])):
            winner = recursive_combat([players[0][:draws[0]].copy(),players[1][:draws[1]].copy()])
        else:
            winner = int(draws[0] < draws[1])
        players[winner].append(draws[winner])
        players[winner].append(draws[1-winner])
        win = (not bool(players[0])) | (not bool(players[1]))
    return winner

# Reset the decks
players = initialize(lines)
# And begin
winner = recursive_combat(players)

ans = 0
for i, card in enumerate(players[winner]):
    ans += card * (len(players[winner])-i)
print(ans)

# Day 23

def crab_cups(cups, num_rounds):
    round_n = 1
    ind = 0
    while round_n <= num_rounds:
        # print(f'\n-- move {round_n} --')
        cups, ind = move_cups(cups, ind)
        round_n += 1
    return cups

def move_cups(cups, ind):
    current = cups[ind]
    # print('cups:', cups)
    # print('current:',current)
    
    if ind + 4 >= len(cups):
        pickup = cups[ind+1:] + cups[:(ind+4)%len(cups)]
        cups = cups[(ind+4)%len(cups):ind+1]
    else:
        pickup = cups[ind+1:ind+4]
        del cups[ind+1:ind+4]
    # print('pickup:', pickup)
            
    dest = current - 1
    while dest in pickup or dest < 1:
        dest = len(cups)+3 if dest <= 1 else dest - 1
    # print('destination:', dest)
    
    insert = cups.index(dest) + 1 # This line is slow
    cups = cups[:insert] + pickup + cups[insert:]
    ind = (cups.index(current) + 1) % len(cups) # So is this one
    return cups, ind
# This works for part 1 but it is not fast enough for part 2
# Part 1 Answer

cups = [7,1,6,8,9,2,5,4,3]
ans1 = crab_cups(cups,100)
# example = [3,8,9,1,2,5,4,6,7]
# ex = crab_cups(example,1)

# Part 2

# A Python user on Reddit suggests a linked list where the nth element in the list points to the cup that follows the nth cup flowing clockwise
import numpy as np
from numba import njit
import time

@njit
def build_links(li):
    links = np.zeros((len(li)+1,),dtype=np.uint32)
    for i, num in enumerate(li): links[num] = li[(i+1)%len(li)]
    return links

@njit
def linked_cups(links,num_rounds,start):
    current = start
    n_round = 1
    while n_round <= num_rounds:
        links, current = adjust_links(links,current)
        n_round += 1
    return links

@njit
def adjust_links(links,current):
    next_1 = links[current]
    next_2 = links[next_1]
    next_3 = links[next_2]
    after_pickup = links[next_3]
    
    dest = current - 1
    while dest in (next_1,next_2,next_3) or dest == 0:
        dest = dest - 1
        if dest == 0:
            dest = len(links) - 1
    
    links[current] = after_pickup # Cut out the next three
    links[next_3] = links[dest] # Stitch those three in after the destination
    links[dest] = next_1
    return links, after_pickup
    
t0 = time.time()

cups = [7,1,6,8,9,2,5,4,3] + list(range(10,1000001))
links = build_links(cups)
ans2 = linked_cups(links,10000000,cups[0])
x = ans2[1]
y = ans2[x]
print(float(x)*y)
print(f'Total time: {round(time.time()-t0,2)} seconds')

# Day 24
tiles = [line.strip() for line in open('Day 24.txt','r').readlines()]
example = [line.strip() for line in open('Day 24 example.txt','r').readlines()]

# Part 1
import numpy as np
def flip_tiles(li):
    floor = {}
    for tile in li:
        coor = get_coor(tile)
        floor.update({tuple(coor):1-bool(floor.get(tuple(coor)))})
    return floor

def get_coor(tile):
    coor = np.zeros(2)
    moves = {'w':(-1,0),'e':(1,0),'se':(0,-1),
             'sw':(-1,-1),'ne':(1,1),'nw':(0,1)}
    i = 0
    while i < len(tile):
        if tile[i] == 's' or tile[i] == 'n':
            direction = tile[i] + tile[i+1]
            i += 1
        else:
            direction = tile[i]
        # print(direction)
        coor += np.array(moves[direction])
        i += 1
    return tuple(coor)

ans = flip_tiles(tiles)
print(sum(ans.values()))

ex_ans = flip_tiles(example)

# Part 2
# So now we know hich tiles are flipped over and we can proceed to build a map
# Very similar to the Conway game
# Gotta stick all the tiles onto a grid
coors = np.array(list(ans.keys())).astype(int)
coors[:,0] -= min(coors[:,0])
coors[:,1] -= min(coors[:,1])
floor = dict(zip([tuple(coor) for coor in coors],list(ans.values())))

floor_map = np.zeros((int(max(coors[:,0]))+1,int(max(coors[:,1]))+1))
for coor in coors:
    floor_map[coor[0],coor[1]] = bool(floor.get(tuple(coor)))

# And then we can begin
def living_art(floor,n):
    t = 0
    print(f'Day {t}: {floor.sum()}')
    while t < n:
        new_floor = expand_floor(floor)
        new_floor = change(new_floor,floor.shape)
        floor = new_floor
        t += 1
        print(f'Day {t}: {floor.sum()}')
    return floor

def change(floor, shape):
    new_floor = floor.copy()
    for x in range(1,shape[0]+1):
        for y in range(1, shape[1]+1):
            num = get_adjacent(floor,x,y)
            color = floor[x,y]
            if (num == 0 or num > 2) and color == 1:
                # print('flipped',x,y)
                new_floor[x,y] = 0
            if num == 2 and color == 0:
                new_floor[x,y] = 1
    return new_floor

def get_adjacent(arr,x,y):
    return int(arr[x-1:x+2,y-1:y+2].sum() - arr[x,y] - arr[x-1,y+1] - arr[x+1,y-1])

def expand_floor(arr):
    expansion = np.zeros((arr.shape[0]+2,arr.shape[1]+2))
    expansion[1:-1,1:-1] = arr
    return expansion

floor_map = expand_floor(floor_map)
n = 100
ans = living_art(floor_map,n)
print(ans.sum())


# Day 25

pub1, pub2 = 15733400, 6408062
sub = 7

# Part 1

def get_loop_size(sub,key):
    n = 1
    loop_size = 0
    while n != key:
        n = n * sub % 20201227
        loop_size += 1
    return loop_size

size1 = get_loop_size(sub,pub1)
size2 = get_loop_size(sub,pub2)
print(pow(pub1,size2,20201227))
print(pow(pub2,size1,20201227))

# Part 2
# Awwww. It's a freebie!. Merry Christmas!