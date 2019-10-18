
import copy
import datetime
import os
import time
import heapq
import matplotlib.pyplot as plt



def manhattanDistance( xy1, xy2 ):
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class Queue:
    def __init__(self):
        self.list = []
    def push(self,item):
        self.list.insert(0,item)
    def pop(self):
        return self.list.pop()
    def isEmpty(self):
        return len(self.list) == 0

class Stack:
    def __init__(self):
        self.list = []
    def push(self,item):
        self.list.append(item)
    def pop(self):
        return self.list.pop()
    def isEmpty(self):
        return len(self.list) == 0
   
class Node :
    def __init__(self, map_lst,foods, num_food, agent1, agent2, path_cost, parent, action):
        self.num_food = num_food
        self.agent1 = agent1
        self.agent2 = agent2
        self.map_lst = copy.deepcopy(map_lst)
        self.path_cost = path_cost
        self.parent = parent
        self.action = action
        self.foods = copy.deepcopy(foods)

    def move_to_update(self,index_agent,action):
        row, col , prev_col , prev_row = 0 , 0 , 0 , 0
        if index_agent == 1 :
            prev_row, prev_col =  self.agent1[0] , self.agent1[1]
        elif index_agent == 2:
            prev_row, prev_col =  self.agent2[0] , self.agent2[1]

        if action ==  'R' :  # right
            col = prev_col + 1
            row = prev_row
        elif action == 'U' : # up
            row = prev_row - 1
            col = prev_col
        elif action == 'L' : # left
            col = prev_col - 1
            row = prev_row
        elif action == 'D' : # down
            row = prev_row + 1
            col = prev_col

        if self.is_food('1',row,col):
            self.food_eat('1',row,col)
        elif self.is_food('2',row,col):
            self.food_eat('2',row,col)
         
        elif self.is_food('3',row,col):
            self.food_eat('3',row,col)
        
            
        
        self.pacman_pos_change(index_agent,prev_row,prev_col,row,col)
     
    def can_move_from(self, index_agent, action):
        row, col , prev_col , prev_row = 0 , 0 , 0 , 0
        if index_agent == 1 :
            prev_row, prev_col =  self.agent1[0] , self.agent1[1]
        elif index_agent == 2:
            prev_row, prev_col =  self.agent2[0] , self.agent2[1]

        if action ==  'R' :  # right
            col = prev_col + 1
            row = prev_row
        elif action == 'U' : # up
            row = prev_row - 1
            col = prev_col
        elif action == 'L' : # left
            col = prev_col - 1
            row = prev_row
        elif action == 'D' : # down
            row = prev_row + 1
            col = prev_col
        
        if self.is_wall(row,col):
            return False
        if index_agent == 1 and self.is_food('2',row,col):
            return False
        if index_agent == 2 and self.is_food('1',row,col):
            return False
        if self.agents_hit(index_agent,row,col):
            return False
        return True

    def make_map_str(self):
        map_str = ""
        for i in range(len(self.map_lst)):
            for j in range(len(self.map_lst[0])):
                map_str = map_str + self.map_lst[i][j]
        return map_str

    def make_set_of_pos(self):
        pos = set()
        for p in self.foods['1']:
            pos.add(p)
        for p in self.foods['2']:
            pos.add(p)
        for p in self.foods['3']:
            pos.add(p)
        pos.add(self.agent1)
        pos.add(self.agent2)
        return pos

    def pacman_pos_change(self,index_agent,prev_row,prev_col,new_row,new_col):
        if index_agent == 1:
            self.map_lst[prev_row][prev_col] = ' '
            self.map_lst[new_row][new_col] = 'P'
            self.agent1 = (new_row,new_col)
        elif index_agent == 2:
            self.map_lst[prev_row][prev_col] = ' '
            self.map_lst[new_row][new_col] = 'Q'
            self.agent2 = (new_row,new_col)

    def print_map(self):
        for i in range(len(self.map_lst)):
            for j in range(len(self.map_lst[0])):
                print(self.map_lst[i][j],end="")
            print()
        
    def is_wall(self,row,col):
        return self.map_lst[row][col] == '%'

    def is_food(self,food_type ,row,col):
        return self.map_lst[row][col] == food_type

    def food_eat(self, type_food, row, col):
        if self.map_lst[row][col] == type_food:
            self.map_lst[row][col] = ' '
            self.num_food -=1
            self.foods[type_food].remove((row,col))

    def agents_hit(self, index_agent_move, new_row, new_col):
        if index_agent_move == 1 :
            return self.agent2 == (new_row, new_col)
           
        elif index_agent_move == 2 :
            return self.agent1 == (new_row , new_col)
             
class Problem:
    def __init__(self, file_name):
        f = open(file_name,"r")
        self.map = f.readlines()
      
    def getStartState(self):
        nrow , ncol = len(self.map), len(self.map[0])-1
        num_food = 0
        map_lst = [[' ' for j in range(ncol)] for i in range(nrow)]
        foods = {}
        foods['1'] = []
        foods['2'] = []
        foods['3'] = []
        
       
        for i in range(nrow):
            for j in range(ncol):
                char = self.map[i][j]
                if char == '%':
                    map_lst[i][j] = '%'
                elif char == '1':
                    map_lst[i][j] = '1'
                    foods['1'].append((i,j))
                    num_food+=1
                elif char == '2' :
                    map_lst[i][j] = '2'
                    foods['2'].append((i,j))
                    num_food += 1
                elif char == '3' :
                    map_lst[i][j] = '3'
                    foods['3'].append((i,j))
                    num_food += 1
                elif char == 'P' :
                    agent1 = (i,j)
                    map_lst[i][j] = 'P'
                elif char == 'Q' :
                    agent2 = (i,j)
                    map_lst[i][j] = 'Q'
        start_node = Node(map_lst,foods, num_food, agent1, agent2, 0, None, None)
        # foods['1'].remove((6,1))
        # print(foods)
        return start_node
        
    def isGoalState(self,node):
        if node.num_food == 0 :
            return True
            
    def getSuccessors(self,node):
        actions = ["1R","1U","1L","1D","2R","2U","2L","2D"]
        successors = []
        for action in actions: 
            if action[0] == "1" :
                if node.can_move_from(1, action[1]):
                    new_node = Node(node.map_lst,node.foods, node.num_food ,node.agent1,
                                    node.agent2, node.path_cost + 1, None,action)
                    new_node.parent = node
                    new_node.move_to_update(1,action[1])
                    successors.append(new_node)

                    
            if action[0] == "2" :
                if node.can_move_from(2, action[1]):
                    new_node = Node(node.map_lst, node.foods, node.num_food , node.agent1, node.agent2,
                                     node.path_cost + 1, None,action)
                    new_node.parent = node
                    new_node.move_to_update(2,action[1])
                    successors.append(new_node)

        return successors

    def BFS(self):
        start_time = datetime.datetime.now()
        fringe = Queue()
        num_visited_total = 0
        visited = []
        node_start = problem.getStartState()
        if node_start.num_food == 0:
            print("already in goal!")
            node_start.print_map()
            return
        fringe.push(node_start)
        
        while not fringe.isEmpty():
            popped_element = fringe.pop()
            node = popped_element
            node_pos_set = node.make_set_of_pos()
           
            if node_pos_set not in visited:
                num_visited_total +=1
                visited.append(node_pos_set)
                successors = problem.getSuccessors(node)
                for successor in successors:
                    child_node = successor
                    child_node_pos_set =  child_node.make_set_of_pos()
                    if  self.isGoalState(child_node):
                        end_time = datetime.datetime.now()
                        moves = []
                        nodes_to_goal = []
                        nodes_to_goal.append(child_node)
                        while child_node.parent != None:
                            moves.append(child_node.action)
                            child_node = child_node.parent
                            nodes_to_goal.append(child_node)
                        nodes_to_goal.reverse()
                        moves.reverse()
                        for node in nodes_to_goal:
                            os.system('cls' if os.name == 'nt' else 'clear')
                            node.print_map()
                            time.sleep(.5) 
                        print("BFS:")
                        print("num node visited unique",len(visited))
                        print("num node visited total" , num_visited_total) 
                        print("cost till node:", nodes_to_goal[-1].path_cost) 
                        print("actions :" ,moves)
                        print("total time:", (end_time - start_time).seconds, "seconds")
                        return
                    fringe.push(child_node)
            num_visited_total +=1

    def DFS(self,level,start_time,num_node_total,num_node_unique):
        print(level)
        fringe = Stack()  
        visited_set_pos = []
        visited_node = []
        node_start = problem.getStartState()
        fringe.push(node_start) 

        while not fringe.isEmpty():
            node = fringe.pop()
            node_set_of_pos = node.make_set_of_pos()
     
            if self.isGoalState(node):
                end_time = datetime.datetime.now()
                moves = []
                nodes_to_goal = []
                nodes_to_goal.append(node)
                while node.parent != None:
                    moves.append(node.action)
                    node = node.parent
                    nodes_to_goal.append(node)

                nodes_to_goal.reverse()
                moves.reverse()
                for node in nodes_to_goal:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    node.print_map()
                    time.sleep(.5)
                print("IDS")
                # print("num node visited",len(visited_node)) 
                print("num node visited total",len(num_node_total)) 
                print("num node visited unique",len(visited_set_pos))
                print("cost till node:", nodes_to_goal[-1].path_cost) 
                print("actions :" ,moves)
                print("total time:", (end_time - start_time).seconds, "seconds")
                return True 
            else:
                if node.path_cost < level :
                    if node_set_of_pos  in visited_set_pos :
                        num_node_total.append(node_set_of_pos)
                        index = visited_set_pos.index(node_set_of_pos)
                        if visited_node[index].path_cost > node.path_cost :
                            del visited_node[index]
                            del visited_set_pos[index]
                            visited_set_pos.append(node_set_of_pos)
                            visited_node.append(node)
                            successors = problem.getSuccessors(node)
                            successors.reverse()
                            for child_node in successors:
                                fringe.push(child_node)
                    else:
                        num_node_total.append(node_set_of_pos)
                        
                        visited_set_pos.append(node_set_of_pos)
                        visited_node.append(node)     
                        successors = problem.getSuccessors(node)
                        successors.reverse()
                        for child_node in successors:
                            fringe.push(child_node)
        return False

    def IDS(self):
        start_time = datetime.datetime.now()
        level = 0
        num_node_total= []
        num_node_unique= set()
        while(True):
            if self.DFS(level,start_time,num_node_total,num_node_unique):
                break
            level = level + 1
        return

    def heuristic(self,node):
        return node.num_food 

        # distance_p = []
        # distance_q = []

        # for pos in node.foods['1']:
        #     distance_p.append(manhattanDistance(pos,node.agent1))
        # for pos in node.foods['2']:
        #     distance_q.append(manhattanDistance(pos,node.agent2))
        
        # x,y = 0 , 0
        # if len(distance_p):
        #     x = min(distance_p)
        # if len(distance_q):
        #     y = min(distance_q)
        #     y = min(distance_q)
        # return x + y
        
    def aStarSearch(self):
        start_time = datetime.datetime.now()
        num_node_total = 0
        fringe = PriorityQueue()    
        visited= []
    
        node_start = problem.getStartState()
        fringe.push(node_start, 0)   
        while not fringe.isEmpty():
            node = fringe.pop()
            node_pos_set = node.make_set_of_pos()
        
            if problem.isGoalState(node):
                end_time = datetime.datetime.now()
                moves = []
                nodes_to_goal = []
                nodes_to_goal.append(node)
                while node.parent != None:
                    moves.append(node.action)
                    node = node.parent
                    nodes_to_goal.append(node)
                nodes_to_goal.reverse()
                moves.reverse()
                for node in nodes_to_goal:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    node.print_map()
                    time.sleep(.5) 
                print("A* search:")
                print("num node visited unique",len(visited)) 
                print("num node visited total" , num_node_total)
                print("cost till node:", nodes_to_goal[-1].path_cost) 
                print("actions :" ,moves)
                print("total time:", (end_time - start_time).seconds, "seconds")
                break    
            else:
                if node_pos_set not in visited:
                    num_node_total +=1    
                    visited.append(node_pos_set)     
                    successors = problem.getSuccessors(node)
                    for child_node in successors:
                        child_cost = child_node.path_cost     
                        fringe.push(child_node, child_cost + self.heuristic(child_node))    
                num_node_total +=1 
        return


if __name__ == "__main__":
    problem = Problem("test1")
   
    # problem.BFS()
    # problem.IDS()
    # problem.aStarSearch()
   
    bfs_goal_depth =[33,17,20,17,14]
    bfs_time = [22,0,0,12,0]

    ids_goal_depth =[33,17,20,17,14]
    ids_time = [3205,5,13,107,0]

    astar_goal_depth =[33,17,20,17,14]
    astar_time = [17,0,0,9,0]


    plt.xlabel('Goal Depth')
    plt.ylabel('Execution Time')
    plt.scatter(bfs_goal_depth,bfs_time)
    plt.grid()
    plt.show()
    

