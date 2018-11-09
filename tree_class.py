import numpy as np
import sys

class Tree(object):
    def __init__(self):
        self.child_dict = {}
    #Linkes to all root nodes for all the given trees. Indexed by stacksize and position
    #Path should be stacksize -> position -> root_node
    #Methods
    #Get root node
    def get_child(self,position):
        if position in self.child_dict:
            return self.child_dict[position]
        else:
            self.child_dict[position] = Node(parent=None)
            return self.child_dict[position]
    
class Node(object):
    def __init__(self,parent):
        self.parent = parent
        self.child_dict = {}
        self.visits = 0
        self.ev = 0
        self.ev_sum = 0
        self.kind = 0
        
    def get_child(self,location):
        if location not in self.child_dict:
            self.child_dict[location] = Node(parent=self)
        return self.child_dict[location]
            
    def update_visit(self):
        self.visits += 1
        
    #pass in root node
    def count_nodes(self,node,num):
        num += 1
        #print('node number',num)
        if len(node.child_dict.keys()) != 0:
            for k,v in node.child_dict.items():
#                 if isinstance(v,dict):
#                     for sub_key,sub_value in v.items():
#                         node.count_nodes(sub_value,num)
#                     print(node.visits,'visits')
#                 else:
                #print(node.visits,'visits')
                node.count_nodes(v,num)
        else:
            print('final num',num)
            return num
    
    #propagates the ev back to root node. 
    #But also returns an array of highest EV/action pairs
    #In the event of a EV tie, it should randomly select the action.
    #returns EVs of only the actions. returns the parent game_state node and game_state
    #Change from recursive to loop
    def fast_propagate(self,node,last_result):
        #print(last_result,'last_result')
        attacks = []
        defends = []
        attack_evs = np.array([])
        defend_evs = np.array([])
        node.ev_sum += last_result
        node.ev = node.ev_sum / node.visits
#         print(node.ev,'node.ev')
        while node.parent != None:
            node = node.parent
            node.ev_sum += last_result
            node.ev = node.ev_sum / node.visits
            #Grab best ev from parent node game_state - child node actions
            if node.kind == 1:
                #split between defending and attacking gamestates
                keys = list(node.child_dict.keys())
                values_temp = list(node.child_dict.values())
                values = [value.ev for value in values_temp]
    #             print(len(keys),'len keys')
    #             print(values,'values')
                location = np.argmax(values)
                action = keys[location]
                ev = values[location]
                if node.game_state == 1:
                    defends.append(int(action))
                    defends.append(node.key)
                    defend_evs = np.append(defend_evs,ev)
                else:
                    attacks.append(int(action))
                    attacks.append(node.key)
                    attack_evs = np.append(attack_evs,ev)
        return attacks,attack_evs,defends,defend_evs

