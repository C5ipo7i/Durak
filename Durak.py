
# coding: utf-8

# In[ ]:


from random import shuffle
from random import choice
from random import randint
import sys
import os
import numpy as np
import copy
import pickle
import json
import time
import binascii
from keras.layers import Input,dot,multiply,Add,Dense,Activation,Lambda,Flatten,Conv1D,Conv2D,LeakyReLU,Reshape,Concatenate,LSTM,Embedding,Masking
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K


# In[ ]:


bytestream = b'\x00k\x8bEg'
hex_data = binascii.hexlify(bytestream)
str_data = hex_data.decode('utf-8')
binascii.unhexlify(str_data.encode('utf-8')) == bytestream


# In[ ]:


a = np.arange(10).reshape(2,5)
print(a)
bytestream = pickle.dumps(a)
print(bytestream)
hex_data = binascii.hexlify(bytestream)
str_data = hex_data.decode('utf-8')
bytestream_2 = binascii.unhexlify(str_data.encode('utf-8'))
print(type(bytestream_2))
c = pickle.loads(bytestream_2)
print(c)


# In[ ]:


deck = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d']]


# In[ ]:


def suits_to_str(cards):
    new_cards = copy.deepcopy(cards)
    for card in new_cards:
        if card[1] == 0:
            card[1] = 's'
        elif card[1] == 1:
            card[1] = 'h'
        elif card[1] == 2:
            card[1] = 'd'
        else:
            card[1] = 'c'
    return new_cards

#2d
def suits_to_num(cards):
    new_cards = copy.deepcopy(cards)
    for card in new_cards:
        if card[1] == 's':
            card[1] = 0
        elif card[1] == 'h':
            card[1] = 1
        elif card[1] == 'd':
            card[1] = 2
        else:
            card[1] = 3
    return new_cards

#takes 2d vector of numbers, turns into (1,4) matrix of numbers between 0-51
#returns np.array
def to_52_vector(vector):
    rank = np.transpose(vector)[:][0]
    suit = np.transpose(vector)[1][:]
    rank = np.subtract(rank,2)
    return np.add(rank,np.multiply(suit,13))

#takes (1,4) vector of numbers between 0-51 and turns into 2d vector of numbers between 0-13 and 1-4
#returns list
def to_2d(vector):
    if type(vector) == np.ndarray or type(vector) == list:
    #    print()
        suit = np.floor(np.divide(vector,13))
        suit = suit.astype(int)
        rank = np.subtract(vector,np.multiply(suit,13))
        rank = np.add(rank,2)
        combined = np.concatenate([rank,suit])
        length = int(len(combined) / 2)
        hand_length = len(vector)
        hand = [[combined[x],combined[x+hand_length]] for x in range(length)]
    else:
        suit = np.floor(np.divide(vector,13))
        suit = suit.astype(int)
        rank = np.subtract(vector,np.multiply(suit,13))
        rank = np.add(rank,2)
        hand = [[rank,suit]]
        print(hand,'hand')
    #print(hand,'combined')
    return hand
    
#takes (1,4) numpy vector of numbers between 0-51 and returns 1 hot encoded vector
#returns list of numpy vectors
def to_1hot(vect):
    hand = []
    for card in vect:
        vector = np.zeros(52)
        vector[card] = 1
        hand.append(vector)
    return hand

#takes (1,52) 1 hot encoded vector and makes it (1,53)
#returns np.array
def hot_pad(vector):
    temp = np.copy(vector)
    padding = np.reshape(np.zeros(len(temp)),(len(temp),1))
    temp = np.hstack((temp,padding))
    return temp

#Takes 1hot encoded vector
#returns (1,4) 52 encoded vector
def from_1hot(vect):
    new_hand = []
    for card in vect:
        #print(card)
        i = np.where(card == 1)
        #print(i)
        new_hand.append(i)
    return new_hand

#Takes 1 hot encoded padded vector and returns 1 hot encoded vector
def remove_padding(vect):
    if len(vect[0]) != 53:
        raise ValueError("1 Hot vector must be padded")
    new_hand = []
    for card in vect:
        new_hand.append(card[:-1])
    return new_hand
    
def convert_str_to_1hotpad(hand):
    vector1 = suits_to_num(hand)
    vector2 = to_52_vector(vector1)
    vector3 = to_1hot(vector2)
    vector4 = hot_pad(vector3)
    return vector4

def convert_1hotpad_to_str(hand):
    vector_unpad = remove_padding(hand)
    vector_unhot = from_1hot(vector_unpad)
    #print(vector_unhot,'vector_unhot')
    vector_un1d = to_2d(vector_unhot)
    #print(vector_un1d,'vector_un1d')
    vector_unnum = suits_to_str(vector_un1d)
    return vector_unnum

def convert_1hot_to_str(hand):
    vector_unhot = from_1hot(hand)
    #print(vector_unhot,'vector_unhot')
    vector_un1d = to_2d(vector_unhot)
    #print(vector_un1d,'vector_un1d')
    vector_unnum = suits_to_str(vector_un1d)
    for card in vector_unnum:
        card[0] = int(card[0][0][0])
    return vector_unnum

def convert_str_to_1hot(hand):
    vector1 = suits_to_num(hand)
    vector2 = to_52_vector(vector1)
    vector3 = to_1hot(vector2)
    return vector3

def convert_52_to_str(hand):
    vector_un1d = to_2d(hand)
    vector_unnum = suits_to_str(vector_un1d)
    return vector_unnum

### Durak helping functions ###
# 9 cards per suit

def to_36_vector(cards):
    new_hand = hand(len(cards))
    for card in cards:
        card[0] - 6
        if card[1] == 's':
            card[0]*4
        elif card[1] == 'h':
            card[0]*3
        elif card[1] == 'd':
            card[0]*2
    pass

def from_1hot_to_36(cards):
    #find each 1 location
    # modulo  and add 6
    pass

def convert_36_to_1hot(hand):
    vector1 = suits_to_num(hand)
    vector2 = to_36_vector(vector1)
    pass

def convert_36_1hot_to_str(hand):
    vector_unhot = from_1hot(hand)
    vector_un1d = to_2d(vector_unhot)
    vector_unnum = suits_to_str(vector_un1d)

def makefile(data,path):
    #store data
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
def return_from_hex(game_state):
    return pickle.loads(binascii.unhexlify(game_state.encode('ascii')))


# In[ ]:


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


# In[ ]:


"""
Could have all levels of abstraction within one model. Or could have them as separate models
Could train how many levels of abstraction are needed. 
Although this seems better to be selected by an outside source.
Like a genetic algorith or EA.

Have a picked_up var for each player that is an feature input

"""
def split_param_game_state(tensor):
    return tensor[:,0]

def split_param_remaining_cards(tensor):
    return tensor[:,1]

def split_param_trump_suit(tensor):
    return tensor[:,2]

def split_param_trump_card(tensor):
    return tensor[:,3:55]

def split_param_hero_hand_length(tensor):
    return tensor[:,55]

def split_param_villain_hand_length(tensor):
    return tensor[:,56]

def split_param_discard_vec(tensor):
    return tensor[:,57:109]

def split_param_played_cards(tensor):
    return tensor[:,109:161]

def split_played_card_emb(tensor):
    return tensor[:,161]

def split_param_hand(tensor):
    return tensor[:,162:214]

def split_possibilities(tensor):
    return tensor[:,214:]

#Embedding splits

def split_param_trump_card_emb(tensor):
    return tensor[:,3]

def split_param_hero_hand_length_emb(tensor):
    return tensor[:,4]

def split_param_villain_hand_length_emb(tensor):
    return tensor[:,5]

def split_param_discard_emb(tensor):
    return tensor[:,6:42]

def split_param_played_cards_emb(tensor):
    return tensor[:,42:54]

def split_param_played_card_emb(tensor):
    return tensor[:,54]

def split_param_hand_emb(tensor):
    return tensor[:,55:91]

def split_possibilities_emb(tensor):
    return tensor[:,91:]

def return_shape(input_shape):
    return (input_shape[1:] + (1,))

#(game_state_vec,remaining_cards_vec,trump_num_vec,trump_card_emb,hero_hand_length,villain_hand_length,discard_emb,played_cards_emb,played_card_emb,hand_emb,possibilities_emb))
    

def VEmbed_ab1(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_1_embeddings = Embedding(53,100,input_length=1)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    played_cards = card_36_embeddings(played_cards)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    #
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only])
    X = Dense(128)(inputs)
    X = LeakyReLU(alpha)(X)
    policy = Dense(128)(X)
    policy = LeakyReLU(alpha)(policy)
    policy_head = Dense(policy_shape)(policy)
#     print(policy_head.shape,possibility_mask.shape,'policy, possibility')
#     policy_head = multiply([policy_head,possibility_input])
#     policy_head = Activation("softmax")(policy_head)
    policy_head = Activation("softmax")(policy_head)
    print(policy_head.shape,'policy head')
    #add argmax so it directly outputs an index
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(X)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_head])
    return model


def VEmbed_ab3(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_12_embeddings = Embedding(53,100,input_length=12)
    card_1_embeddings = Embedding(53,100,input_length=1)
    played_cards_only = card_12_embeddings(played_cards)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #played cards
    played_cards_only = Dense(128,activation='relu')(played_cards_only)
    played_cards_only = Dense(1,activation='relu')(played_cards_only)
    print(played_cards_only.shape,'played_cards_only')
    played_cards_only = Reshape((12,))(played_cards_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    #policy 1
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand])
    policy = Dense(128)(inputs)
    policy = LeakyReLU(alpha)(policy)
    policy_head1 = Dense(policy_shape)(policy)
    policy_head1 = Activation("softmax")(policy_head1)
    print(policy_head1.shape,'policy head')
    
    #policy 2
    inputs2 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only])
    policy2 = Dense(128)(inputs2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy2 = Dense(128)(policy2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy_head2 = Dense(policy_shape)(policy2)
    policy_head2 = Activation("softmax")(policy_head2)
    
    #policy 3
    inputs3 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only])
    policy3 = Dense(128)(inputs)
    policy3 = LeakyReLU(alpha)(policy3)
    policy3 = Dense(128)(policy3)
    policy3 = LeakyReLU(alpha)(policy3)
    policy_head3 = Dense(policy_shape)(policy3)
    policy_head3 = Activation("softmax")(policy_head3)
    
    #final policy decision
    policies = Add()([policy_head1,policy_head2,policy_head3])#,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])
    #print(policies.shape,'policies')
    policy_final = Dense(128)(policies)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(128)(policy_final)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final_head = Dense(policy_shape,activation="softmax")(policy_final)
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(inputs3)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_final_head])
    return model

def VEmbed(input_shape,policy_shape,alpha,reg_const):
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card_emb)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length_emb,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length_emb,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_emb)(X_input)
    played_cards = Lambda(split_param_played_cards_emb)(X_input)
    played_card = Lambda(split_param_played_card_emb)(X_input)
    hand_only = Lambda(split_param_hand_emb)(X_input)
    possibility_mask = Lambda(split_possibilities_emb)(X_input)
    #Embeddings
    game_state_only = Embedding(3,25, input_length=1)(game_state_only)
    remaining_cards_only = Embedding(25,10,input_length=1)(remaining_cards_only)
    trump_suit_only = Embedding(4,10,input_length=1)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20,input_length=1)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20,input_length=1)(villain_hand_length_only)
    #Card embeddings
    card_36_embeddings = Embedding(53,100,input_length=36)
    card_12_embeddings = Embedding(53,100,input_length=12)
    card_1_embeddings = Embedding(53,100,input_length=1)
    played_cards_only = card_12_embeddings(played_cards)
    discard_vec_only = card_36_embeddings(discard_vec_only)
    hand_only = card_36_embeddings(hand_only)
    trump_card_only = card_1_embeddings(trump_card_only)
    played_card_only = card_1_embeddings(played_card)
    possibility_mask = Embedding(53,100,input_length=53)(possibility_mask)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((25,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #played card
    played_card_only = Reshape((100,))(played_card_only)
    #played cards
    played_cards_only = Dense(128,activation='relu')(played_cards_only)
    played_cards_only = Dense(1,activation='relu')(played_cards_only)
    print(played_cards_only.shape,'played_cards_only')
    played_cards_only = Reshape((12,))(played_cards_only)
    #trump card
    trump_card_only = Reshape((100,))(trump_card_only)
    #hand
    hand = Dense(128,activation='relu')(hand_only)
    hand = Dense(1,activation='relu')(hand)
    hand = Reshape((36,))(hand)
    print(hand.shape,'hand shape')
    #mask
    possibility_input = Dense(1,activation='relu')(possibility_mask)
    possibility_input = Reshape((53,))(possibility_input)
    #policy 1
    inputs = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand])
    policy = Dense(128)(inputs)
    policy = LeakyReLU(alpha)(policy)
    policy_head1 = Dense(policy_shape)(policy)
    policy_head1 = Activation("softmax")(policy_head1)
    print(policy_head1.shape,'policy head')
    
    #policy 2
    inputs2 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only])
    policy2 = Dense(128)(inputs2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy2 = Dense(128)(policy2)
    policy2 = LeakyReLU(alpha)(policy2)
    policy_head2 = Dense(policy_shape)(policy2)
    policy_head2 = Activation("softmax")(policy_head2)
    
    #policy 3
    inputs3 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only])
    policy3 = Dense(128)(inputs)
    policy3 = LeakyReLU(alpha)(policy3)
    policy3 = Dense(128)(policy3)
    policy3 = LeakyReLU(alpha)(policy3)
    policy_head3 = Dense(policy_shape)(policy3)
    policy_head3 = Activation("softmax")(policy_head3)
    
    #abstract policy4
    policy_4 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,trump_card_only,remaining_cards_only])
    #print(policy_4.shape,'policy_4')
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    #print(policy_4.shape)
    policy_head_4 = Dense(policy_shape,activation="softmax")(policy_4)
    #print(policy_head_4.shape,'policy_head_4')
    
    #abstract policy5
    policy_5 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only,discard_vec_only])
    #print(policy_5.shape,'policy_5')
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    #print(policy_5.shape)
    policy_head_5 = Dense(policy_shape,activation="softmax")(policy_5)
    #print(policy_head_5.shape,'policy_head_5')
    
    #abstract policy6
    policy_6 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,played_card_only,played_cards_only,discard_vec_only,trump_card_only,remaining_cards_only])
    #print(policy_6.shape,'policy_6')
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    #print(policy_6.shape)
    policy_head_6 = Dense(policy_shape,activation="softmax")(policy_6)
    #print(policy_head_6.shape,'policy_head_6')
    
    #abstract policy7
    policy_7 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand,discard_vec_only,remaining_cards_only])
    #print(policy_7.shape,'policy_7')
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    #print(policy_7.shape)
    policy_head_7 = Dense(policy_shape,activation="softmax")(policy_7)
    #print(policy_head_7.shape,'policy_head_7')
    
    #abstract policy8
    policy_8 = Concatenate(axis=-1)([game_state_only,trump_suit_only,trump_card_only,hand,played_card_only,played_cards_only,discard_vec_only,remaining_cards_only,hero_hand_length_only,villain_hand_length_only])
    #print(policy_8.shape,'policy_8')
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    #print(policy_8.shape)
    policy_head_8 = Dense(policy_shape,activation="softmax")(policy_8)
    #print(policy_head_8.shape,'policy_head_8')
    
    #final policy decision
    policies = Add()([policy_head1,policy_head2,policy_head3,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])#,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])
    #print(policies.shape,'policies')
    policy_final = Dense(128)(policies)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(128)(policy_final)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final_head = Dense(policy_shape,activation="softmax")(policy_final)
    
    #print(X.shape,policy_head.shape,'pre concat')
    #value = Concatenate()([inputs,policy_head])
    value = Dense(128,activation='relu')(inputs3)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_final_head])
    return model

def V1hot(input_shape,policy_shape,alpha,reg_const):
    #Has the following policies
    #all policies look at trump_suit and game_state
    #Policy that looks at hand
    #Policy that looks at hand and played_cards
    #Policy that looks at hand and trump card
    #Policy that looks at hand, discard pile and trump card
    #Policy that looks at hand, discard pile, played_cards and trump card
    #Policy that looks at hand, remaining deck length, played_cards and trump card
    #Policy that looks at hand, discard pile, remaining deck length, played_cards and trump card
    #Policy that looks at everything
    #Input shape = 213 (without picked up cards vec or a villain hand vec)
    #game_state_vec,remaining_cards_vec,trump_num_vec,features.trump_card_vec,hero_hand_length,villain_hand_length,discard_vec,played_cards_vec,hand)
    #0 = game_state_vec (0-2)(1)
    #1 = remaining_cards_vec (0-26)(1)
    #2 = trump_suit_vec (0-3)(1)
    #3-55 = trump_card_vec (52)
    #55-56 = hero_hand_length (0-36)(1)
    #56-57 = villain_hand_length (0-36)(1)
    #57-109 = discard_vec (52)
    #109-161 = played_cards_vec (12)(52)
    #161-213 = hand_vec
    
    X_input = Input(input_shape)
    #split input
    game_state_only = Lambda(split_param_game_state,output_shape=(1,1))(X_input)
    remaining_cards_only = Lambda(split_param_remaining_cards,output_shape=(1,1))(X_input)
    trump_suit_only = Lambda(split_param_trump_suit,output_shape=(1,1))(X_input)
    trump_card_only = Lambda(split_param_trump_card)(X_input)
    hero_hand_length_only = Lambda(split_param_hero_hand_length,output_shape=(1,1))(X_input)
    villain_hand_length_only = Lambda(split_param_villain_hand_length,output_shape=(1,1))(X_input)
    discard_vec_only = Lambda(split_param_discard_vec)(X_input)
    played_cards = Lambda(split_param_played_cards)(X_input)
    hand_only = Lambda(split_param_hand)(X_input)
    possibility_mask = Lambda(split_possibilities)(X_input)
    #print(hand_only.shape,'hand shape')
    #Embeddings
    game_state_only = Embedding(3,10)(game_state_only)
    remaining_cards_only = Embedding(25,10)(remaining_cards_only)
    trump_suit_only = Embedding(4,10)(trump_suit_only)
    hero_hand_length_only = Embedding(37,20)(hero_hand_length_only)
    villain_hand_length_only = Embedding(37,20)(villain_hand_length_only)
    #reshape the 1 dimensional outputs
    game_state_only = Reshape((10,))(game_state_only)
    remaining_cards_only = Reshape((10,))(remaining_cards_only)
    trump_suit_only = Reshape((10,))(trump_suit_only)
    hero_hand_length_only = Reshape((20,))(hero_hand_length_only)
    villain_hand_length_only = Reshape((20,))(villain_hand_length_only)
    #everything
    X = Dense(128)(X_input)
    X = LeakyReLU(alpha)(X)
    X = Dense(128)(X)
    X = LeakyReLU(alpha)(X)
    #print(X.shape,'X')
    
    #Available actions -> emb -> dot with FC -> softmax -> sample/argmax -> action
    
    #full policy
    policy_1 = Dense(128)(X)
    policy_1 = LeakyReLU(alpha)(policy_1)
    policy_1 = Dense(128)(policy_1)
    policy_1 = LeakyReLU(alpha)(policy_1)
    #print(policy_1.shape)
    policy_head_1 = Dense(policy_shape,activation="softmax")(policy_1)
    #print(policy_head_1.shape,'policy_head_1')
    #add argmax so it directly outputs an index
    
    #abstract policy2
    policy_2 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only])
    #print(policy_2.shape,'policy_2')
    policy_2 = Dense(128)(policy_2)
    policy_2 = LeakyReLU(alpha)(policy_2)
    policy_2 = Dense(128)(policy_2)
    policy_2 = LeakyReLU(alpha)(policy_2)
    #print(policy_2.shape)
    policy_head_2 = Dense(policy_shape,activation="softmax")(policy_2)
    #print(policy_head_2.shape,'policy_head_2')
    
    #abstract policy3
    policy_3 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,played_cards])
    #print(policy_3.shape,'policy_3')
    policy_3 = Dense(128)(policy_3)
    policy_3 = LeakyReLU(alpha)(policy_3)
    policy_3 = Dense(128)(policy_3)
    policy_3 = LeakyReLU(alpha)(policy_3)
    #print(policy_3.shape)
    policy_head_3 = Dense(policy_shape,activation="softmax")(policy_3)
    #print(policy_head_3.shape,'policy_head_3')
    
    #abstract policy4
    policy_4 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,trump_card_only])
    #print(policy_4.shape,'policy_4')
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    policy_4 = Dense(128)(policy_4)
    policy_4 = LeakyReLU(alpha)(policy_4)
    #print(policy_4.shape)
    policy_head_4 = Dense(policy_shape,activation="softmax")(policy_4)
    #print(policy_head_4.shape,'policy_head_4')
    
    #abstract policy5
    policy_5 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only])
    #print(policy_5.shape,'policy_5')
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    policy_5 = Dense(128)(policy_5)
    policy_5 = LeakyReLU(alpha)(policy_5)
    #print(policy_5.shape)
    policy_head_5 = Dense(policy_shape,activation="softmax")(policy_5)
    #print(policy_head_5.shape,'policy_head_5')
    
    #abstract policy6
    policy_6 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only,trump_card_only,remaining_cards_only])
    #print(policy_6.shape,'policy_6')
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    policy_6 = Dense(128)(policy_6)
    policy_6 = LeakyReLU(alpha)(policy_6)
    #print(policy_6.shape)
    policy_head_6 = Dense(policy_shape,activation="softmax")(policy_6)
    #print(policy_head_6.shape,'policy_head_6')
    
    #abstract policy7
    policy_7 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only,remaining_cards_only])
    #print(policy_7.shape,'policy_7')
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    policy_7 = Dense(128)(policy_7)
    policy_7 = LeakyReLU(alpha)(policy_7)
    #print(policy_7.shape)
    policy_head_7 = Dense(policy_shape,activation="softmax")(policy_7)
    #print(policy_head_7.shape,'policy_head_7')
    
    #abstract policy8
    policy_8 = Concatenate(axis=-1)([game_state_only,trump_suit_only,hand_only,discard_vec_only,discard_vec_only,remaining_cards_only])
    #print(policy_8.shape,'policy_8')
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    policy_8 = Dense(128)(policy_8)
    policy_8 = LeakyReLU(alpha)(policy_8)
    #print(policy_8.shape)
    policy_head_8 = Dense(policy_shape,activation="softmax")(policy_8)
    #print(policy_head_8.shape,'policy_head_8')
    
    #final policy decision
    policies = Add()([policy_head_1,policy_head_2,policy_head_3,policy_head_4,policy_head_5,policy_head_6,policy_head_7,policy_head_8])
    #print(policies.shape,'policies')
    policy_final = Dense(128)(policies)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(128)(policy_final)
    policy_final = LeakyReLU(alpha)(policy_final)
    policy_final = Dense(policy_shape,activation="softmax")(policy_final)
    
    #print(X.shape,policy_head.shape,'pre concat')
    value = Concatenate(axis=-1)([X,policies])
    value = Dense(128,activation='relu')(value)
    value = Dense(128,activation='relu')(value)
    value_head = Dense(1,activation='tanh')(value)
    
    model = Model(inputs = X_input, outputs = [value_head,policy_final])
    return model


# In[ ]:


"""
Possibility Vector is length 53. 52 card choices + picking up the card (as defender)
In the event the defender picks up cards, 53rd option for attacker is pass on giving more cards. 

TODO
Mask possibility vector in the model instead of masking before input
Switch to 37 length vectors
flatten discard pile
add a cards given to opponent vector so that the bot knows what cards the opponent has if any.

"""

class Durak(object):
    def __init__(self,deck,models,funcs,threshold=100,num_players=2,trunk=None,play=False,tournament=False):
        self.deck = deck
        self.num_players = num_players
        self.players = []
        self.models = models
        self.threshold = threshold
        self.play = play
        if self.play == False:
            self.models[-1]._make_predict_function()
            self.models[-2]._make_predict_function()
        self.funcs = funcs
        self.results = [0,0]
        self.possibilities = np.arange(53)
        self.epsilon = 0.0001
        self.tournament = tournament
        #reusing previously generated tree
        if trunk == None:
            self.trunk = Tree()
        else:
            self.trunk = trunk
        
    def save_tree(self,tree_path):
        with open(tree_path, 'wb') as handle:
            pickle.dump(self.trunk, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def load_tree(self,tree_path):
        with open(tree_path, 'rb') as handle:
            tree = pickle.load(handle)
        self.trunk = tree
        
    def start_from_state(self,game_state_dict):
        self.play_deck = copy.deepcopy(game_state_dict['deck'])
        self.players = []
        #player1
        hand1 = copy.deepcopy(game_state_dict['hand1'])
        print(hand1,'hand1')
        hand_1hot1 = convert_str_to_1hot(hand1)
        pl1 = Player(hand1,hand_1hot1,0)
        self.players.append(pl1)
        #player2
        hand2 = copy.deepcopy(game_state_dict['hand2'])
        print(hand2,'hand2')
        hand_1hot2 = convert_str_to_1hot(hand2)
        pl2 = Player(hand2,hand_1hot2,1)
        self.players.append(pl2)
        #
        trump_card = copy.deepcopy(game_state_dict['trump_card'])
        trump_card_vec = convert_str_to_1hot([trump_card])[0]
        trump_suit = trump_card[1]
        attacking_player = copy.deepcopy(game_state_dict['attacking_player'])
        previous_winner = (False,0)
        first_root = self.trunk.get_child(attacking_player)
        second_root = self.trunk.get_child((attacking_player + 1)%2)

        discard_pile = copy.deepcopy(game_state_dict['discard_pile'])
        discard_pile_1hot = copy.deepcopy(game_state_dict['discard_pile_1hot'])
        self.game_state = Game_state(trump_card,trump_card_vec,trump_suit,len(self.play_deck),attacking_player,previous_winner,first_root,second_root,attacking_player,discard_pile=discard_pile,discard_pile_1hot=discard_pile_1hot)
        
    def init_game(self,previous_winner):
        self.play_deck = copy.deepcopy(self.deck)
        shuffle(self.play_deck)
        self.players = []
        for player_id in range(self.num_players):
            #switching to dealing off the front. Then we can append the trump card to the end
            hand = self.play_deck[:6]
            hand_1hot = convert_str_to_1hot(hand)
            del self.play_deck[:6]
            pl = Player(hand,hand_1hot,player_id)
            self.players.append(pl)
        trump_card = self.play_deck[0]
        trump_card_vec = convert_str_to_1hot([trump_card])[0]
        trump_suit = trump_card[1]
#         print(trump_card,'trump card')
        self.play_deck.append(self.play_deck[0])
        self.play_deck.pop(0)
        attacking_player = self.who_starts(previous_winner) #This is better for training, later we can make it switch
        #Tree
        first_root = self.trunk.get_child(attacking_player)
        second_root = self.trunk.get_child((attacking_player + 1)%2)
        #Instantiate game state
        self.game_state = Game_state(trump_card,trump_card_vec,trump_suit,len(self.play_deck),attacking_player,previous_winner,first_root,second_root,attacking_player,discard_pile=[],discard_pile_1hot=[])
        
    def update_game_state(self):
        self.game_state.remaining_cards = len(self.play_deck)
        self.game_state.played_cards = []
        self.game_state.played_cards_1hot = []
        self.game_state.played_card = None
        self.game_state.played_card_1hot = None
        self.game_state.picked_up = False
        
    def int_from_1hot(self,card):
        integer = np.where(card==1)[0]
        
    def print_game_state(self):
        print(self.game_state.remaining_cards,'remaining cards')
        print(self.game_state.played_cards,'played cards')
        print(len(self.game_state.played_cards_1hot),'len played cards 1hot')
        print(self.game_state.played_card,'played card')
        #print(self.game_state.played_card_1hot,'played card 1hot')
        print(self.game_state.discard_pile,'discard_pile')
        #print(self.game_state.discard_pile_1hot.shape,'shape of discard_pile_1hot')
        print(self.game_state.attacking_player,'attacking_player')
        print(self.game_state.defending_player,'defending_player')
        for player in self.players:
            print(len(player.hand),'hand length')
        
    def print_player_attributes(self):
        for player in self.players:
            print(len(player.hand),'hand')
            #print(convert_1hot_to_str(player.hand_1hot),'1hot')
        
    def draw_cards(self,player_id):
        if len(self.play_deck) > 0:
            while len(self.players[player_id].hand) < 6 and len(self.play_deck) > 0:
                self.players[player_id].hand.append(self.play_deck.pop(0))
            self.players[player_id].hand_1hot = convert_str_to_1hot(self.players[player_id].hand)
        
    def who_starts(self,previous_winner):
        if previous_winner[0] == True:
            return previous_winner[1]
        else:
            #check for lowest trump, if no trump then check for lowest card. if tie then split based on suit
            return choice([0,1])
            
    def play_game(self):
        #increase root node visit counts
        self.game_state.first_root.update_visit()
        self.game_state.second_root.update_visit()
        while self.game_state.game_over == False:
            self.print_game_state() #PRINT STATEMENT
            self.attack()
            if self.game_state.played_card:
#                 print('defending')
                self.defend()
            #check if round is over, else continue
            #self.print_player_attributes()
            self.is_round_over()
            if self.game_state.round_over == True:
                self.update_game_state
                self.is_game_over()
        self.record_outcome()
            
    #to account for giving multiple cards. I'll just check for the taking condition 
    #and then allow the attacker to give up to 6 cards
    def attack(self):
#         print('attacking')
        if self.game_state.played_cards:
            #not first action: We will mask our possibility vector with the like rank cards already played
            #53 because of the possibility of picking up the cards (needs to be consistent for the model)
            possibility_vec = np.zeros((1,53)).flatten()
            possibility_vec[52] = 1
            ranks = []
            [ranks.append(card[0]) for card in self.game_state.played_cards]
#             print(ranks,'ranks')
            for i in range(len(self.players[self.game_state.attacking_player].hand)):
                if self.players[self.game_state.attacking_player].hand[i][0] in ranks:
                    location = np.where(self.players[self.game_state.attacking_player].hand_1hot[i]==1)[0]
                    possibility_vec[location] = 1
        else:
            #first action, pick any card Need to locate each card in terms of the entire deck.
            #the possibility vector will be of length 36. with 1s for cards we have and 0s for the rest
#             print('first action')
            #Should just be the hand_1hot vector plus 0
            possibility_vec = np.zeros((1,53)).flatten()
            for i in range(len(self.players[self.game_state.attacking_player].hand)):
                location = np.where(self.players[self.game_state.attacking_player].hand_1hot[i]==1)[0]
                possibility_vec[location] = 1
        #inputs = (self.players[player_id].hand_1hot,self.game_state)
        #wrap all options plus game state into something to pass into model
        #if first action can play any card, else has to match
        #decision = self.funcs[self.game_state.attacking_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.attacking_player].hand_1hot),self.players,self.game_state.attacking_player,0,self)
        decision = self.funcs[self.game_state.attacking_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.attacking_player].hand_1hot),self.players,self.game_state.attacking_player,0,self)
#         print(decision,'attacking decision')
        if decision == 52:
            #player passed - This is redundent safety
            self.game_state.played_card = None
            self.game_state.played_card_1hot = None
        else:
            #remove card from hand add to played list
            self.remove_card(True,self.game_state.attacking_player,decision)
    
    #defender has the option to defend with any legal card, or take the card
    #Needs current played cards
    def defend(self):
        player_id = self.game_state.defending_player
        possibility_vec = np.zeros((1,53)).flatten()
        attacker_card = self.game_state.played_card
        print(attacker_card,'attacker_card')
#         [print(card) for card in self.players[self.game_state.defending_player].hand]
#         print('defending players hand')
        #Go through hand and see if any cards can beat it
        for i in range(len(self.players[self.game_state.defending_player].hand)):
            if self.players[player_id].hand[i][0] > attacker_card[0] and self.players[player_id].hand[i][1] == attacker_card[1]            or attacker_card[1] != self.game_state.trump_suit and self.players[player_id].hand[i][1] == self.game_state.trump_suit:
                #possible defend
                location = np.where(self.players[self.game_state.defending_player].hand_1hot[i]==1)[0]
#                 print(location,'defending location')
#                 print(self.players[self.game_state.defending_player].hand[i],'possible card')
                possibility_vec[location] = 1
                i += 1
        #check any cards can defend, otherwise auto pickup
        ### Streamline this later ###
#         if 1 not in possibility_vec:
#             #can't defend, pick up the card
#             picking_up_cards()
#         else:
#             #Add ability to pick up the cards 
#             #pass possibility vec through to model, get action
        ###
        possibility_vec[52] = 1
        decision = self.funcs[self.game_state.defending_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.defending_player].hand_1hot),self.players,self.game_state.defending_player,1,self)
#         print(decision,'defend decision')
        if decision == 52:
            self.picking_up_cards()
        else:
            #chose defend
            #remove that card from hand, add to played list
            self.remove_card(False,self.game_state.defending_player,decision)
            
    def picking_up_cards(self):
        print('picking up cards')
        self.print_game_state() ## PRINT STATEMENT
        #add all played cards to hand, check if opponent wants to add any more cards?
        #Check which cards are legal up to the remaining cards in defender's hand
        #self.game_state.played_cards.flatten()
        max_cards = len(self.players[self.game_state.defending_player].hand_1hot)
        
        while max_cards > 0 and len(self.players[self.game_state.attacking_player].hand) > 0:
            possibility_vec = np.zeros((1,53)).flatten()
            possibility_vec[52] = 1
#             if not self.game_state.played_cards:
#                 ranks = self.game_state.played_card[0]
           
            ranks = [card[0] for card in self.game_state.played_cards]
#             print(ranks,'ranks')
            for i in range(len(self.players[self.game_state.attacking_player].hand)):
                if self.players[self.game_state.attacking_player].hand[i][0] in ranks:
                    #get 1hot location
                    location = np.where(self.players[self.game_state.attacking_player].hand_1hot[i]==1)[0]
                    possibility_vec[location] = 1
            decision = self.funcs[self.game_state.attacking_player](possibility_vec,self.game_state,np.vstack(self.players[self.game_state.attacking_player].hand_1hot),self.players,self.game_state.attacking_player,2,self)
            if decision == 52:
                #to give more cards or not
                break
            else:
                #add card to self.game_state.played_cards
                self.remove_card(False,self.game_state.attacking_player,decision)
                max_cards -= 1
        #add cards to defender's hand
#         print(self.game_state.played_cards,'played cards')
        self.players[self.game_state.defending_player].hand = self.players[self.game_state.defending_player].hand+self.game_state.played_cards
        self.players[self.game_state.defending_player].hand_1hot = self.players[self.game_state.defending_player].hand_1hot+self.game_state.played_cards_1hot
        #clear played cards
        self.game_state.played_cards = []
        self.game_state.played_cards_1hot = []
        self.game_state.played_card = None
        self.game_state.played_card_1hot = None
        self.game_state.picked_up = True
    
    def remove_card(self,attack,player_id,decision):
        #remove card from player's hand
#         print(attack,player_id,decision,'remove_card vars')
        for i in range(len(self.players[player_id].hand)):
            if np.where(self.players[player_id].hand_1hot[i]==1) == decision:
                #add card to played cards list and played card
                if attack == True:
                    self.game_state.played_card = self.players[player_id].hand[i]
                    self.game_state.played_card_1hot = self.players[player_id].hand_1hot[i]
                    self.game_state.played_cards.append(self.game_state.played_card)
                    self.game_state.played_cards_1hot.append(self.game_state.played_card_1hot)
                    #remove card from hand
                    self.players[player_id].hand.pop(i)
                    self.players[player_id].hand_1hot.pop(i)
                else:
                    self.game_state.played_cards.append(self.players[player_id].hand[i])
                    self.game_state.played_cards_1hot.append(self.players[player_id].hand_1hot[i])
                    self.players[player_id].hand.pop(i)
                    self.players[player_id].hand_1hot.pop(i)
                break
        
    def is_round_over(self):
#         print('is round over')
        #is either player out of cards
        #can the attacker continue to play
        #has the defender picked up the cards
        #then check for whether either player needs to draw up to 6
        #check for number of defender's cards
        current_max = len(self.players[self.game_state.attacking_player].hand)
        number_of_attacks = len(self.game_state.played_cards) % 2
        if number_of_attacks == 6 or current_max == 0 or self.game_state.played_card == None:
            #round is over
            self.game_state.round_over = True
            if self.game_state.picked_up == True:
                #attacker draws
                if len(self.play_deck) > 0:
                    self.draw_cards(self.game_state.attacking_player)
                else:
                    pass
            else:
#                 print('successful defense')
                #successful defense, both draw
                self.draw_cards(self.game_state.attacking_player)
                self.draw_cards(self.game_state.defending_player)
                #update attacker and defender
                self.game_state.attacking_player = (self.game_state.attacking_player + 1) % 2
                self.game_state.defending_player = (self.game_state.defending_player + 1) % 2
                #Move played cards to discard pile
                if self.game_state.played_cards:
                    self.game_state.discard_pile.append(self.game_state.played_cards)
                    self.game_state.discard_pile_1hot.append(self.game_state.played_cards_1hot)
            self.update_game_state()
        else:
            #go to additional rounds
            pass
        
    def is_game_over(self):
        for player in self.players:
            if len(player.hand) == 0 and self.game_state.remaining_cards == 0:
                self.game_state.game_over = True
        
    def record_outcome(self):
        if len(self.players[0].hand) == 0 and len(self.players[1].hand) == 0:
            #tie
            self.players[0].outcome = 0
            self.players[1].outcome = 0
            self.game_state.previous_winner = (False,0)
        elif len(self.players[0].hand) == 0:
            self.players[0].outcome = 1
            self.players[1].outcome = -1
            self.results[0] += 1
            self.results[1] -= 1
            self.game_state.previous_winner = (True,0)
        else:
            self.players[0].outcome = -1
            self.players[1].outcome = 1
            self.results[0] -= 1
            self.results[1] += 1
            self.game_state.previous_winner = (True,1)
#         print('game over')
            
#Decision tree
#each action will be a path to a node. 
#after each action, there will be a new game state. We can stringify the game_state. Each will be unique
#So there will be two stages between every action
#gamestate update
# hand_round_info.SB_node = find_cards(gen_board,hand_round_info.SB_node)
# hand_round_info.BB_node = find_cards(gen_board,hand_round_info.BB_node)
# node = root.get_child(key,sub_key)
#action update
# hand_round_info.SB_node = hand_round_info.SB_node.get_child(location)
# hand_round_info.BB_node = hand_round_info.BB_node.get_child(location)
# hand_round_info.SB_root.update_visit()
# hand_round_info.BB_root.update_visit()
        
#Necessary information
#Trump card. 
#Number of cards remaining in the deck
#Number of cards in hand and opponent's hand
#cards in hand
#Whose turn it is

class Game_state(object):
    def __init__(self,trump_card,trump_card_vec,trump_suit,remaining_cards,attacking_player,previous_winner,first_root,second_root,first_player,discard_pile=[],discard_pile_1hot=[]):
        self.trump_suit = trump_suit
        self.trump_card = trump_card
        self.trump_num = (suits_to_num([self.trump_card])[0][1])
        self.trump_card_vec = trump_card_vec
        self.remaining_cards = remaining_cards
        self.attacking_player = attacking_player
        self.defending_player = (self.attacking_player + 1) % 2
        self.played_cards = []
        self.played_cards_1hot = []
        self.played_card = None
        self.played_card_1hot = None
        self.discard_pile = discard_pile
        self.discard_pile_1hot = discard_pile_1hot
        self.round_over = False
        self.previous_winner = previous_winner
        self.game_over = False
        self.picked_up = False
        self.first_root = first_root
        self.second_root = second_root
        self.first_player = first_player
        self.first_node = self.first_root
        self.second_node = self.second_root
        
    def update_player_turn(self):
        self.player_turn = (self.player_turn + 1) % 2
        
class Player(object):
    def __init__(self,hand,hand_1hot,player_id):
        self.hand = hand
        self.hand_1hot = hand_1hot
        self.player_id = player_id
        self.outcome = 0
        self.results = 0

def return_emb_vector(durak,features,game_state,hand_vec,possibility_vec,player_id):
    #making everything for embeddings
    #hand nums will be len 36 vec of ints 0-53
    #played_cards_vec = (1,12) vec of ints 0-53
    #discard_pile_vec = (1,36) vec of ints 0-53
    #game_state_vec = int 0-2
    #hero_hand_length = int 0-36
    #villain_hand_length = int 0-36
    #remaining_cards_vec = int 0-24
    #trump suit = int 0-3
    #trump vec = int 0-52
    #possibility_vec = 53
    #######
    #input_vec = (1,143)
    possibilities = np.where(possibility_vec==1)[0]
#     pad = 53 - int(possibilities.shape[0])
#     print(pad,'pad')
#     padding = np.full(pad,52)
#     possibilities_input = np.hstack((possibilities,padding))
    game_state_vec = np.array(game_state)
    remaining_cards_vec = np.array(features.remaining_cards)
    trump_num_vec = np.array(features.trump_num)
    if len(features.discard_pile_1hot) > 0:
        discard_vec = np.zeros(52)
        temp = np.vstack(features.discard_pile_1hot)
        locats = np.where(temp == 1)[1]
        discard_vec[locats] = 1
        discard_loc = np.where(discard_vec == 1)[0]
        print(discard_loc,'discard_loc')              
        temp_pad = 36 - len(discard_loc)
        padding = np.full(temp_pad,52)
        discard_emb = np.hstack((discard_loc,padding))
    else:
        discard_emb = np.full(36,52)
    #Played cards
    if len(features.played_cards_1hot):
        played_cards_vec = np.zeros(52)
        played_cards_temp = np.vstack(features.played_cards_1hot)
#         print(len(features.played_cards_1hot))
#         print(features.played_cards_1hot,'features.played_cards_1hot')
#         print(np.where(features.played_cards_1hot[0] == 1),'prior to locats')
        locats = np.where(played_cards_temp == 1)[1]
#         print(locats,'locats')
        played_cards_vec[locats] = 1
        temp_pad = 12 - len(locats)
        padding = np.full(temp_pad,52)
        played_cards_emb = np.hstack((locats,padding))
    else:
        played_cards_emb = np.full(12,52)
    #Played card
    if features.played_card:
        played_card_emb = np.where(features.played_card_1hot == 1)[0]
    else:
        played_card_emb = np.array(52)
        
    hero_hand_length = np.array(len(durak.players[player_id].hand))
    villain_hand_length = np.array(len(durak.players[(player_id+1)%2].hand))
    #
    hand_loc = np.where(hand_vec == 1)[1]
    print(hand_loc,'hand_loc')
    temp_pad = 36 - len(hand_loc)
    padding = np.full(temp_pad,52)
#     print(hand_loc.shape,padding.shape,'handemb')
    hand_emb = np.hstack((hand_loc,padding))
    trump_card_emb = np.where(features.trump_card_vec == 1)[0]
    #possibility emb
    temp_pad = 53 - len(possibilities)
    padding = np.full(temp_pad,52)
    possibilities_emb = np.hstack((possibilities,padding))
#     print(trump_card_emb,'trump_card_emb')
    print(played_card_emb,'played_card_emb',played_cards_emb.shape,'played_cards_emb',discard_emb.shape,'discard_emb',hand_emb.shape,'hand_emb',trump_card_emb.shape,'trump_card_emb',possibilities_emb.shape,'possibilities_emb')
    emb_vector = np.hstack((game_state_vec,remaining_cards_vec,trump_num_vec,trump_card_emb,hero_hand_length,villain_hand_length,discard_emb,played_cards_emb,played_card_emb,hand_emb,possibilities_emb))
    #1+1+1+1+1+1+36+12+1+36+53
    #technically number of possibilities should be 37?
    #print(emb_vector.shape,'emb_vector')
    model_emb_input = emb_vector.reshape(1,144)
    return model_emb_input
    

def return_1hot_vector(durak,features,game_state,hand_vec,possibility_vec,player_id):
    #remaining cards = int 0-24
    #hand_vec = (1,36) vec of ints 0-53
    #played_cards_vec = (1,12) vec of ints 0-53
    #discard_pile_vec = (1,36) vec of ints 0-53
    #played card = (1) 52
    #played cards = (12) 52
    #trump suit = int 0-3
    #trump vec = int 0-52
    #input_vec = (1,266)
    unknown_card = np.array(53)
    possibilities = np.where(possibility_vec==1)[0]
    game_state_vec = np.array(game_state)
    remaining_cards_vec = np.array(features.remaining_cards)
    trump_num_vec = np.array(features.trump_num)
    if durak.game_state.played_card:
        played_card_vec = durak.game_state.played_card_1hot
    else:
        played_card_vec = np.array(53)
    #discards
    discard_vec = np.zeros(52)
    if len(features.discard_pile_1hot) > 1:
        temp = np.vstack(features.discard_pile_1hot)
        locats = np.where(temp == 1)[1]
        discard_vec[locats] = 1
    played_cards_vec = np.zeros(52)
    if len(features.played_cards_1hot):
        locats = np.where(features.played_cards_1hot[0] == 1)
        played_cards_vec[locats] = 1
    hand = np.zeros(52)
    hand_nums = np.where(hand_vec == 1)[1]
    hand[hand_nums] = 1
    hero_hand_length = np.array(len(durak.players[player_id].hand))
    villain_hand_length = np.array(len(durak.players[(player_id+1)%2].hand))
    input_vector_1hot = np.hstack((game_state_vec,remaining_cards_vec,trump_num_vec,features.trump_card_vec,hero_hand_length,villain_hand_length,discard_vec,played_cards_vec,hand,possibility_vec)) 
    model_input = input_vector_1hot.reshape(1,266)
    return model_input
        
def random_model(possibility_vec,features,hand_vec,players,player_id,game_state,durak):
    #prepare all the inputs for the model
    #pad possibilities for inputs. Embeddings only work this way.
    unknown_card = np.array(53)
    possibilities = np.where(possibility_vec==1)[0]
    model_emb_input = return_emb_vector(durak,features,game_state,hand_vec,possibility_vec,player_id)
    #model_1hot_input = return_1hot_vector(durak,features,game_state,hand_vec,possibility_vec,player_id)
    #print(model_emb_input.shape,'model_emb_input')
    #print(model_1hot_input.shape,'model_1hot_input')
    #print(model_emb_input)
    #For splitting between attack and defend models
    #If defend call defend model
    if game_state == 1:
        print('defending')
        model_ev,model_action = durak.models[1].predict(model_emb_input)
    else:
        #call attack model for states 0 and 2
        print('attacking')
        model_ev,model_action = durak.models[0].predict(model_emb_input)
    #for splitting between players
#     if player_id == 0:
#         model_ev,model_action = durak.models[0].predict(model_emb_input)
#     else:
#         model_ev,model_action = durak.models[1].predict(model_emb_input)
        
    #For 1hot inputs
#     masked_choices = np.multiply(model_action,possibility_vec)
#     probability_vector = np.divide(masked_choices,np.sum(masked_choices)).flatten()
#     else:
#     model_ev,model_action = durak.models[player_id].predict(model_emb_input)
    #print(model_action.reshape(53,),'model dist')
    #for collecting end states
#     if features.remaining_cards ==0:
#     print(model_action,'model_action')
    print(player_id,'player_id')
    print(features.played_card,'played_card')
    print(features.played_cards,'played_cards')
    print(players[0].hand,'player1 hand')
    print(players[1].hand,'player2 hand')
#         print(features.discard_pile_1hot,'1hot discard')
#         print(features.trump_card,'trump')
        #print(features.play_deck)
    model_action = np.add(model_action,durak.epsilon)
    masked_choices = np.multiply(model_action.reshape(53,),possibility_vec)
    probability_vector = np.divide(masked_choices,np.sum(masked_choices)).flatten()
#     print(possibility_vec,'possibility_vec')
    print(model_ev,'model_ev')
    print(probability_vector,'probability_vector')
#     print(possibilities,'possibilities')
#     print(features.first_player,'starting player id')
#     print(durak.possibilities.shape,probability_vector.shape,'should be equal')
#     print(np.sum(probability_vector),'prob',np.sum(masked_choices),'mask')
    #Exploring with decisions
    if durak.tournament == False:
        dice_roll = randint(0,100)
    #     print(dice_roll,'dice_roll')
        if dice_roll < durak.threshold:
            decision = np.random.choice(durak.possibilities,p=probability_vector)
        else:
            decision = choice(possibilities)
    else:
        #Pure decisions
        #decision = durak.possibilities[np.argmax(probability_vector)]
        decision = np.random.choice(durak.possibilities,p=probability_vector)
    print(decision,'decision')
#     if durak.embedding != True:
#     bytestream = pickle.dumps(input_vector_1hot)
#     hex_data = binascii.hexlify(bytestream)
#     key = hex_data.decode('ascii')
#     subkey = str(decision)
#     else:
    bytestream = pickle.dumps(model_emb_input)
    hex_data = binascii.hexlify(bytestream)
    key = hex_data.decode('ascii')
    subkey = str(decision)
#     print(key,'key')
#     print(subkey,'subkey')
    if player_id == features.first_player:
        print(player_id,'first triggered')
        features.first_node = features.first_node.get_child(key)
        features.first_node.update_visit()
        features.first_node.kind = 1
        features.first_node.game_state = game_state
        features.first_node.key = key
        
        features.first_node = features.first_node.get_child(subkey)
        features.first_node.update_visit()
        features.first_node.kind = 0
    else:
        print(player_id,'second_node triggered')
        features.second_node = features.second_node.get_child(key)
        features.second_node.update_visit()
        features.second_node.kind = 1
        features.second_node.game_state = game_state
        features.second_node.key = key
        
        features.second_node = features.second_node.get_child(subkey)
        features.second_node.update_visit()
        features.second_node.kind = 0
    return decision

def player(possibility_vec,features,hand_vec,players,player_id,game_state,durak):
    #need to display card values
    locations = np.where(possibility_vec==1)[0]
    #print(players[player_id].hand,'hand')
    #print 2d representations of locations
    #print(locations[-1],'locations[-1]')
    if locations[-1] == 52:
        cards_to_display = convert_52_to_str(locations[:-1])
        if game_state == 0 or game_state == 2:
            cards_to_display.append('pass')
        elif game_state == 1:
            cards_to_display.append('pick up')
    else:
        cards_to_display = convert_52_to_str(locations)
    print(features.trump_suit,'trump suit')
    print(cards_to_display,'cards')
    print(locations,'locations')
#     print(np.arange(len(locations)))
    location = int(input('enter location'))
    print(location,'location',type(location))
    index = np.where(locations == location)
    print(index,'index')
    print(locations[index],'choice')
    
    #Update Tree. Only for the player with the decision
    #key = gamestate
    #subkey = action
    model_emb_input = return_emb_vector(durak,features,game_state,hand_vec,possibility_vec,player_id)
    #model_1hot_input = return_1hot_vector(durak,features,game_state,hand_vec,possibility_vec,player_id)
    bytestream = pickle.dumps(model_emb_input)
    hex_data = binascii.hexlify(bytestream)
    key = hex_data.decode('ascii')
    subkey = str(locations[index])
    if player_id == features.first_player:
        features.first_node = features.first_node.get_child(key)
        features.first_node.update_visit()
        features.first_node.kind = 1
        features.first_node.gamestate = key
        features.first_node = features.first_node.get_child(subkey)
        features.first_node.update_visit()
        features.first_node.kind = 0
    else:
        features.second_node = features.second_node.get_child(key)
        features.second_node.update_visit()
        features.second_node.kind = 1
        features.second_node.gamestate = key
        features.second_node = features.second_node.get_child(subkey)
        features.second_node.update_visit()
        features.second_node.kind = 0
        
    return locations[index]
        


# In[ ]:


#Training parameters
num_epochs = 1
iterations = 250
training_cycles = 1
model_checkpoint = 500
#Threshold of randomness. 0 = completely random. 100 = entirely according to the model output
threshold = 50
#model dirs
# first_model_dir = os.path.join(os.path.dirname(sys.argv[0]), "Durak_models/First_models/")
# second_model_dir = os.path.join(os.path.dirname(sys.argv[0]), "Durak_models/Second_models/")
tree_path = '/Users/Shuza/Code/Durak/Tree/durak_tree'
attack_models_dir = '/Users/Shuza/Code/Durak/attack_models'
defend_models_dir = '/Users/Shuza/Code/Durak/defend_models'
attack_model_path = os.path.join(attack_models_dir,'attack_model')
defend_model_path = os.path.join(defend_models_dir,'defend_model')
#trunk = openpickle(tree_path)
#model input dimensions for hand generation
model_input = (266,)
model_emb_input = (144,)
policy_shape = (53)
intermediate_input = (1,14,1)
attribute_input = (1,1)
#model hyper parameters. Needs tuning
alpha = 0.002
reg_const = 0.0001
learning_rate=0.002
opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
#(input_shape,policy_shape,alpha)
model_attack = VEmbed(model_emb_input,policy_shape,alpha,reg_const)
model_defend = VEmbed(model_emb_input,policy_shape,alpha,reg_const)
# model_first = V1abstract1(model_input,policy_shape,alpha,reg_const)
# model_second = V1abstract1(model_input,policy_shape,alpha,reg_const)
model_attack.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
model_attack.summary()
# model_second.summary()


# In[ ]:


#play vs a specific model
#model_first = load_model('/Users/Shuza/Code/Durak/First_models/first_model1000')
model_attack = 'hi'
model_defend = load_model('/Users/Shuza/Code/Durak/Second_models/second_model1000')
#model_first.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
model_list = [model_attack,model_defend]
function_list = [player,random_model]
durak = Durak(deck,model_list,function_list,threshold,play=True,tournament=True)
previous_winner = (False,0)
durak.init_game(previous_winner)
durak.play_game()


# In[ ]:


#Testing a particular situation
tree_path = '/Users/Shuza/Code/Durak/Tree/durak_tree_endgame'
deck_pos = []
trump_card_pos = [14, 's']
attacking_player_pos = 0
hand1 = [[14, 'c'], [7, 'c'],[11, 'd']]
hand2 = [[12, 'c'],[12, 'd'], [10, 's']]
#discard_pile = [[[13, 'c'], [13, 'd']], [[6, 'c'], [12, 'd']], [[8, 'c'], [11, 'c']], [[11, 'd'], [14, 'd']], [[9, 's'], [10, 's'], [9, 'c'], [9, 'd']], [[8, 'h'], [8, 'd']], [[7, 'h'], [13, 'h']], [[10, 'c'], [6, 'd'], [6, 's'], [11, 's'], [6, 'h'], [14, 'h'], [14, 's'], [7, 'd']], [[7, 's'], [13, 's']], [[8, 's'], [10, 'd']]]
discard_pile = [[8, 'c'], [11, 'c'], [12, 'h'], [11, 'h'], [14, 'd'], [9, 's'],[12, 's'], [9, 'h'], [10, 'h'], [13, 'c'], [13, 'd'], [6, 'c'], [9, 'c'], [9, 'd'], [8, 'h'], [8, 'd'], [7, 'h'], [13, 'h'], [10, 'c'], [6, 'd'], [6, 's'], [11, 's'], [6, 'h'], [14, 'h'], [14, 's'], [7, 'd'], [7, 's'], [13, 's'], [8, 's'], [10, 'd']]
discard_pile_1hot = convert_str_to_1hot(discard_pile)
#print(discard_pile_1hot,'discard_pile_1hot')

situation_dict = {
    'deck':deck_pos,
    'hand1':hand1,
    'hand2':hand2,
    'trump_card':trump_card_pos,
    'attacking_player':attacking_player_pos,
    'discard_pile':discard_pile,
    'discard_pile_1hot':discard_pile_1hot
}


# In[ ]:


#Training parameters
num_epochs = 1
iterations = 2000
training_cycles = 1
model_checkpoint = 500
tree_path = '/Users/Shuza/Code/Durak/Tree/durak_tree'
#Test durak game env
model_list = [model_attack,model_defend]
function_list = [random_model,random_model]
durak = Durak(deck,model_list,function_list,threshold)
durak.load_tree(tree_path)
previous_winner = (False,0)
#training env
for i in range(iterations):
#     print(previous_winner,'previous_winner')
    durak.init_game(previous_winner)
#     durak.update_game_state()
#     durak.start_from_state(situation_dict)
    durak.play_game()
    previous_winner = durak.game_state.previous_winner
    #count nodes
#     number_first_nodes = durak.game_state.first_root.count_nodes(durak.game_state.first_root,0)
#     number_second_nodes = durak.game_state.second_root.count_nodes(durak.game_state.second_root,0)
    #Propogate EV and get game_states/actions
    print(durak.game_state.first_player,'first player')
    first_outcome = durak.players[durak.game_state.first_player].outcome
    second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
    print(first_outcome,second_outcome,'outcomes of iteration '+str(i))
    print('results')
    print(durak.results[0],durak.results[1])
    played_1_attacks,played_1_attack_evs,played_1_defends,played_1_defend_evs = durak.game_state.first_node.fast_propagate(durak.game_state.first_node,first_outcome)
    played_2_attacks,played_2_attack_evs,played_2_defends,played_2_defend_evs = durak.game_state.second_node.fast_propagate(durak.game_state.second_node,second_outcome)
    #separate game_states from actions. Both players start game_state first. Then alternates
#     first_actions_temp = first_actions[:-1]
#     second_actions_temp = second_actions[:-1]
    #last EV is 0 because for the last node, it has no children and thus has no outcome? WHY? FIX THIS
#     first_evs_temp = first_evs[1:]
#     second_evs_temp = second_evs[1:]
#     print(len(first_actions_temp),'len actions')
#     print(len(second_actions_temp),'len actions')
    print(played_1_attack_evs,'played_1_attack_evs')
    print(played_1_defend_evs,'played_1_defend_evs')
    print(played_2_attack_evs,'played_2_attack_evs')
    print(played_2_defend_evs,'played_2_defend_evs')
#     print(len(first_evs_temp))
# #     print(len(second_evs_temp))
#     player_1_actions = first_actions[0::2]
#     player_2_actions = second_actions[0::2]
# #     print(player_1_actions,'player_1_actions')
# #     print(player_2_actions,'player_2_actions')
#     player_1_game_states = first_actions[1::2]
#     player_2_game_states = second_actions[1::2]
#     print(player_1_game_states,'player_1_game_states')
#     print(player_2_game_states,'player_2_game_states')
    #grab odd and even value locations for game_state and action respectively
    played_1_attack_actions = played_1_attacks[0::2]
    played_1_attack_game_states = played_1_attacks[1::2]
    played_1_defend_actions = played_1_defends[0::2]
    played_1_defend_game_states = played_1_defends[1::2]
    played_2_attack_actions = played_2_attacks[0::2]
    played_2_attack_game_states = played_2_attacks[1::2]
    played_2_defend_actions = played_2_defends[0::2]
    played_2_defend_game_states = played_2_defends[1::2]
    #stack attacks and defend for training
    attacks = np.hstack((played_1_attack_actions,played_2_attack_actions))
    attack_gamestates = np.hstack((played_1_attack_game_states,played_2_attack_game_states))
    attack_evs = np.hstack((played_1_attack_evs,played_2_attack_evs))
    defends = np.hstack((played_1_defend_actions,played_2_defend_actions))
    defend_gamestates = np.hstack((played_1_defend_game_states,played_2_defend_game_states))
    defend_evs = np.hstack((played_1_defend_evs,played_2_defend_evs))
    print(attacks,'attacks')
    print(defends,'defends')
    value_attacks = np.where(attack_evs>-1)[0]
    value_defends = np.where(defend_evs>-1)[0]
    print(value_attacks,'value_attacks')
    print(value_defends,'value_defends')
#     game_states_value_1 = []
#     actions_value_1 = []
#     [actions_value_1.append(value) if value % 2 == 0 else game_states_value_1.append(value) for value in value_locations_1]
#     game_states_value_2 = []
#     actions_value_2 = []
#     [actions_value_2.append(value) if value % 2 == 0 else game_states_value_2.append(value) for value in value_locations_2]
#     game_states_value_1 = value_locations_1[1::2]
#     game_states_value_2 = value_locations_2[1::2]
#     actions_value_1 = value_locations_1[0::2]
#     actions_value_2 = value_locations_2[0::2]
    size_attacks = value_attacks.size
    size_defends = value_defends.size
#     print(value_locations_1,'value_locations_1')
#     print(value_locations_2,'value_locations_2')
#     print(game_states_value_1,'game_states_value_1')
#     print(game_states_value_2,'game_states_value_2')
#     print(actions_value_1,'actions_value_1')
#     print(actions_value_2,'actions_value_2')
    ### train on everything ###
    print('value attacks')
    a = attack_evs.shape[0]
    print(a,'a')
    attack_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in attack_gamestates]
    #[print(state,'game_state') for state in attack_states_list]
    attack_states = np.vstack(attack_states_list)
    player_1_hot = np.zeros(53)
    player_1_hot[int(attacks[0])] = 1
    if len(attacks) > 1:
        for action in attacks[1:]:
            temp = np.zeros(53)
            temp[int(action)] = 1
            player_1_hot = np.vstack((player_1_hot,temp))
    else:
        player_1_hot = player_1_hot.reshape(a,53)
    model_attack.fit(attack_states,[attack_evs.reshape(a,1),player_1_hot],verbose=1)
    #train defending
    print('defending')
    a = defend_evs.shape[0]
    defend_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in defend_gamestates]
    defend_states = np.vstack(defend_states_list) 
    #[print(state,'game_state') for state in attack_states_list]
    player_2_hot = np.zeros(53)
    player_2_hot[int(defends[0])] = 1
    if len(defends) > 1:
        for action in defends[1:]:
            temp = np.zeros(53)
            temp[int(action)] = 1
            player_2_hot = np.vstack((player_2_hot,temp))
    else:
        player_2_hot = player_2_hot.reshape(a,53)
    model_defend.fit(defend_states,[defend_evs.reshape(a,1),player_2_hot],verbose=1)
        
    ### value training ###
#     if size_attacks:
#         print('value attacks')
#     #         print(player_1_actions,'player_1_actions')
#     #         print(actions_value_1,'actions_value_1')
#     #         if len(game_states_value_1) > len(actions_value_1):
#     #             game_states_value_1 = game_states_value_1[1:]
#         attack_str_states = attack_gamestates[value_attacks]
#         attack_evs_train = attack_evs[value_attacks]
#         print(attack_evs_train,'attack_evs_train')
#         a = attack_evs_train.shape[0]
#         print(a,'a')
#         attack_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in attack_str_states]
#         #[print(state,'game_state') for state in attack_states_list]
#         attack_states = np.vstack(attack_states_list)
#         player_1_hot = np.zeros(53)
#         player_1_hot[int(attacks[value_attacks[0]])] = 1
#         if len(value_attacks) > 1:
#             for action in attacks[value_attacks[1:]]:
#                 temp = np.zeros(53)
#                 temp[int(action)] = 1
#                 player_1_hot = np.vstack((player_1_hot,temp))
#         else:
#             player_1_hot = player_1_hot.reshape(a,53)
#     #         print(player_1_evs_train,'player_1_evs_train')
#     #         print('a',a,'player_1_states',player_1_states.shape,'player_1_evs_train',player_1_evs_train.reshape(a,1).shape,'player_1_hot',player_1_hot.shape,'input shapes 1')
#         model_attack.fit(attack_states,[attack_evs_train.reshape(a,1),player_1_hot],verbose=1)
#     #train defending
#     if size_defends:
#         print('value defends')
#     #         print(player_2_actions,'player_2_actions')
#     #         print(actions_value_2,'actions_value_2')
#     #         if len(game_states_value_2) > len(actions_value_2):
#     #             game_states_value_2 = game_states_value_2[1:]
#         defend_str_states = defend_gamestates[value_defends]
#         defend_evs_train = defend_evs[value_defends]
#         print(defend_evs_train,'defend_evs_train')
#         a = defend_evs_train.shape[0]
#         defend_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in defend_str_states]
#         defend_states = np.vstack(defend_states_list) 
#         #[print(state,'game_state') for state in attack_states_list]
#         player_2_hot = np.zeros(53)
#         player_2_hot[int(defends[value_defends[0]])] = 1
#         if len(value_defends) > 1:
#             for action in defends[value_defends[1:]]:
#                 temp = np.zeros(53)
#                 temp[int(action)] = 1
#                 player_2_hot = np.vstack((player_2_hot,temp))
#         else:
#             player_2_hot = player_2_hot.reshape(a,53)
#         #player_1_states = player_1_states[value_locations_1]
#     #         print(player_2_evs_train,'player_2_evs_train')
#     #         print('a',a,'player_2_states',player_2_states.shape,'player_2_evs_train',player_2_evs_train.reshape(a,1).shape,'player_2_hot',player_2_hot.shape,'input shapes 2')
#         model_defend.fit(defend_states,[defend_evs_train.reshape(a,1),player_2_hot],verbose=1)
        #save models for round robins and detailing progress
    if i % model_checkpoint == 0:
        attack_path = attack_model_path + str(i)
        defend_path = defend_model_path + str(i)
        model_attack.save(attack_path)
        model_defend.save(defend_path)
        #Save tree
        durak.save_tree(tree_path)
#Save tree
durak.save_tree(tree_path)
#print results
print('results')
print(durak.results[0],durak.results[1])
        


# In[ ]:


#Round Robin with split models (attacking and defending)
#Load models
model_attacking_dir = '/Users/Shuza/Code/Durak/attack_models'
model_defending_dir = '/Users/Shuza/Code/Durak/defend_models'
models = []
model_names = []
for file in os.listdir(model_dir):
    if not file.startswith('.'):
        print(file,'Model')
        path = model_dir + '/' + file
        print(path,'path')
        model = load_model(path)
        models.append(model)
        model_names.append(file)
#model input dimensions for hand generation
model_input = (1,30,14,1)
intermediate_input = (1,14,1)
attribute_input = (1,1)
on_1d = True
tourney_iterations = 50
#Compute matchups
num_players = len(models)
matchups = [(((models[i],models[j]),(models[j],models[i]))) for i in range(num_players-1) for j in range(i+1,num_players)]
score_table = np.zeros((num_players,num_players))
inserts = [[(i,j),(j,i)] for i in range(num_players-1) for j in range(i+1,num_players)]
results = []
#print(matchups,'matchups')
tic = time.time()
for matchup in matchups:
    match_results = []
    for match in matchup:
        model_list = [match[0],match[1]]
        durak = Durak(deck,model_list,function_list,tournament=True)
        previous_winner = (False,0)
        #function_list = [random_model,random_model]
        for i in range(tourney_iterations):
            durak.init_game(previous_winner)
            durak.play_game()
        #Get end result
        match_results.append((durak.results[0],durak.results[1]))
    results.append(match_results)
for matchup in range(len(results)):
    player_1 = 0
    player_2 = 0
    #print(results[matchup],'results')
    player_1 += results[matchup][0][0]
    player_1 += results[matchup][1][1]
    player_2 += results[matchup][0][1]
    player_2 += results[matchup][1][0]
    #print(player_1,'player_1')
    #print(player_2,'player_2')
    i = inserts[matchup][0][0]
    j = inserts[matchup][0][1]
    score_table[j][i] = player_2
    i = inserts[matchup][1][1]
    j = inserts[matchup][1][0]
    score_table[i][j] = player_1
#print(score_table)
for player in range(num_players):
    print(model_names[player],score_table[player],'Total',sum(score_table[player]))
toc = time.time()
print("Tournament took ",str((toc-tic)/60),'Minutes')


# In[ ]:


#Round Robin
#Load models
model_dir = '/Users/Shuza/Code/Durak/Main_models'
models = []
model_names = []
for file in os.listdir(model_dir):
    if not file.startswith('.'):
        print(file,'Model')
        path = model_dir + '/' + file
        print(path,'path')
        model = load_model(path)
        models.append(model)
        model_names.append(file)
#model input dimensions for hand generation
model_input = (1,30,14,1)
intermediate_input = (1,14,1)
attribute_input = (1,1)
on_1d = True
tourney_iterations = 50
#Compute matchups
num_players = len(models)
matchups = [(((models[i],models[j]),(models[j],models[i]))) for i in range(num_players-1) for j in range(i+1,num_players)]
score_table = np.zeros((num_players,num_players))
inserts = [[(i,j),(j,i)] for i in range(num_players-1) for j in range(i+1,num_players)]
results = []
#print(matchups,'matchups')
tic = time.time()
for matchup in matchups:
    match_results = []
    for match in matchup:
        model_list = [match[0],match[1]]
        durak = Durak(deck,model_list,function_list,tournament=True)
        previous_winner = (False,0)
        #function_list = [random_model,random_model]
        for i in range(tourney_iterations):
            durak.init_game(previous_winner)
            durak.play_game()
        #Get end result
        match_results.append((durak.results[0],durak.results[1]))
    results.append(match_results)
for matchup in range(len(results)):
    player_1 = 0
    player_2 = 0
    #print(results[matchup],'results')
    player_1 += results[matchup][0][0]
    player_1 += results[matchup][1][1]
    player_2 += results[matchup][0][1]
    player_2 += results[matchup][1][0]
    #print(player_1,'player_1')
    #print(player_2,'player_2')
    i = inserts[matchup][0][0]
    j = inserts[matchup][0][1]
    score_table[j][i] = player_2
    i = inserts[matchup][1][1]
    j = inserts[matchup][1][0]
    score_table[i][j] = player_1
#print(score_table)
for player in range(num_players):
    print(model_names[player],score_table[player],'Total',sum(score_table[player]))
toc = time.time()
print("Tournament took ",str((toc-tic)/60),'Minutes')


# In[ ]:


#models 1 vs models 2
#Each first model plays every second model and vice versa
#Load models
#model_dir = os.path.join(os.path.dirname(sys.argv[0]), "Model_Tournament_1d/")
first_models = []
first_model_names = []
second_models = []
second_model_names = []
print('load first models')
for file in os.listdir(first_models_dir):
    if not file.startswith('.'):
        print(file,'Model')
        path = first_models_dir + '/' + file
        print(path,'path')
        model = load_model(path)
        first_models.append(model)
        first_model_names.append(file)
print('load second models')
for file in os.listdir(second_models_dir):
    if not file.startswith('.'):
        print(file,'Model')
        path = second_models_dir + '/' + file
        print(path,'path')
        model = load_model(path)
        second_models.append(model)
        second_model_names.append(file)
#model input dimensions for hand generation
model_input = (1,30,14,1)
intermediate_input = (1,14,1)
attribute_input = (1,1)
on_1d = True
tourney_iterations = 50
#Compute matchups
num_first_players = len(first_model_names)
num_second_players = len(second_model_names)
name_matchups = [(first_model_names[i],second_model_names[j]) for i in range(num_first_players) for j in range(num_second_players)]
matchups = [(first_models[i],second_models[j]) for i in range(num_first_players) for j in range(num_second_players)]
inserts = [(i,j) for i in range(num_first_players) for j in range(num_second_players)]
score_table = np.zeros((num_first_players,num_second_players))
results = []
print(name_matchups,'name_matchups')
tic = time.time()
idx = 0
for matchup in matchups:
    print(name_matchups[idx],'match')
    model_list = [matchup[0],matchup[1]]
    durak = Durak(deck,model_list,function_list,tournament=True)
    previous_winner = (False,0)
    #function_list = [random_model,random_model]
    for i in range(tourney_iterations):
        durak.init_game(previous_winner)
        durak.play_game()
    #Get end result
    match_results = (durak.results[0],durak.results[1])
    results.append(match_results)
    idx += 1
for matchup in range(len(results)):
    player_1 = results[matchup][0]
    player_2 = results[matchup][1]
    #print(player_1,'player_1')
    #print(player_2,'player_2')
    i = inserts[matchup][0]
    j = inserts[matchup][1]
    score_table[i][j] = player_1
print(score_table)
for player in range(len(first_model_names)):
    print(first_model_names[player],score_table[player],'Total',sum(score_table[player]))
for player in range(len(second_model_names)):
    print(second_model_names[player],score_table[:,player],'Total',-sum(score_table[:,player]))
toc = time.time()
print("Tournament took ",str((toc-tic)/60),'Minutes')


# In[ ]:


#Check model layer outputs
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

i = 0
for layer in model.layers:
    print(layer.name)
    if layer.name == 'dense_379' or layer.name == 'dense_380' or layer.name == 'dense_381':
        config = layer.get_config()
        weights = layer.get_weights() # list of numpy arrays
        print(layer.name)
        print(config,'config')
        print(weights,'weights')
    i+=1


# In[ ]:


#Check tree
#Attacker should start with 1 game_state because the initial game_state is always the same
#The Defender should start with N game_states, where N is number of Attacker choices
print(durak.game_state.first_root.parent)
print(durak.game_state.second_root.parent)
print(durak.game_state.first_root.ev)
print(durak.game_state.second_root.ev)
print(len(list(durak.game_state.first_root.child_dict.keys())))
print(len(list(durak.game_state.second_root.child_dict.keys())))
root_child_sec = list(durak.game_state.second_root.child_dict.keys())[0]
readable_child_sec = return_from_hex(root_child_sec)
print(readable_child_sec,'readable_child_sec')
root_child_sec2 = list(durak.game_state.second_root.child_dict.keys())[1]
readable_child_sec2 = return_from_hex(root_child_sec2)
print(readable_child_sec2,'readable_child_sec2')
root_child1 = list(durak.game_state.first_root.child_dict.keys())[0]
readable_child1 = return_from_hex(root_child1)
print(readable_child1,'readable_child1')
print(len(list(durak.game_state.first_root.child_dict[root_child1].child_dict.keys())),'num actions')
root_action1 = list(durak.game_state.first_root.child_dict[root_child1].child_dict.keys())[0]
print(root_action1,'root_action1')
print(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action1].ev,'root_action1.ev')
#readable_action1 = return_from_hex(root_action1)
#print(readable_action1,'readable_action1')
root_action2 = list(durak.game_state.first_root.child_dict[root_child1].child_dict.keys())[1]
#readable_action2 = return_from_hex(root_action2)
#print(readable_action2,'readable_action2')
print(root_action2,'root_action2')
print(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].ev,'root_action2.ev')
#game_state after poor choice
root_gamestate1 = list(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict.keys())[0]
root_action_sub1 = list(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict[root_gamestate1].child_dict.keys())[0]
#pass action
print(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict[root_gamestate1].ev,'root_gamestate1.ev')
print(len(list(durak.game_state.first_root.child_dict[root_child1].child_dict[root_action2].child_dict[root_gamestate1].child_dict.keys())),'game_state children')
print(root_action_sub1,'root_action_sub1')

second_root_child = list(durak.game_state.first_root.child_dict[root_child].child_dict.keys())[0]
third_root_child = durak.game_state.first_root.child_dict[root_child].child_dict[second_root_child]
#print(list(third_root_child.child_dict.keys()),'third_root_child')
print(durak.game_state.first_root.child_dict[root_child].child_dict.keys(),'child keys')
print(durak.game_state.first_root.child_dict[root_child].ev,'child ev')
print(len(durak.game_state.first_root.child_dict.keys()),'len root keys')
print(len(durak.game_state.second_root.child_dict.keys()),'len root keys')
print(durak.trunk.child_dict[0].ev)
# print(durak.game_state.first_root.child_dict.keys())
# print(durak.game_state.second_root.child_dict.keys())
# number_first_nodes = durak.game_state.first_root.count_nodes(durak.game_state.first_root,0)
# number_second_nodes = durak.game_state.second_root.count_nodes(durak.game_state.second_root,0)

# print(durak.game_state.first_node.parent)
# print(durak.game_state.second_node.parent)
# print(durak.game_state.first_node.ev,'first_node ev')
# print(durak.game_state.second_node.ev,'second node ev')
# print(durak.game_state.first_node.child_dict.keys())
# print(durak.game_state.second_node.child_dict.keys())

# print(durak.game_state.first_node.parent.parent.parent.ev)
# print(durak.game_state.second_node.parent.parent.parent.ev)


# In[ ]:


table = np.array([[ 1.,  1., -1.],
 [-1., -1., -1.],
 [-1.,  1.,  1.]])
print(table[:,1])


# In[ ]:


first_model_names = ['billy','bob','bastard','bilbo']
second_model_names = ['clarissa','clair','christie','calendar']
num_players = len(second_model_names)
matchups = [(first_model_names[i],second_model_names[j]) for i in range(num_players) for j in range(num_players)]
#matchups = [(((models[i],models[j]),(models[j],models[i]))) for i in range(num_players-1) for j in range(i+1,num_players)]

inserts = [(i,j) for i in range(num_first_players) for j in range(num_second_players)]
score_table = np.zeros((num_players,num_players))
print(matchups,'matchups')
print(inserts)
print(inserts[1][0])


# In[ ]:


poss = np.arange(53)
prob = np.array([[0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.14919799, 0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.15426552, 0.,         0.,         0.,         0.,         0.17081946,
  0.17467335, 0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.,         0.,         0.,         0.,         0.,         0.,
  0.19267564, 0.,         0.15836803, 0.,         0.        ]]).flatten()

print(poss.shape,prob.shape)
chosen = np.argmax(prob)
print(chosen)
cho = np.random.choice(poss,p=prob)

