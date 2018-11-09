import numpy as np
import pickle
import json
import binascii
import copy
from random import randint
from random import choice

deck = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d']]

def player(possibility_vec,features,hand_vec,players,player_id,game_state,durak):
    #need to display card values
    locations = np.where(possibility_vec==1)[0]
    #print(players[player_id].hand,'hand')
    #print 2d representations of locations
    #print(locations[-1],'locations[-1]')
    durak.print_game_state()
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

def model_decision(possibility_vec,features,hand_vec,players,player_id,game_state,durak):
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
    if durak.tournament == False:
        if game_state == 1:
            # print('defending')
            model_ev,model_action = durak.models[1].predict(model_emb_input)
        else:
            #call attack model for states 0 and 2
            # print('attacking')
            model_ev,model_action = durak.models[0].predict(model_emb_input)
    else:
        #tournament, two groups of 2 attacking and defending models
        if game_state == 1:
            model_ev,model_action = durak.models[player_id][1].predict(model_emb_input)
        else:
            model_ev,model_action = durak.models[player_id][0].predict(model_emb_input)

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
    # print(player_id,'player_id')
    # print(features.played_card,'played_card')
    # print(features.played_cards,'played_cards')
    # print(players[0].hand,'player1 hand')
    # print(players[1].hand,'player2 hand')
#         print(features.discard_pile_1hot,'1hot discard')
#         print(features.trump_card,'trump')
        #print(features.play_deck)
    model_action = np.add(model_action,durak.epsilon)
    masked_choices = np.multiply(model_action.reshape(53,),possibility_vec)
    probability_vector = np.divide(masked_choices,np.sum(masked_choices)).flatten()
#     print(possibility_vec,'possibility_vec')
    # print(model_ev,'model_ev')
    # print(probability_vector,'probability_vector')
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
    #print(decision,'decision')
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
        #print(player_id,'first triggered')
        features.first_node = features.first_node.get_child(key)
        features.first_node.update_visit()
        features.first_node.kind = 1
        features.first_node.game_state = game_state
        features.first_node.key = key
        
        features.first_node = features.first_node.get_child(subkey)
        features.first_node.update_visit()
        features.first_node.kind = 0
    else:
        #print(player_id,'second_node triggered')
        features.second_node = features.second_node.get_child(key)
        features.second_node.update_visit()
        features.second_node.kind = 1
        features.second_node.game_state = game_state
        features.second_node.key = key
        
        features.second_node = features.second_node.get_child(subkey)
        features.second_node.update_visit()
        features.second_node.kind = 0
    return decision


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
        # print(discard_loc,'discard_loc')              
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
    # print(hand_loc,'hand_loc')
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
    # print(played_card_emb,'played_card_emb',played_cards_emb.shape,'played_cards_emb',discard_emb.shape,'discard_emb',hand_emb.shape,'hand_emb',trump_card_emb.shape,'trump_card_emb',possibilities_emb.shape,'possibilities_emb')
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