from durak import Durak
from tree_class import Tree,Node
from durak_utils import model_decision,return_emb_vector,deck
from durak_models import VEmbed,VEmbed_full

import time
import tensorflow as tf
import numpy as np
import copy
import os
import sys
import pickle
import json
import binascii
import copy
from random import randint
from random import choice
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
"""
some stuff
"""

def train_start(initialization_params):
    #Training parameters
    num_epochs = 1
    training_cycles = 1
    iterations = initialization_params['iterations']
    model_checkpoint = initialization_params['model_checkpoint']
    #Threshold of randomness. 0 = completely random. 100 = entirely according to the model output
    threshold = initialization_params['threshold']
    #model dirs
    tree_path = initialization_params['tree_path']
    attack_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'attack_models')
    defend_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'defend_models')
    attack_model_path = os.path.join(attack_models_dir,'attack_model')
    defend_model_path = os.path.join(defend_models_dir,'defend_model')
    #trunk = openpickle(tree_path)
    previous_winner = (False,0)
    if initialization_params['initialize_models'] == True:
        model_names = initialization_params['model_names']
        model_attack,model_defend = instantiate_models(model_names,initialization_params['multigpu'])
        model_list = [model_attack,model_defend]
    else:
        model_attack,model_defend = load_models(initialization_params['model_paths'],initialization_params['multigpu'])
        model_list = [model_attack,model_defend]
    function_list = [model_decision,model_decision]
    #Create the game env
    durak = Durak(deck,model_list,function_list,threshold)
    if initialization_params['load_tree'] == True:
        if os.path.getsize(target) > 0:
            durak.load_tree(tree_path)

    training_dict = {
        'iterations':iterations,
        'model_attack':model_attack,
        'model_defend':model_defend,
        'tree_path':tree_path,
        'start':'beginning',
        'situation_dict':None,
        'model_checkpoint':model_checkpoint,
        'attack_model_path':attack_model_path,
        'defend_model_path':defend_model_path,
        'previous_winner':previous_winner,
        'verbosity':initialization_params['verbosity'],
        'print':initialization_params['print']
    }
    train(durak,training_dict)

def train_endgame(initialization_params):
    from durak_utils import convert_str_to_1hot
    #Testing a particular situation
    endgame_tree_path = initialization_params['tree_endgame_path']
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
    #Training parameters
    num_epochs = 1
    iterations = 250
    training_cycles = 1
    model_checkpoint = 500
    #Threshold of randomness. 0 = completely random. 100 = entirely according to the model output
    threshold = 50
    #Test durak game env
    attack_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'attack_models')
    defend_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'defend_models')
    attack_model_path = os.path.join(attack_models_dir,'attack_model')
    defend_model_path = os.path.join(defend_models_dir,'defend_model')
    #load models
    model_names = [VEmbed]
    model_attack,model_defend = instantiate_models(model_names)
    model_list = [model_attack,model_defend]
    function_list = [model_decision,model_decision]
    durak = Durak(deck,model_list,function_list,threshold)
    if load_tree == True:
        durak.load_tree(tree_path)
    previous_winner = (False,0)
    training_dict = {
        'iterations':iterations,
        'model_attack':model_attack,
        'model_defend':model_defend,
        'tree_path':endgame_tree_path,
        'start':'endgame',
        'situation_dict':situation_dict,
        'model_checkpoint':model_checkpoint,
        'attack_model_path':attack_model_path,
        'defend_model_path':defend_model_path,
        'previous_winner':previous_winner,
        'verbosity':1,
        'print':initialization_params['print']
    }
    train(durak,training_dict)

def train(durak,training_dict):
    print('TRAINING')
    tic = time.time()
    attack_model_path = training_dict['attack_model_path']
    defend_model_path =training_dict['defend_model_path']
    model_attack = training_dict['model_attack']
    model_defend = training_dict['model_defend']
    tree_path = training_dict['tree_path']
    start = training_dict['start']
    previous_winner = training_dict['previous_winner']
    #training env
    for i in range(training_dict['iterations']):
    #     print(previous_winner,'previous_winner')
    #     durak.update_game_state()
        if start == 'endgame':
            durak.start_from_state(training_dict['situation_dict'])
        else:
            durak.init_game(previous_winner)
        durak.play_game()
        #previous_winner = durak.game_state.previous_winner
        #count nodes
    #     number_first_nodes = durak.game_state.first_root.count_nodes(durak.game_state.first_root,0)
    #     number_second_nodes = durak.game_state.second_root.count_nodes(durak.game_state.second_root,0)
        #Propogate EV and get game_states/actions
        # print(durak.game_state.first_player,'first player')
        first_outcome = durak.players[durak.game_state.first_player].outcome
        second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
        # print(first_outcome,second_outcome,'outcomes of iteration '+str(i))
        # print('results')
        # print(durak.results[0],durak.results[1])
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
        # print(played_1_attack_evs,'played_1_attack_evs')
        # print(played_1_defend_evs,'played_1_defend_evs')
        # print(played_2_attack_evs,'played_2_attack_evs')
        # print(played_2_defend_evs,'played_2_defend_evs')
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
        # print(attacks,'attacks')
        # print(defends,'defends')
        value_attacks = np.where(attack_evs>-1)[0]
        value_defends = np.where(defend_evs>-1)[0]
        # print(value_attacks,'value_attacks')
        # print(value_defends,'value_defends')
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
        # print('value attacks')
        a = attack_evs.shape[0]
        # print(a,'a')
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
        model_attack.fit(attack_states,[attack_evs.reshape(a,1),player_1_hot],verbose=training_dict['verbosity'])
        #train defending
        # print('defending')
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
        model_defend.fit(defend_states,[defend_evs.reshape(a,1),player_2_hot],verbose=training_dict['verbosity'])
            
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
        print(i)
        if i % training_dict['model_checkpoint'] == 0:
            print('MODEL CHECKPOINT')
            attack_path = attack_model_path + str(i)
            defend_path = defend_model_path + str(i)
            model_attack.save(attack_path)
            model_defend.save(defend_path)
            #Save tree
            if training_dict['save_tree'] == True:
                durak.save_tree(tree_path)
    #Save tree
    if training_dict['save_tree'] == True:
        durak.save_tree(tree_path)
    #print results
    print('results')
    print(durak.results[0],durak.results[1])
    toc = time.time()
    print("Training took ",str((toc-tic)/60),'Minutes')

def load_models(model_paths,multigpu):
    #model input dimensions for hand generation
    model_input = (266,)
    model_emb_input = (144,)
    policy_shape = (53)
    #model hyper parameters. Needs tuning
    alpha = 0.002
    reg_const = 0.0001
    learning_rate=0.002
    opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
    #(input_shape,policy_shape,alpha)
    if len(model_paths) > 1:
        model_attack = load_model(model_paths[0])
        model_defend = load_model(model_paths[1])
    else:
        model_attack = load_model(model_paths[0])
        model_defend = load_model(model_paths[0])
    # model_first = V1abstract1(model_input,policy_shape,alpha,reg_const)
    # model_second = V1abstract1(model_input,policy_shape,alpha,reg_const)
    if multigpu == True:
        with tf.device("/cpu:0"):
            model_attack.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
            model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
        model_attack.compile(optimizer=opt,loss=['logcosh','categorical_crossentropy'])
        model_defend.compile(optimizer=opt,loss=['logcosh','categorical_crossentropy'])
    else:
        model_attack.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
        model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
    model_attack.summary()
    return model_attack,model_defend

def instantiate_models(model_names,multigpu):
    #model input dimensions for hand generation
    model_input = (266,)
    model_emb_input = (144,)
    policy_shape = (53)
    #model hyper parameters. Needs tuning
    alpha = 0.002
    reg_const = 0.0001
    learning_rate=0.002
    opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
    #(input_shape,policy_shape,alpha)
    if len(model_names) > 1:
        model_attack = model_names[0](model_emb_input,policy_shape,alpha,reg_const)
        model_defend = model_names[1](model_emb_input,policy_shape,alpha,reg_const)
    else:
        model_attack = model_names[0](model_emb_input,policy_shape,alpha,reg_const)
        model_defend = model_names[0](model_emb_input,policy_shape,alpha,reg_const)
    # model_first = V1abstract1(model_input,policy_shape,alpha,reg_const)
    # model_second = V1abstract1(model_input,policy_shape,alpha,reg_const)
    if multigpu == True:
        with tf.device("/cpu:0"):
            model_attack.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
            model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
        model_attack.compile(optimizer=opt,loss=['logcosh','categorical_crossentropy'])
        model_defend.compile(optimizer=opt,loss=['logcosh','categorical_crossentropy'])
    else:
        model_attack.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
        model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
    model_attack.summary()
    return model_attack,model_defend

if __name__ == '__main__':
    train_endgame()
