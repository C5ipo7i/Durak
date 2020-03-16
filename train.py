from durak import Durak
from durak_rl import Durak as DRL
from durak_multi import Durak as DM
from tree_class import Tree,Node
from durak_utils import model_decision,model_decision_rl,model_decision_rl_print,model_decision_multi,return_emb_vector,deck
from durak_models import VEmbed,VEmbed_full
from itertools import chain

from multiprocessing import Pool
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
    models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'durak_models')
    attack_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'attack_models')
    defend_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'defend_models')
    model_path = os.path.join(models_dir,'single_model')
    attack_model_path = os.path.join(attack_models_dir,'attack_model')
    defend_model_path = os.path.join(defend_models_dir,'defend_model')
    #trunk = openpickle(tree_path)
    previous_winner = (False,0)
    if initialization_params['single_model'] == False:
        #load split models for attack and defense
        if initialization_params['initialize_models'] == True:
            model_names = initialization_params['model_names']
            model_attack,model_defend = instantiate_models(model_names,initialization_params['multigpu'])
            model_list = [model_attack,model_attack]
        else:
            model_attack,model_defend = load_models(initialization_params['model_paths'],initialization_params['multigpu'])
            model_list = [model_attack,model_attack]
    else:
        if initialization_params['initialize_models'] == True:
            model_names = initialization_params['model_names']
            model,model_2 = instantiate_models(model_names,initialization_params['multigpu'])
            model_list = [model,model]
        else:
            model,model_2 = load_models(initialization_params['model_paths'],initialization_params['multigpu'])
            model_list = [model,model]

    training_dict = {
        'iterations':iterations,
        'epochs':initialization_params['epochs'],
        'model':model,
        'model_path':model_path,
        # 'model_attack':model_attack,
        # 'model_defend':model_defend,
        'tree_path':tree_path,
        'start':'beginning',
        'situation_dict':None,
        'model_checkpoint':model_checkpoint,
        'attack_model_path':attack_model_path,
        'defend_model_path':defend_model_path,
        'previous_winner':previous_winner,
        'verbosity':initialization_params['verbosity'],
        'save_tree':initialization_params['save_tree'],
        'print':initialization_params['print'],
        'model_list':model_list,
        'learning_cycles':initialization_params['learning_cycles'],
        'discount_factor':initialization_params['discount_factor']
    }
    if initialization_params['rl'] == True:
        function_list = [model_decision_rl,model_decision_rl]
        #Create the game env
        durak = DRL(deck,model_list,function_list,threshold)
        if initialization_params['train_on_batch'] == True:
            train_on_batch_rl(durak,training_dict)
        else:
            train_rl(durak,training_dict)
    else:
        function_list = [model_decision,model_decision]
        #Create the game env
        durak = Durak(deck,model_list,function_list,threshold)
        if initialization_params['load_tree'] == True:
            if os.path.getsize(tree_path) > 0:
                durak.load_tree(tree_path)
        if initialization_params['train_on_batch'] == True:
            train_on_batch(durak,training_dict)
        else:
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
    training_cycles = 1
    iterations = initialization_params['iterations']
    model_checkpoint = initialization_params['model_checkpoint']
    #Threshold of randomness. 0 = completely random. 100 = entirely according to the model output
    threshold = initialization_params['threshold']
    #model dirs
    models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'durak_models')
    attack_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'attack_models')
    defend_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'defend_models')
    model_path = os.path.join(models_dir,'single_model')
    attack_model_path = os.path.join(attack_models_dir,'attack_model')
    defend_model_path = os.path.join(defend_models_dir,'defend_model')
    #load models
    if initialization_params['single_model'] == False:
        #load split models for attack and defense
        if initialization_params['initialize_models'] == True:
            model_names = initialization_params['model_names']
            model_attack,model_defend = instantiate_models(model_names,initialization_params['multigpu'])
            model_list = [model_attack,model_attack]
        else:
            model_attack,model_defend = load_models(initialization_params['model_paths'],initialization_params['multigpu'])
            model_list = [model_attack,model_attack]
    else:
        if initialization_params['initialize_models'] == True:
            model_names = initialization_params['model_names']
            model,model_2 = instantiate_models(model_names,initialization_params['multigpu'])
            model_list = [model,model]
        else:
            model,model_2 = load_models(initialization_params['model_paths'],initialization_params['multigpu'])
            model_list = [model,model]
    previous_winner = (False,0)
    training_dict = {
        'iterations':iterations,
        'epochs':initialization_params['epochs'],
        'model':model,
        'model_path':model_path,
        # 'model_attack':model_attack,
        # 'model_defend':model_defend,
        'tree_path':endgame_tree_path,
        'start':'endgame',
        'situation_dict':situation_dict,
        'model_checkpoint':model_checkpoint,
        'attack_model_path':attack_model_path,
        'defend_model_path':defend_model_path,
        'previous_winner':previous_winner,
        'verbosity':initialization_params['verbosity'],
        'save_tree':initialization_params['save_tree'],
        'print':initialization_params['print'],
        'model_list':model_list,
        'learning_cycles':initialization_params['learning_cycles'],
        'discount_factor':initialization_params['discount_factor']
    }
    if initialization_params['rl'] == True:
        function_list = [model_decision_rl_print,model_decision_rl_print]
        durak = DRL(deck,model_list,function_list,threshold)
        if initialization_params['train_on_batch'] == True:
            train_on_batch_rl(durak,training_dict)
        else:
            train_rl(durak,training_dict)
    else:
        function_list = [model_decision,model_decision]
        durak = Durak(deck,model_list,function_list,threshold)
        if initialization_params['load_tree'] == True:
            durak.load_tree(tree_path)
        if initialization_params['train_on_batch'] == True:
            train_on_batch(durak,training_dict)
        else:
            train(durak,training_dict)

def train_rl(durak,training_dict):
    print('TRAINING RL')
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
        first_outcome = durak.players[durak.game_state.first_player].outcome
        second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
        played_1_attacks,played_1_attack_evs,played_1_defends,played_1_defend_evs = durak.game_state.first_node.fast_propagate(durak.game_state.first_node,first_outcome)
        played_2_attacks,played_2_attack_evs,played_2_defends,played_2_defend_evs = durak.game_state.second_node.fast_propagate(durak.game_state.second_node,second_outcome)
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
        a = attack_evs.shape[0]
        input_attack_gamestates,input_attack_evs,player_1_hot,input_defend_gamestates,input_defend_evs,player_2_hot = return_everything_train(attacks,attack_evs,attack_gamestates,defends,defend_evs,defend_gamestates)
        model_attack.fit(input_attack_gamestates,[input_attack_evs,player_1_hot],verbose=1)
        model_defend.fit(input_defend_gamestates,[input_defend_evs,player_2_hot],verbose=1)
        if i % training_dict['model_checkpoint'] == 0 and i != 0:
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
        first_outcome = durak.players[durak.game_state.first_player].outcome
        second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
        played_1_attacks,played_1_attack_evs,played_1_defends,played_1_defend_evs = durak.game_state.first_node.fast_propagate(durak.game_state.first_node,first_outcome)
        played_2_attacks,played_2_attack_evs,played_2_defends,played_2_defend_evs = durak.game_state.second_node.fast_propagate(durak.game_state.second_node,second_outcome)
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
        a = attack_evs.shape[0]
        input_attack_gamestates,input_attack_evs,player_1_hot,input_defend_gamestates,input_defend_evs,player_2_hot = return_everything_train(attacks,attack_evs,attack_gamestates,defends,defend_evs,defend_gamestates)
        model_attack.fit(input_attack_gamestates,[input_attack_evs,player_1_hot],verbose=1)
        model_defend.fit(input_defend_gamestates,[input_defend_evs,player_2_hot],verbose=1)
        if i % training_dict['model_checkpoint'] == 0 and i != 0:
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

def train_on_batch_rl(durak,training_dict):
    print('TRAINING ON BATCH RL')
    tic = time.time()
    model_path = training_dict['model_path']
    model = training_dict['model']
    tree_path = training_dict['tree_path']
    start = training_dict['start']
    previous_winner = training_dict['previous_winner']
    discount_factor = training_dict['discount_factor']
    #training env
    for j in range(training_dict['learning_cycles']):
        for i in range(training_dict['iterations']):
            if start == 'endgame':
                durak.start_from_state(training_dict['situation_dict'])
            else:
                durak.init_game(previous_winner)
            durak.play_game()
            first_outcome = durak.players[durak.game_state.first_player].outcome
            second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
            player_1_hist = durak.game_state.player_1_history
            played_1_actions = player_1_hist[1::2]
            played_1_game_states = player_1_hist[0::2]
            played_1_actions = np.hstack((played_1_actions))
            played_1_game_states = np.hstack((played_1_game_states))
            player_2_hist = durak.game_state.player_2_history
            played_2_actions = player_2_hist[1::2]
            played_2_game_states = player_2_hist[0::2]
            played_2_actions = np.hstack((played_2_actions))
            played_2_game_states = np.hstack((played_2_game_states))
            if first_outcome > 0:
                #we train on positive outcomes or draws only
                # actions = played_1_actions
                # game_states = played_1_game_states
                player_1_length = played_1_game_states.shape[0]
                player_2_length = played_2_game_states.shape[0]
                total_length = player_1_length + player_2_length
                blank_actions = np.zeros((player_2_length,53))
                discounted_rewards_pos = np.array([1*discount_factor**i for i in range(0,player_1_length)])
                discounted_rewards_neg = np.array([-1*discount_factor**i for i in range(0,player_2_length)])
                discounted_rewards = np.hstack((discounted_rewards_pos,discounted_rewards_neg)).reshape(total_length,1)
                game_states = np.hstack((played_1_game_states,played_2_game_states))
                actions_train,gamestates_train,discounted_rewards_train = return_everything_train_rl(played_1_actions,game_states,discounted_rewards)
                actions_train = np.vstack((actions_train,blank_actions))
                #train on negative outcomes with zero'd out actions and EVs
            elif second_outcome > 0:
                player_1_length = played_1_game_states.shape[0]
                player_2_length = played_2_game_states.shape[0]
                total_length = player_1_length + player_2_length
                blank_actions = np.zeros((player_1_length,53))
                discounted_rewards_pos = np.array([1*discount_factor**i for i in range(0,player_2_length)])
                discounted_rewards_neg = np.array([-1*discount_factor**i for i in range(0,player_1_length)])
                discounted_rewards = np.hstack((discounted_rewards_pos,discounted_rewards_neg)).reshape(total_length,1)
                game_states = np.hstack((played_2_game_states,played_1_game_states))
                actions_train,gamestates_train,discounted_rewards_train = return_everything_train_rl(played_2_actions,game_states,discounted_rewards)
                actions_train = np.vstack((actions_train,blank_actions))
            else:
            #draw train on both
                player_2_length = played_2_game_states.shape[0]
                player_1_length = played_1_game_states.shape[0]
                total_length = player_1_length + player_2_length
                discounted_rewards_2 = np.array([1*discount_factor**i for i in range(0,player_2_length)])
                discounted_rewards_1 = np.array([1*discount_factor**i for i in range(0,player_1_length)])
                discounted_rewards = np.hstack((discounted_rewards_1,discounted_rewards_2)).reshape(total_length,1)
                actions = np.vstack((played_1_actions.reshape(player_1_length,1),played_2_actions.reshape(player_2_length,1)))
                game_states = np.hstack((played_1_game_states,played_2_game_states))
                actions_train,gamestates_train,discounted_rewards_train = return_everything_train_rl(actions,game_states,discounted_rewards)
        
            #game_states = list(chain(*game_states_temp))
            #get model inputs
            if i != 0:
                historical_gamestates = np.vstack((historical_gamestates,gamestates_train))
                historical_discounted_rewards = np.vstack((historical_discounted_rewards,discounted_rewards_train))
                historical_actions = np.vstack((historical_actions,actions_train))
            else:
                historical_actions = actions_train
                historical_discounted_rewards = discounted_rewards_train
                historical_gamestates = gamestates_train
            print(i)
        print('MODEL CHECKPOINT ',j)
        model.fit(historical_gamestates,[historical_discounted_rewards,historical_actions],epochs=training_dict['epochs'],verbose=1)
        recent_model_path = model_path + str(j)
        model.save(recent_model_path)
    #print results
    print('results')
    print(durak.results[0],durak.results[1])
    toc = time.time()
    print("Training on batch took ",str((toc-tic)/60),'Minutes')

def trigger(inputs):
    print('in trigger')
    dictionary = inputs[0]
    model_list = dictionary['model_list']
    threshold = dictionary['threshold']
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    default_graph = tf.get_default_graph()
    default_graph.finalize()
    function_list = [model_decision_multi,model_decision_multi]
    durak = DM(deck,model_list,default_graph,function_list,threshold)
    iterations = dictionary['iterations']
    start = dictionary['start']
    if start == 'endgame':
        init = training_dict['situation_dict']
        prep = durak.start_from_state
    else:
        init = previous_winner
        prep = durak.init_game
    for i in range(iterations):
        prep(init)
        durak.play_game()
        first_outcome = durak.players[durak.game_state.first_player].outcome
        second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
        played_1_attacks,played_1_attack_evs,played_1_defends,played_1_defend_evs = durak.game_state.first_node.fast_propagate(durak.game_state.first_node,first_outcome)
        played_2_attacks,played_2_attack_evs,played_2_defends,played_2_defend_evs = durak.game_state.second_node.fast_propagate(durak.game_state.second_node,second_outcome)
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
        value_attacks = np.where(attack_evs>-1)[0]
        value_defends = np.where(defend_evs>-1)[0]
        size_attacks = value_attacks.size
        size_defends = value_defends.size    
        #get model inputs
        input_attack_gamestates,input_attack_evs,player_1_hot,input_defend_gamestates,input_defend_evs,player_2_hot = return_everything_train(attacks,attack_evs,attack_gamestates,defends,defend_evs,defend_gamestates)
        if i != 0:
            train_attack_gamestates = np.vstack((train_attack_gamestates,input_attack_gamestates))
            train_attack_evs = np.vstack((train_attack_evs,input_attack_evs))
            train_attack_policy = np.vstack((train_attack_policy,player_1_hot))
            train_defend_gamestates = np.vstack((train_defend_gamestates,input_defend_gamestates))
            train_defend_evs = np.vstack((train_defend_evs,input_defend_evs))
            train_defend_policy = np.vstack((train_defend_policy,player_2_hot))
        else:
            train_attack_gamestates = input_attack_gamestates
            train_attack_evs = input_attack_evs
            train_attack_policy = player_1_hot
            train_defend_gamestates = input_defend_gamestates
            train_defend_evs = input_defend_evs
            train_defend_policy = player_2_hot
        print(i,'ith iteration')
    return train_attack_gamestates,train_attack_evs,train_attack_policy,train_defend_gamestates,train_defend_evs,train_defend_policy

def train_on_batch(durak,training_dict):
    print('TRAINING ON BATCH')
    tic = time.time()
    model_path = training_dict['model_path']
    model = training_dict['model']
    for model in training_dict['model_list']:
        model._make_predict_function
    tree_path = training_dict['tree_path']
    start = training_dict['start']
    previous_winner = training_dict['previous_winner']
    pool = Pool(processes=2)
    #training env
    for j in range(training_dict['learning_cycles']):
        inputs = [[training_dict],[training_dict]]
        results = pool.map(trigger,inputs)
        print(results,'results')
        print('MODEL CHECKPOINT ',j)
        model.fit(train_attack_gamestates,[train_attack_evs,train_attack_policy],epochs=training_dict['epochs'],verbose=1)
        model.fit(train_defend_gamestates,[train_defend_evs,train_defend_policy],epochs=training_dict['epochs'],verbose=1)
        recent_model_path = model_path + str(j)
        model.save(recent_model_path)
        #Save tree
        if training_dict['save_tree'] == True:
            durak.save_tree(tree_path)
    #print results
    print('results')
    print(durak.results[0],durak.results[1])
    toc = time.time()
    print("Training on batch took ",str((toc-tic)/60),'Minutes')


def train_on_batch_save(durak,training_dict):
    print('TRAINING ON BATCH')
    tic = time.time()
    model_path = training_dict['model_path']
    model = training_dict['model']
    tree_path = training_dict['tree_path']
    start = training_dict['start']
    previous_winner = training_dict['previous_winner']
    if start == 'endgame':
        inputs = training_dict['situation_dict']
        prep = durak.start_from_state
    else:
        inputs = previous_winner
        prep = durak.init_game
    #training env
    for j in range(training_dict['learning_cycles']):
        for i in range(training_dict['iterations']):
            prep(inputs)
            durak.play_game()
            first_outcome = durak.players[durak.game_state.first_player].outcome
            second_outcome = durak.players[(durak.game_state.first_player + 1)%2].outcome
            played_1_attacks,played_1_attack_evs,played_1_defends,played_1_defend_evs = durak.game_state.first_node.fast_propagate(durak.game_state.first_node,first_outcome)
            played_2_attacks,played_2_attack_evs,played_2_defends,played_2_defend_evs = durak.game_state.second_node.fast_propagate(durak.game_state.second_node,second_outcome)
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
            value_attacks = np.where(attack_evs>-1)[0]
            value_defends = np.where(defend_evs>-1)[0]
            size_attacks = value_attacks.size
            size_defends = value_defends.size    
            #get model inputs
            input_attack_gamestates,input_attack_evs,player_1_hot,input_defend_gamestates,input_defend_evs,player_2_hot = return_everything_train(attacks,attack_evs,attack_gamestates,defends,defend_evs,defend_gamestates)
            if i != 0:
                train_attack_gamestates = np.vstack((train_attack_gamestates,input_attack_gamestates))
                train_attack_evs = np.vstack((train_attack_evs,input_attack_evs))
                train_attack_policy = np.vstack((train_attack_policy,player_1_hot))
                train_defend_gamestates = np.vstack((train_defend_gamestates,input_defend_gamestates))
                train_defend_evs = np.vstack((train_defend_evs,input_defend_evs))
                train_defend_policy = np.vstack((train_defend_policy,player_2_hot))
            else:
                train_attack_gamestates = input_attack_gamestates
                train_attack_evs = input_attack_evs
                train_attack_policy = player_1_hot
                train_defend_gamestates = input_defend_gamestates
                train_defend_evs = input_defend_evs
                train_defend_policy = player_2_hot
            print(i)
        print('MODEL CHECKPOINT ',j)
        model.fit(train_attack_gamestates,[train_attack_evs,train_attack_policy],epochs=training_dict['epochs'],verbose=1)
        model.fit(train_defend_gamestates,[train_defend_evs,train_defend_policy],epochs=training_dict['epochs'],verbose=1)
        recent_model_path = model_path + str(j)
        model.save(recent_model_path)
        #Save tree
        if training_dict['save_tree'] == True:
            durak.save_tree(tree_path)
    #print results
    print('results')
    print(durak.results[0],durak.results[1])
    toc = time.time()
    print("Training on batch took ",str((toc-tic)/60),'Minutes')

def return_everything_train_rl(actions,gamestates,discounted_rewards):
    a = discounted_rewards.shape[0]
    states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in gamestates]
    states = np.vstack(states_list)
    player_actions = np.zeros(53)
    player_actions[int(actions[0])] = 1
    if len(actions) > 1:
        for act in actions[1:]:
            temp = np.zeros(53)
            temp[int(act)] = 1
            player_actions = np.vstack((player_actions,temp))
    else:
        player_actions = player_actions.reshape(a,53)
    return player_actions,states,discounted_rewards.reshape(a,1)

def return_everything_train(attacks,attack_evs,attack_gamestates,defends,defend_evs,defend_gamestates):
    a = attack_evs.shape[0]
    attack_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in attack_gamestates]
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
    #train defending
    b = defend_evs.shape[0]
    defend_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in defend_gamestates]
    defend_states = np.vstack(defend_states_list) 
    player_2_hot = np.zeros(53)
    player_2_hot[int(defends[0])] = 1
    if len(defends) > 1:
        for action in defends[1:]:
            temp = np.zeros(53)
            temp[int(action)] = 1
            player_2_hot = np.vstack((player_2_hot,temp))
    else:
        player_2_hot = player_2_hot.reshape(a,53)
    return attack_states,attack_evs.reshape(a,1),player_1_hot,defend_states,defend_evs.reshape(b,1),player_2_hot  

def return_value_train(attacks,attack_evs,attack_gamestates,defends,defend_evs,defend_gamestates):
    ### value training ###
    value_attacks = np.where(attack_evs>-1)[0]
    value_defends = np.where(defend_evs>-1)[0]
    size_attacks = value_attacks.size
    size_defends = value_defends.size
    if size_attacks:
        attack_str_states = attack_gamestates[value_attacks]
        attack_evs_train = attack_evs[value_attacks]
        a = attack_evs_train.shape[0]
        attack_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in attack_str_states]
        attack_states = np.vstack(attack_states_list)
        player_1_hot = np.zeros(53)
        player_1_hot[int(attacks[value_attacks[0]])] = 1
        if len(value_attacks) > 1:
            for action in attacks[value_attacks[1:]]:
                temp = np.zeros(53)
                temp[int(action)] = 1
                player_1_hot = np.vstack((player_1_hot,temp))
        else:
            player_1_hot = player_1_hot.reshape(a,53)
        attack_input_vector = attack_states,[attack_evs.reshape(a,1),player_1_hot]
    else:
        attack_input_vector = np.zeros()
    #train defending
    if size_defends:
        defend_str_states = defend_gamestates[value_defends]
        defend_evs_train = defend_evs[value_defends]
        a = defend_evs_train.shape[0]
        defend_states_list = [pickle.loads(binascii.unhexlify(state.encode('ascii'))) for state in defend_str_states]
        defend_states = np.vstack(defend_states_list) 
        player_2_hot = np.zeros(53)
        player_2_hot[int(defends[value_defends[0]])] = 1
        if len(value_defends) > 1:
            for action in defends[value_defends[1:]]:
                temp = np.zeros(53)
                temp[int(action)] = 1
                player_2_hot = np.vstack((player_2_hot,temp))
        else:
            player_2_hot = player_2_hot.reshape(a,53)
        defend_input_vector = defend_states,[defend_evs.reshape(a,1),player_2_hot]
    else:
        defend_input_vector = np.zeros()
    return attack_input_vector,defend_input_vector     


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
    pass
