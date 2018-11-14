import sys
import os
from durak import Durak
from tree_class import Tree,Node
from durak_utils import deck,model_decision
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
import time
import numpy as np

def round_robin_split():
    #Round Robin with split models (attacking and defending)
    #Load models
    model_attacking_dir = os.path.join(os.path.dirname(sys.argv[0]), 'attack_models')
    model_defending_dir = os.path.join(os.path.dirname(sys.argv[0]), 'defend_models')
    attacking_models = []
    defending_models = []
    model_names = []
    #create teams
    teams = []
    for file in os.listdir(model_attacking_dir):
        if not file.startswith('.'):
            print(file,'Model')
            path = model_attacking_dir + '/' + file
            print(path,'path')
            attacking_model = load_model(path)
            model_name = file.strip('attack_')
            number = file.strip('attack_model')
            for dfile in os.listdir(model_defending_dir):
                if not dfile.startswith('.'):
                    if dfile.strip('defend_') == model_name:
                        print(dfile,'Model')
                        path = model_defending_dir + '/' + dfile
                        print(path,'path')
                        defending_model = load_model(path)
                        teams.append((attacking_model,defending_model))
                        break
            model_names.append('team_'+str(number))
    #model input dimensions for hand generation
    tourney_iterations = 50
    function_list = [model_decision,model_decision]
    #Compute matchups
    num_players = len(teams)
    matchups = [(((teams[i],teams[j]),(teams[j],teams[i]))) for i in range(num_players-1) for j in range(i+1,num_players)]
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
            #function_list = [model_decision,model_decision]
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

def round_robin():
    #Round Robin
    #Load models
    #model_dir = os.path.join(os.path.dirname(sys.argv[0]), 'durak_models')
    model_dir = os.path.join(os.path.dirname(sys.argv[0]), 'tournament_models')
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
    function_list = [model_decision,model_decision]
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
            previous_winner = (True,0) #So each player goes first
            #function_list = [model_decision,model_decision]
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
    print('{} Tournament games'.format(tourney_iterations*2))
    for player in range(num_players):
        print(model_names[player],score_table[player],'Total',sum(score_table[player]))
    toc = time.time()
    print("Tournament took ",str((toc-tic)/60),'Minutes')

def model_1_vs_model_2():
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
        #function_list = [model_decision,model_decision]
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
