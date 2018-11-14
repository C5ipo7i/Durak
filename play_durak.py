from durak import Durak
from durak_rl import Durak as DRK
from tree_class import Tree,Node
from train import load_models
from durak_utils import player,player_rl,model_decision,model_decision_rl_print,deck
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
import sys
import os

def play_vs_model_rl(model_path):
    #play vs a specific model
    #model_first = load_model('/Users/Shuza/Code/Durak/First_models/first_model1000')
    model = load_model(model_path)
    #model_first.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
    learning_rate=0.002
    opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
    model.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
    model_list = [model,model]
    function_list = [player_rl,model_decision_rl_print]
    threshold = 0
    durak = DRK(deck,model_list,function_list,threshold,play=True,tournament=True)
    previous_winner = (False,0)
    durak.init_game(previous_winner)
    durak.play_game()

def play_vs_model(model_path):
    #play vs a specific model
    #model_first = load_model('/Users/Shuza/Code/Durak/First_models/first_model1000')
    model_attack = 'hi'
    model_defend = load_model(model_path)
    #model_first.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
    learning_rate=0.002
    opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
    model_defend.compile(optimizer=opt,loss=['mse','categorical_crossentropy'])
    model_list = [model_attack,model_defend]
    function_list = [player,model_decision]
    durak = Durak(deck,model_list,function_list,threshold,play=True,tournament=True)
    previous_winner = (False,0)
    durak.init_game(previous_winner)
    durak.play_game()

def play_vs_splitmodel(attack_path,defend_path):
    #play vs a specific model
    #model_first = load_model('/Users/Shuza/Code/Durak/First_models/first_model1000')
    model_attack,model_defend = load_models([attack_path,defend_path])
    threshold = 50
    model_list = [model_attack,model_defend]
    function_list = [player,model_decision]
    durak = Durak(deck,model_list,function_list,threshold,play=True,tournament=False)
    previous_winner = (False,0)
    durak.init_game(previous_winner)
    durak.play_game()

if __name__ == '__main__':
    # attack_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'attack_models')
    # defend_models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'defend_models')
    # attack_model_path = os.path.join(attack_models_dir,'attack_model0')
    # defend_model_path = os.path.join(defend_models_dir,'defend_model0')
    # models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'tournament_models'
    # model_path = os.path.join(models_dir,'rl_model')
    models_dir = os.path.join(os.path.dirname(sys.argv[0]), 'durak_models')
    model_path = os.path.join(models_dir,'single_model0')
    play_vs_model_rl(model_path)
