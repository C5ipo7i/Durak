from train import train_start,train_endgame
from tournament import round_robin_split,round_robin
from durak_models import VEmbed_full,VEmbed_V1,VEmbed_ab3,VEmbed_ab1,VEmbed
import tensorflow as tf
import sys
import os
sys.setrecursionlimit(10000)

#train and then tournament and repeat
#also for hyperparameter training
#also for training vs previous model iterations
#To do threshold update, update threshold value and recall train_start?

#training params
model_names = [VEmbed_full]
#attacking_model_dir = os.path.join(os.path.dirname(sys.argv[0]),'attack_models')
#defending_model_dir = os.path.join(os.path.dirname(sys.argv[0]),'defend_models')
#attacking_model_path = os.path.join(os.path.dirname(sys.argv[0]),'attack_model20000')
#defending_model_path = os.path.join(os.path.dirname(sys.argv[0]),'defend_model20000')
attacking_model_path = '/home/shuza/Code/Durak/attack_models/attack_model50k'
defending_model_path = '/home/shuza/Code/Durak/defend_models/defend_model50k'
# model_path = '/home/shuza/Code/Durak/durak_models/single_model1'
model_path = '/Users/Shuza/Code/Durak/durak_models/single_model0'
model_paths = [model_path,model_path]
tree_endgame_path = os.path.join(os.path.dirname(sys.argv[0]),'Tree/durak_tree_endgame')
tree_path = os.path.join(os.path.dirname(sys.argv[0]),'Tree/durak_tree')
#instantiate dictionary
initialization_params = {
    'single_model':True,
    'train_on_batch':True,
    'learning_cycles':1,
    'iterations':5,
    'model_checkpoint':5000,
    'threshold':50,
    'model_names':model_names,
    'load_tree':False,
    'epochs':5,
    'verbosity':0,
    'model_paths':model_paths,
    'initialize_models':False,
    'multigpu':True,
    'save_tree':False,
    'tree_path':tree_path,
    'tree_endgame_path':tree_endgame_path,
    'print':200,
    'rl':True,
    'discount_factor':0.95
}

# train_endgame(initialization_params)
train_start(initialization_params)
#round_robin_split()
#round_robin()
