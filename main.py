from train import train_start,train_endgame
from tournament import round_robin_split
from durak_models import VEmbed_full
import tensorflow as tf
import sys
sys.setrecursionlimit(10000)

#train and then tournament and repeat
#also for hyperparameter training
#also for training vs previous model iterations
#To do threshold update, update threshold value and recall train_start?

#training params
iterations = 20001
model_checkpoint = 5000
threshold = 50
model_names = [VEmbed_full]
load_tree = True
verbosity = 0
initialize_models = False
attacking_model_dir = os.path.join(os.path.dirname(sys.argv[0]),'attack_models')
defending_model_dir = os.path.join(os.path.dirname(sys.argv[0]),'defend_models')
attacking_model_path = os.path.join(os.path.dirname(sys.argv[0]),'attack_model20000')
defending_model_path = os.path.join(os.path.dirname(sys.argv[0]),'defend_model20000')
model_paths = [attacking_model_path,defending_model_path]
multigpu = True
#instantiate dictionary
initialization_params = {
    'iterations':iterations,
    'model_checkpoint':model_checkpoint,
    'threshold':threshold,
    'model_names':model_names,
    'load_tree':load_tree,
    'verbosity':verbosity,
    'model_paths':model_paths,
    'initialize_models':initialize_models,
    'multigpu':multigpu,
    'print':200
}

train_start(initialization_params)
round_robin_split()
