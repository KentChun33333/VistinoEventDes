###########################################################################
# USAGE                                                                   #
###########################################################################
# SHELL  : python --conf conf_hub/confID                                  #
# OUTPUT : model in model_hub                                             #
###########################################################################
__author__ = 'Kent (Jin-Chun Chiu)'

from common_tool_agent import n_feat, p_feat, train_svm
from common_tool_agent.conf import Conf
import argparse 

def get_args():
    '''Goal : Parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' use -conf to train model detectors''')
    # Add arguments
    parser.add_argument('-conf', "--conf_path", help="Path to conf_hub",
            required=True)
    # Parses
    args = parser.parse_args()
    # Assign args to variables
    conf_path = args.conf_path
    # Return all variable values
    return conf_path

def main():
    # 1. load the configuration file
    conf = Conf(get_args())
    # 2. extract distracting features with region-based hard mining
    n_feat.extract(conf)
    # 3. extract P features with rotation
    p_feat.extract(conf)
    # 4. train_svm
    train_svm.training(conf)

    # training models

if __name__=='__main__':
	main()

