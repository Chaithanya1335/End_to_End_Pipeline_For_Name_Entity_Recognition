import os
import sys
import pickle

def save_object(path,object):
    with open(path,'wb') as file:
        pickle.dump(object,file)