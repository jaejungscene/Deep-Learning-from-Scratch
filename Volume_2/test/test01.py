import numpy as np
import sys
sys.path.append('/Users/jaejungscene/Projects/AI/Deep_Learning_from_Scratch/Volume_2')
import pickle

class tmp:
   def __init__(self, name):
      self.name = name

# tmp01 = tmp('jaejung')

# with open('pickle_test.pkl', 'wb') as f:
#    pickle.dump(tmp01, f)

with open('pickle_test.pkl', 'rb') as f:
   tmp_load = pickle.load(f)

print(tmp_load.name)
