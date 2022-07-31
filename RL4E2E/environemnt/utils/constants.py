import numpy as np


# STATE_ELEMENTS = 2
# TURN_POS = 1

# DIALOG_POS = 0

TRANSFORMATIONS = ["charinsert", "chardrop", "charreplace", "wordinsert", "worddrop", "worddrop"]


# VALID_RATE = 0.25 # I don't really know if right file/ right rate (because we will be having a valid rate per action)


# """
# NOT HERE, PLEASE CHANGE THE PLACE
# """
# MAX_N_TURNS = 22. # 18 : test_dials, 22 : train dials, 17 dev dials 
# MAX_N_DIALOGUES = 8425. #999 : test_dials, 999 : dev dials, 8424 : train dials
# MAX_WORDS = 40 # max words in a turn in the data set 


# #Observation space lower and upper bound 
# OBSERVATION_LOWER = np.array([0., 0.])
# OBSERVATION_UPPER = np.array([MAX_N_DIALOGUES, MAX_N_TURNS])
