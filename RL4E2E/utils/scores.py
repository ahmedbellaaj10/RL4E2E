import numpy as np
import math
import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu(sen1,sen2):
    s1 , s2 = sen1.split() , sen2.split()
    return sentence_bleu(s1, s2, smoothing_function=SmoothingFunction().method1, weights=(0.25, 0.25, 0.25, 0.25))
    # return sentence_bleu([s1], s2,  weights=(0.25, 0.25, 0.25, 0.25))

