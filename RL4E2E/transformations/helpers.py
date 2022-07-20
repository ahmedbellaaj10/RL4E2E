import nltk
from nltk.corpus import wordnet
from RLTest4Chatbot.transformation.constants import WORD_TAG_DICT


def char_repeat(sentence, char_index):
    char = sentence[char_index]
    sentence = sentence[:char_index] + char + sentence[char_index:]
    return sentence


def char_insert(sentence, char_index, char):
    sentence = sentence[:char_index] + char + sentence[char_index:]
    return sentence


def get_char(number, min=ord('a'), max=ord('z')):
    return round((max-min)*number + min)


def char_drop(sentence, char_index):
    sentence = sentence[:char_index] + sentence[char_index+1:]
    return sentence


def char_replace(sentence, char_index, char):
    sentence = sentence[:char_index] + char + sentence[char_index+1:]
    return sentence


def char_swap(sentence, char_index):
    """
    Swap between chars at char_index and char_index+1
    If char_index is the last element swap between chars at char_index-1 and at char_index
    """
    length_ = len(sentence)
    if char_index == length_-1:  # the last element in the sentence
        sentence = sentence[:char_index-1] + \
            sentence[char_index] + sentence[char_index-1]
    else:
        sentence = sentence[:char_index] + sentence[char_index +
                                                    1] + sentence[char_index] + sentence[char_index+2:]
    return sentence


def word_insert(sentence, word_index, word):
    words = nltk.word_tokenize(sentence)
    words[word_index] = words[word_index] + " " + word
    return " ".join(words)


def word_piece_insert(sentence, word_index, model, tokenizer):
    """
    Insert a word given by bert pre-trained model
    PS : It's not the same model used in Dial test (uses masked whole word)
    """
    input_ids = tokenizer(
        sentence, return_tensors="pt").input_ids.tolist()[0][1:-1]
    input_ids.insert(word_index, 103)  # 103 is the [MASK] token
    sentence = tokenizer.decode(input_ids)
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    mask_token_indexs = [i for i in range(len(input_ids)) if(
        input_ids[i] == tokenizer.mask_token_id)]
    predicted_token_id = [logits[0, mask_token_index].argmax(
        axis=-1).item() for mask_token_index in mask_token_indexs]
    for i in range(len(mask_token_indexs)):
        index = mask_token_indexs[i]
        input_ids[index] = predicted_token_id[i]
    senetnce = tokenizer.decode(input_ids)
    return senetnce


def word_drop(sentence, word_index):
    words = nltk.word_tokenize(sentence)
    words[word_index] = ""
    return " ".join(words)


def word_replace(sentence, word_index, word):
    words = nltk.word_tokenize(sentence)
    words[word_index] = word
    return " ".join(words)


def word_swap(sentence, word_index):
    words = nltk.word_tokenize(sentence)
    if word_index == len(words) - 1:  # last word
        swap_index = word_index-1
    else:
        swap_index = word_index+1

    swap = words[swap_index]
    word = words[word_index]

    words[word_index] = swap
    words[swap_index] = word
    return " ".join(words)


def get_general_tag(tag: str):
    """
    Given a specific word tag, return the genral
    """
    tags = ["verb", "adjectif", "adverb", "noun"]
    for t in tags:
        if tag in WORD_TAG_DICT[t]:
            return t
    return None


def get_synonyms(word: str, word_tag: str):
    """
    Given a word and a tag, returns a list of the word synonyms having the same genral tag
    """
    word_tag = get_general_tag(word_tag)
    if word_tag == None:
        return [word]

    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    if word in synonyms:
        synonyms.remove(word)
    if len(synonyms) == 0:
        return [word]
    syn_list_ = [synonym.replace("-", " ").replace("_", " ")
                 for synonym in list(synonyms)]
    syn_tags_list = nltk.pos_tag(syn_list_)
    syn_list = []
    for (syn, tag) in syn_tags_list:
        if tag in WORD_TAG_DICT[word_tag]:
            syn_list.append(syn)
    if len(syn_list) == 0:
        return [word]
    return syn_list


def construct_dict_file(file: str):
    """
    Given a file, that contains words and corresponding misspelled,
    returns a dictionnary {"word" : [misspelled words]}
    """
    punkt = '!#$%&()*+,-./:;<=>?@[\]^_{|}~'
    file = open(file, 'r')
    lines = file.readlines()
    misspelled = dict()
    for line in lines:
        line = line.lower()
        line = line.strip()
        line = line.translate(str.maketrans('', '', punkt))
        words = line.split(" ")
        key = words[0]
        value = words[1:]
        if key not in list(misspelled.keys()):
            misspelled[key] = value
        else:
            misspelled[key].extend(value)
    return misspelled


def get_active_params(params, rate, vector_size):
    hidden_trans = len(params) // vector_size
    active_trans = round(rate * (hidden_trans))
    active_params = params[:vector_size*active_trans-1]
    return active_params


def calculate_modif_rate(sentence, sentence_): # word swap is not fully explored but it is okey
    words = nltk.word_tokenize(sentence)
    words_ =  nltk.word_tokenize(sentence_)
    if len(words_)==0 or len(words) == 0 :
        return 1
    shared = set(words).intersection(words_)
    n_shared = len(shared)
    union = set(words).union(words_)
    n_union = len(union)
    # res1 = [w for w in words if w in shared]
    # res2 = [w for w in words_ if w in shared]

    # for i in range(len(shared)-1):
    #     if res1[i]!=res2[i]:
    #         n_shared = n_shared-1
    
    # return 1 - n_shared/len(words)
    # return 1- n_shared/max(len(words), len(words_))
    return n_shared/n_union




