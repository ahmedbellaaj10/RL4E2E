from abc import ABC, abstractmethod
import os
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
import gensim.downloader as api
from RL4E2E.transformations.helpers import char_insert, char_replace, get_char, char_repeat, char_drop, char_swap, word_insert, word_piece_insert, word_drop, word_replace, word_swap, get_synonyms, construct_dict_file, get_active_params
# jaccard_modif_rate, nltk_modif, 
from RL4E2E.transformations.constants import DIACTIRICS, PUNKT, VOWELS, ADJACENT_AZERTY, ADJACENT_QUERTY, EMOTICONS, MISSPELLED_FILE, TOP_N, WORD_TAG_DICT
from RL4E2E.transformations.constants import WORD_DROP_N_TRANS, WORD_INSERT_N_TRANS, WORD_REPLACE_N_TRANS, CHAR_DROP_N_TRANS, CHAR_INSERT_N_TRANS, CHAR_REPLACE_N_TRANS
from RL4E2E.transformations.constants import WORD_INSERT_VECTOR_SIZE, WORD_REPLACE_VECTOR_SIZE, WORD_DROP_VECTOR_SIZE, CHAR_DROP_VECTOR_SIZE, CHAR_INSERT_VECTOR_SIZE, CHAR_REPLACE_VECTOR_SIZE
from RL4E2E.transformations.constants import CHAR_DROP_MAX_TRANS, WORD_DROP_MAX_TRANS, WORD_INSERT_MAX_TRANS, CHAR_INSERT_MAX_TRANS, CHAR_REPLACE_MAX_TRANS, WORD_REPLACE_MAX_TRANS
from RL4E2E.transformations.constants import MAX_WORDS, VALID_RATE
from RL4E2E.utils.constants import PROJECT_PATH ,FRAMEWORK_PATH
import logging
# logging.basicConfig(filename='output.log'  , level=logging.INFO ,format="%(message)s")

class Transformer(ABC):
    def __init__(self):
        self.lower_bound, self.upper_bound, self.vector_size = None, None, None
        self.max_trans = round(MAX_WORDS*VALID_RATE)

    def sample(self):
        pass

    @abstractmethod
    def apply(self, sentence, transformation_vectors):
        try:
            assert type(sentence) == str
        except:
            sentence = str(sentence)
        return sentence

    @abstractmethod
    def get_upper_bound(self):
        pass

    @abstractmethod
    def get_lower_bound(self):
        pass

    @abstractmethod
    def get_vector_size(self):
        pass

    # def get_changed_words(s1,s2):
    #     indices = {}
    #     sen1 , sen2 = s1.split(), s2.split()
    #     for i , s in enumerate(sen1):



class CharInsert(Transformer):
    def __init__(self):
        super().__init__()
        self.punkt = PUNKT
        self.vector_size = CHAR_INSERT_VECTOR_SIZE
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.n_trans = CHAR_INSERT_N_TRANS
        self.valid_trans = CHAR_INSERT_MAX_TRANS

    def sample(self, sentence):
        n_chars = round(len(nltk.word_tokenize(sentence)))
        return np.round(
            np.random.uniform(0, 1, (self.vector_size * n_chars)), 4
        )

    def sample_one(self, sentence):
        valid = False
        length_ = len(sentence)
        while(not valid):
            params = np.round(
                np.random.uniform(0, 1, (self.vector_size)), 4
            )
            char_index = int(round(length_ * params[0]))
            valid = char_index != 0
        return params

    def sample_(self, sentence, n_trans):
        params = []
        for i in range(n_trans):
            params.extend(self.sample_one(sentence))
        return params

    def apply(self, ori_sentence, transformation_vectors):
        trans_rate = 0
        sentence = super().apply(ori_sentence, transformation_vectors)
        length_ = len(sentence)
        n_t = 0
        for i in range(len(transformation_vectors)//self.vector_size):
            char_index = transformation_vectors[i*self.vector_size]
            char_index = int(round(length_ * char_index))
            if char_index != 0:  # Transformation is active
                n_t = n_t+1  # number of active transformations
                
                char_index = char_index-1
                trans_index = transformation_vectors[i*self.vector_size+1]
                trans_index = int(round(trans_index*(self.n_trans - 1)))

                if trans_index == 0:  # char repeat
                    sentence, rate = char_repeat(sentence, char_index)

                if trans_index == 1:  # random char insert
                    char = transformation_vectors[i*self.vector_size+2]
                    char = chr(get_char(char))
                    sentence, rate = char_insert(sentence, char_index, char)

                if trans_index == 2:  # punctuation insert
                    punk_index = transformation_vectors[i*self.vector_size+2]
                    punk_index = int(round(punk_index * (len(self.punkt) - 1)))
                    punkt = self.punkt[punk_index]
                    sentence,rate = char_insert(sentence, char_index, punkt)
                trans_rate+= rate
            if n_t >= self.valid_trans:
                # return sentence, 1 - length_/len(sentence)
                return sentence, trans_rate

        # return sentence, 1 - length_/len(sentence)
        return sentence, trans_rate

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_valid_trans(self):
        return self.valid_trans


class CharDrop(Transformer):
    def __init__(self):
        self.punkt = PUNKT
        self.vowel_list = VOWELS
        self.vector_size = CHAR_DROP_VECTOR_SIZE
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.n_trans = CHAR_DROP_N_TRANS
        self.valid_trans = CHAR_DROP_MAX_TRANS

    def sample(self, sentence):
        n_chars = round(len(nltk.word_tokenize(sentence)) * VALID_RATE)
        return np.round(
            np.random.uniform(0, 1, (self.vector_size * n_chars)), 4
        )

    def sample_one(self, sentence):
        valid = False
        length_ = len(sentence)
        while(not valid):
            params = np.round(
                np.random.uniform(0, 1, (self.vector_size)), 4
            )
            char_index = int(round(length_ * params[0]))
            valid = char_index != 0
        return params

    def sample_(self, sentence, n_trans):
        params = []
        for i in range(n_trans+1):
            params.extend(self.sample_one(sentence))

        return params

    def apply(self, ori_sentence, transformation_vectors):
        sentence = super().apply(ori_sentence, transformation_vectors)
        n_t = 0
        length_ = len(sentence)
        trans_rate = 0
        for i in range(len(transformation_vectors)//self.vector_size):
            trans_index = transformation_vectors[i*self.vector_size+1]
            trans_index = int(round(trans_index*(self.n_trans - 1)))
            char_index = transformation_vectors[i*self.vector_size]
            rate = 0
            if (trans_index == 0):  # random char drop
                length_ = len(sentence)
                char_index = int(round(length_ * char_index))
                if char_index != 0:
                    n_t = n_t + 1
                    char_index = char_index - 1
                    sentence, rate = char_drop(sentence, char_index)

            if (trans_index == 1):  # space drop
                spaces = [i for i, x in enumerate(sentence) if x == " "]
                space_index = int(round(len(spaces) * char_index))
                if space_index != 0:
                    n_t = n_t + 1
                    space_index = space_index-1
                    rate = 1
                    sentence , rate = char_drop(sentence, spaces[space_index])

            if (trans_index == 2):  # vowel drop
                vowels = [i for i, x in enumerate(
                    sentence) if x in self.vowel_list]
                vowel_index = int(round(len(vowels) * char_index))
                if vowel_index != 0:
                    n_t = n_t + 1
                    vowel_index = vowel_index-1
                    sentence, rate = char_drop(sentence, vowels[vowel_index])

            if (trans_index == 3):  # punctuation drop
                punkts = [i for i, x in enumerate(sentence) if x in self.punkt]
                punkt_index = int(round(len(punkts) * char_index))
                if punkt_index != 0:
                    n_t = n_t + 1
                    punkt_index = punkt_index-1
                    sentence, rate = char_drop(sentence, punkts[punkt_index])
            trans_rate += rate
            if n_t >= self.valid_trans:
                # return sentence, 1 - len(sentence)/length_
                return sentence, trans_rate
        # return sentence, 1 - len(sentence)/length_
        return sentence, trans_rate

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_valid_trans(self):
        return self.valid_trans


class CharReplace(Transformer):
    def __init__(self):
        super().__init__()
        self.punkt = PUNKT
        self.azerty = ADJACENT_AZERTY
        self.querty = ADJACENT_QUERTY
        self.diactrics = DIACTIRICS
        self.vector_size = CHAR_REPLACE_VECTOR_SIZE
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.n_trans = CHAR_REPLACE_N_TRANS
        self.valid_trans = CHAR_REPLACE_MAX_TRANS

    def sample(self, sentence):
        n_chars = round(len(nltk.word_tokenize(sentence)) * VALID_RATE)
        return np.round(
            np.random.uniform(0, 1, (self.vector_size * n_chars)), 4
        )

    def sample_one(self, sentence):
        valid = False
        length_ = len(sentence)
        while(not valid):
            params = np.round(
                np.random.uniform(0, 1, (self.vector_size)), 4
            )
            char_index = int(round(length_ * params[0]))
            valid = char_index != 0
        return params

    def sample_(self, sentence, n_trans):
        params = []
        for i in range(n_trans):
            params.extend(self.sample_one(sentence))
        return params

    def apply(self, ori_sentence, transformation_vectors):
        n_t = 0
        length_ = len(ori_sentence)
        trans_rate = 0
        sentence = super().apply(ori_sentence, transformation_vectors)
        for i in range(len(transformation_vectors)//self.vector_size):
            rate = 0
            trans_index = transformation_vectors[i*self.vector_size+1]
            trans_index = int(round(trans_index*(self.n_trans - 1)))
            char_index = transformation_vectors[i*self.vector_size]
            length_ = len(sentence)
            char_index = int(round(length_ * char_index))
            if (trans_index == 0):  # random char replace
                if char_index != 0:
                    n_t = n_t+1
                    char_index = char_index-1
                    char = transformation_vectors[i*self.vector_size+2]
                    char = chr(get_char(char))
                    sentence, rate = char_replace(sentence, char_index, char)

            if (trans_index == 1):  # replace adjacent azerty
                if char_index != 0:
                    n_t = n_t+1
                    char_index = char_index-1
                    char = sentence[char_index]
                    if char in list(self.azerty.keys()):
                        azerty_list = self.azerty[char]
                        char_azerty = transformation_vectors[i *
                                                             self.vector_size+2]
                        char_azerty = int(round(len(azerty_list) * char_azerty))
                        char_azerty = char_azerty - 1
                        char_azerty = azerty_list[char_azerty]
                        sentence, rate = char_replace(
                            sentence, char_index, char_azerty)

            if (trans_index == 2):  # replace adjacent querty
                if char_index != 0:
                    n_t = n_t+1
                    char_index = char_index-1
                    char = sentence[char_index]
                    if char in list(self.querty.keys()):
                        querty_list = self.querty[char]
                        char_querty = transformation_vectors[i *
                                                             self.vector_size+2]
                        char_querty = int(round(len(querty_list) * char_querty))
                        char_querty = char_querty - 1
                        char_querty = querty_list[char_querty]
                        sentence, rate = char_replace(
                            sentence, char_index, char_querty)

            if (trans_index == 3):  # replace with diactric form
                diactric_list = [i for i, x in enumerate(
                    sentence) if x in list(self.diactrics.keys())]
                char_index = int(round(len(diactric_list) * char_index))
                if char_index != 0:
                    n_t = n_t+1
                    char_index = char_index-1
                    char_index = diactric_list[char_index]
                    char = sentence[char_index]  # to_replace
                    diactric_ = self.diactrics[char]
                    replace_with = transformation_vectors[i*self.vector_size+2]
                    replace_with = int(round(len(diactric_) * replace_with))
                    replace_with = replace_with-1
                    replace_with = diactric_[replace_with]
                    sentence, rate = char_replace(sentence, char_index, replace_with)

            if (trans_index == 4):  # swap between two characters
                if char_index != 0:
                    n_t = n_t+1
                    char_index = char_index-1
                    sentence, rate = char_swap(sentence, char_index)
            trans_rate += trans_rate
            if n_t >= self.valid_trans:
                # return sentence,  n_t/length_
                return sentence, trans_rate

        # return sentence, n_t/length_
        return sentence, trans_rate

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_valid_trans(self):
        return self.valid_trans


class WordInsert(Transformer):
    def __init__(self):
        super().__init__()
        self.vector_size = WORD_INSERT_VECTOR_SIZE
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.emoji = EMOTICONS
        self.stop_words = stopwords.words('english')
        try :
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(PROJECT_PATH,'bert_model/bert-base-uncased') , from_tf=True)
            self.model = BertForMaskedLM.from_pretrained(os.path.join(PROJECT_PATH,'bert_model/bert-base-uncased'), from_tf=True)
        self.n_trans = WORD_INSERT_N_TRANS
        self.valid_trans = WORD_INSERT_MAX_TRANS

    def sample(self, sentence):
        # as a max number of words to insert = initial number of words
        n_words = round(len(nltk.word_tokenize(sentence))*VALID_RATE)
        return np.round(
            np.random.uniform(0, 1, (self.vector_size * n_words)), 4
        )

    def sample_one(self, sentence):
        valid = False
        length_ = len(nltk.word_tokenize(sentence))
        while(not valid):
            params = np.round(
                np.random.uniform(0, 1, (self.vector_size)), 4
            )
            word_index = int(round(length_ * params[0]))
            valid = word_index != 0
        return params

    def sample_(self, sentence, n_trans):
        params = []
        for i in range(n_trans):
            params.extend(self.sample_one(sentence))
        return params

    def apply(self, sentence, transformation_vectors):
        words = nltk.word_tokenize(sentence)
        trans_rate = 0
        length_ = len(words)  # n words, original word length
        sentence = super().apply(sentence, transformation_vectors)
        n_t = 0
        for i in range(len(transformation_vectors)//self.vector_size):
            rate = 0
            trans_index = transformation_vectors[i*self.vector_size+1]
            trans_index = int(round(trans_index*(self.n_trans - 1)))
            word_index = transformation_vectors[i*self.vector_size]
            word_index = int(round(length_ * word_index))
            if word_index > 0:
                n_t = n_t+1
                word_index = word_index - 1
                if (trans_index == 0):  # insert a word using a pre-trained model
                    sentence, rate = word_piece_insert(
                        sentence, word_index, self.model, self.tokenizer)
                if (trans_index == 1):  # insert a stop word
                    stop_word_index = transformation_vectors[i *
                                                             self.vector_size+2]
                    stop_word_index = int(round(
                        stop_word_index * len(self.stop_words)) - 1)
                    stop_word = self.stop_words[stop_word_index]
                    sentence, rate = word_insert(sentence, word_index, stop_word)
                # if (trans_index == 2):  # insert an emoji
                #     print("insert an emoji")
                #     emoji_index = transformation_vectors[i*self.vector_size+2]
                #     emoji_index = int(round(emoji_index * len(self.emoji))) - 1
                #     emoji = self.emoji[emoji_index]
                #     print("emoji",emoji)
                #     sentence, rate = word_insert(sentence, emoji_index, emoji)
                if (trans_index == 2):  # repeat a word
                    to_insert = words[word_index]
                    sentence, rate = word_insert(sentence, word_index, to_insert)
            trans_rate += rate
            if n_t >= self.valid_trans:
                words = nltk.word_tokenize(sentence)
                # return sentence, 1-length_/len(words)  # jaccard distance
                return sentence, trans_rate

        words = nltk.word_tokenize(sentence)
        # return sentence, 1-length_/len(words)  # jaccard distance
        return sentence, trans_rate

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_valid_trans(self):
        return self.valid_trans


class WordDrop(Transformer):
    def __init__(self):
        super().__init__()
        self.vector_size = WORD_DROP_VECTOR_SIZE
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound
        self.n_trans = WORD_DROP_N_TRANS
        self.valid_trans = WORD_DROP_MAX_TRANS
        self.stop_words = stopwords.words('english')

    def sample(self, sentence):
        n_words = round(len(nltk.word_tokenize(sentence))*VALID_RATE)
        return np.round(
            np.random.uniform(0, 1, (self.vector_size * n_words)), 4
        )

    def sample_one(self, sentence):
        valid = False
        length_ = len(nltk.word_tokenize(sentence))
        while(not valid):
            params = np.round(
                np.random.uniform(0, 1, (self.vector_size)), 4
            )
            word_index = int(round(length_ * params[0]))
            valid = word_index != 0
        return params

    def sample_(self, sentence, n_trans):
        params = []
        for i in range(n_trans):
            params.extend(self.sample_one(sentence))
        return params

    def apply(self, sentence, transformation_vectors):
        n_t = 0
        words = nltk.word_tokenize(sentence)
        length_ = len(words)
        trans_rate = 0
        sentence = super().apply(sentence, transformation_vectors)
        for i in range((len(transformation_vectors) // self.vector_size)-1):
            rate = 0
            trans_index = transformation_vectors[i*self.vector_size+1]
            trans_index = int(round(trans_index*(self.n_trans - 1)))
            word_index = transformation_vectors[i*self.vector_size]
            words = nltk.word_tokenize(sentence)
            if trans_index == 0:  # drop a random word
                word_index = int(round(word_index * len(words)))
                if word_index != 1:
                    n_t = n_t + 1
                    word_index = word_index-1
                    sentence, rate = word_drop(sentence, word_index)

            elif trans_index == 1:  # drop a stop word
                stop_words = [i for i, x in enumerate(
                    words) if x in self.stop_words]
                word_index = int(round(word_index * len(stop_words)))
                if word_index != 1:
                    n_t = n_t + 1
                    word_index = word_index-1
                    sentence, rate = word_drop(sentence, word_index)

            elif trans_index == 2:  # drop a verb
                verbs = [i for i, x in enumerate(
                    nltk.pos_tag(words)) if x in WORD_TAG_DICT["verb"]]
                word_index = int(round(word_index * len(verbs)))
                if word_index != 1:
                    n_t = n_t + 1
                    word_index = word_index-1
                    sentence, rate = word_drop(sentence, word_index)
            trans_rate += rate
            if n_t >= self.valid_trans:
                words = nltk.word_tokenize(sentence)
                # return sentence, 1-len(sentence)/length_  # jaccard distance
                return sentence, trans_rate

        words = nltk.word_tokenize(sentence)
        # return sentence, 1-len(words)/length_  # jaccard distance
        return sentence, trans_rate

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_valid_trans(self):
        return self.valid_trans


class WordReplace(Transformer):
    def __init__(self):
        super().__init__()
        self.vector_size = WORD_REPLACE_VECTOR_SIZE
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.punkt = PUNKT
        self.stop_words = stopwords.words('english')
        self.misspelled = construct_dict_file(MISSPELLED_FILE)
        self.model = api.load("glove-wiki-gigaword-100")
        self.top_n = TOP_N
        self.n_trans = WORD_REPLACE_N_TRANS
        self.valid_trans = WORD_REPLACE_MAX_TRANS

    def sample(self, sentence):
        n_words = round(len(nltk.word_tokenize(sentence))*VALID_RATE)
        return np.round(
            np.random.uniform(0, 1, (self.vector_size * n_words)), 4
        )

    def sample_one(self, sentence):
        valid = False
        length_ = len(nltk.word_tokenize(sentence))
        while(not valid):
            params = np.round(
                np.random.uniform(0, 1, (self.vector_size)), 4
            )
            word_index = int(round(length_ * params[0]))
            valid = word_index != 0
        return params

    def sample_(self, sentence, n_trans):
        params = []
        for i in range(n_trans):
            params.extend(self.sample_one(sentence))
        return params

    def apply(self, sentence, transformation_vectors):
        n_t = 0
        words = nltk.word_tokenize(sentence)
        length_ = len(words)
        sentence = super().apply(sentence, transformation_vectors)
        tran_rate = 0
        trans_rate = 0
        for i in range(len(transformation_vectors)//self.vector_size):
            rate = 0
            words = nltk.word_tokenize(sentence)
            words_tags = nltk.pos_tag(words)
            length_ = len(words)
            trans_index = transformation_vectors[i*self.vector_size+1]
            trans_index = int(round(trans_index*(self.n_trans - 1)))
            word_index = transformation_vectors[i*self.vector_size]
            word_index = int(round(length_ * word_index))

            if word_index > 0:
                n_t = n_t + 1
                word_index = word_index - 1
                if (trans_index == 0):  # Synonym replace
                    word = words[word_index]
                    word_tag = words_tags[word_index][1]
                    synonyms = get_synonyms(word, word_tag)
                    syn_index = transformation_vectors[i*self.vector_size + 2]
                    syn_index = int(round(syn_index * (len(synonyms)-1)))
                    synonym = synonyms[syn_index]
                    sentence, rate = word_replace(sentence, word_index, synonym)
                    # trans_rate = trans_rate + nltk_modif(word, synonym)/length_

                if (trans_index == 1):  # Similar replace
                    try:  # not all words can be in the vocabulary
                        word = words[word_index]
                        bests = self.model.most_similar(word, topn=self.top_n)
                        similars = [best[0] for best in bests]
                        sim_index = transformation_vectors[i *
                                                           self.vector_size + 2]
                        sim_index = int(round(sim_index * (len(similars)-1)))
                        similar = similars[sim_index]
                    except:
                        similar = words[word_index]
                    # trans_rate = trans_rate + nltk_modif(word, similar)/length_
                    sentence, rate = word_replace(sentence, word_index, similar)

                # replace with misspelled form, we calculate jaccard disatnce between the two words
                if (trans_index == 2):
                    word = words[word_index]
                    if word in list(self.misspelled.keys()):
                        misspelled_list = self.misspelled[word]
                        miss_index = transformation_vectors[i *
                                                            self.vector_size + 2]
                        miss_index = int(round(
                            miss_index * (len(misspelled_list)-1)))
                        miss_word = misspelled_list[miss_index]
                        sentence, rate = word_replace(
                            sentence, word_index, miss_word)

                        # trans_rate = trans_rate + \
                        #     jaccard_modif_rate(word, miss_word)/length_
                if (trans_index == 3):  # swap words, word swap is considered as two transformations
                    sentence, rate = word_swap(sentence, word_index)
                    trans_rate = trans_rate + 2/length_

                tran_rate += rate
                if n_t >= self.valid_trans:
                    # return sentence, trans_rate
                    return sentence, tran_rate

        # return sentence, trans_rate
        return sentence, tran_rate

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_valid_trans(self):
        return self.valid_trans


def build_transformation(transformation_name):
    if transformation_name.lower() == 'charinsert':
        return CharInsert()

    elif transformation_name.lower() == 'chardrop':
        return CharDrop()

    if transformation_name.lower() == 'charreplace':
        return CharReplace()

    elif transformation_name.lower() == 'worddrop':
        return WordDrop()

    if transformation_name.lower() == 'wordinsert':
        return WordInsert()

    elif transformation_name.lower() == 'wordreplace':
        return WordReplace()

    else:
        raise Exception('Not Existed Transformation')


class CompoundTransformer(Transformer):
    def __init__(self, transformations=[]):
        super().__init__()
        self.transfromation_names = transformations
        self.n_hidden_action = round(MAX_WORDS*VALID_RATE)
        self.transformation_objects = []
        self.transformation_offsets = []
        self.build_transformation_data()
        self.vector_size = self.transformation_offsets[-1][1]
        self.upper_bound = self.get_upper_bound()
        self.lower_bound = self.get_lower_bound()
        self.num_actions = len(self.transformation_objects)

    def build_transformation_data(self):
        start_idx = 0
        for transformation_name in self.transfromation_names:
            transformation_obj = build_transformation(transformation_name)
            self.transformation_objects.append(transformation_obj)
            end_idx = start_idx + transformation_obj.get_vector_size()*self.n_hidden_action
            self.transformation_offsets.append((start_idx, end_idx))
            start_idx = end_idx



    def apply(self, sentence, transformation_vectors):
        origin_sentence = sentence
        cum_trans_rate = 0
        # try:
        actions, params = transformation_vectors
        for i in range(self.num_actions):
            action_rate = actions[2*i + 1]
            if action_rate != 0.0:
                transform = self.transformation_objects[i]
                start_params, end_params = self.transformation_offsets[i]
                action_params = params[start_params:end_params]
                active_params = get_active_params(
                    action_params,    VALID_RATE, transform.get_vector_size())
                sentence, rate = transform.apply(sentence, active_params)
                # print("partial rate", rate)
                # print("sentence", sentence)
                # print("rate", rate)
                
                cum_trans_rate = cum_trans_rate + rate
            else:  # action is not activated ~ not from top k
                pass
        # except:
        #     return sentence, cum_trans_rate/len(sentence.split())
        x = max(len(sentence.split()),len(origin_sentence.split()))
        print('new rate', cum_trans_rate/x)
        return sentence, cum_trans_rate/x

    def get_upper_bound(self):
        return np.array([1]*self.get_vector_size())

    def get_lower_bound(self):
        return np.array([0]*self.get_vector_size())

    def get_vector_size(self):
        return self.vector_size

    def get_actions(self):
        ACTIONS = {}
        for i in range(len(self.transformation_objects)):
            ACTIONS[i] = (self.transformation_objects[i],
                          self.transformation_objects[i].get_vector_size())
        return ACTIONS