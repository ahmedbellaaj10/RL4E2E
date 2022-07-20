FRAMEWORK_PATH = "/home/ahmed/RL4E2E/RL4E2E"
PUNKT = '!#$%&()*+,-./:;<=>?@[\]^_{|}~' 

VOWELS = 'aeiuoy'

# adjacent english keyboard (querty) from https://github.com/ybisk/charNMT-noise/blob/master/noise/en.key
ADJACENT_QUERTY = {
    'q': ['1', '2', 'w', 's', 'a'],
    'w': ['q', '2', '3', 'e', 'd', 's', 'a'],
    'e': ['w', '3', '4', 'r', 'f', 'd', 's'],
    'r': ['e', '4', '5', 't', 'g', 'f', 'd'],
    't': ['r', '5', '6', 'y', 'h', 'g', 'f'],
    'y': ['g', 't', '6', '7', 'u', 'j', 'h'],
    'u': ['y', '7', '8', 'i', 'k', 'j', 'h'],
    'i': ['j', 'u', '8', '9', 'o', 'l', 'k'],
    'o': ['i', '9', '0', 'p', ';', 'l', 'k'],
    'p': ['o', '0', '-', '[', "'", ';', 'l'],
    'a': ['q', 'w', 's', 'x', 'z'],
    's': ['a', 'w', 'e', 'd', 'c', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'v', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', '.', ',', 'm'],
    'l': ['k', 'o', 'p', ';', '.', ','],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k', ',']

}

# azerty adjacent keyboard (or french ) from https://github.com/ybisk/charNMT-noise/blob/master/noise/en.key
ADJACENT_AZERTY = {
    'a': ['&', 'é', 'z', 's', 'q'],
    'z': ['a', 'é', '"', 'e', 'd', 's', 'q'],
    'e': ['z', '"', "'", 'r', 'f', 'd', 's'],
    'r': ['e', "'", '(', 't', 'g', 'f', 'd'],
    't': ['r', '(', '§', 'y', 'h', 'g', 'f'],
    'y': ['t', '§', 'è', 'u', 'j', 'h', 'g'],
    'u': ['y', 'è', '!', 'i', 'k', 'j', 'h'],
    'i': ['u', '!', 'ç', 'o', 'l', 'k', 'j'],
    'o': ['i', 'ç', 'à', 'p', 'm', 'l', 'k'],
    'p': ['o', 'à', ')', '^', 'ù', 'm', 'l'],
    'q': ['a', 'z', 's', 'x', 'w'],
    's': ['q', 'z', 'e', 'd', 'c', 'x', 'w'],
    'd': ['s', 'z', 'e', 'r', 'f', 'v', 'c', 'x'],
    'f': ['d', 'e', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', ',', 'n'],
    'k': ['j', 'i', 'o', 'l', ';', ','],
    'l': ['k', 'o', 'p', 'm', ':', ';'],
    'm': ['l', 'p', '^', 'ù', '=', ':'],
    'ù': ['m', '^', '$', '=', 'w', 'q', 's', 'x'],
    'x': ['w', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n']
}


# Lowercase and upper case from https://support.microsoft.com/en-us/office/keyboard-shortcuts-to-add-language-accent-marks-in-word-3801b103-6a8d-42a5-b8ba-fdc3774cfc76
DIACTIRICS = { 
    'a': ['à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ā'],
    'A': ['À', 'Á', 'Â', 'Ã', 'Ä', 'Å', "Æ", 'Ā'],

    'i': ['ì', 'í', 'î', 'ï', 'ī'],
    'I': ['Ì', 'Í', 'Î', 'Ï', 'Ī'],

    'o': ['ò', 'ó', 'ô', 'õ', 'ö', 'œ', 'ō'],
    'O': ['Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Œ', 'Ō'],

    'u': ['ù', 'ú', 'û', 'ü', 'ū'],
    'U': ['Ù', 'Ú', 'Û', 'Ü', 'Ū'],

    'e': ['è', 'é', 'ê', 'ë', 'ē', '€'],
    'E': ['È', 'É', 'Ê', 'Ë', 'Ē', '€'],

    'y': ['ý'],
    'Y': ['Ý'],

    'n': ['ñ'],
    'N': ['Ñ'],

    'y': ['ÿ'],
    'Y': ['Ÿ'],

    'c': ['ç'],
    'C': ['Ç'],

    'd': ['ð'],
    'D': ['Ð'],

    'o': ['ø'],
    'O': ['Ø'],

    'b': ['ß'],
    'B': ['B'],


    '?': ['¿'],
    '!': ['¡'],
}

EMOTICONS = list(map(chr, range(0x1F601, 0x1F64F + 1)))
DINGBATS = list(map(chr, range(0x2702, 0x27B0 + 1)))
TRANSPORT_SYMBOLS = list(map(chr, range(0x1F680, 0x1F6C0 + 1)))
ENCLOSED_CHARACTERS = list(map(chr, range(0x24C2, 0x1F251 + 1)))
ADDITIONAL_EMOTICONS = list(map(chr, range(0x1F600, 0x1F636 + 1)))
ADDITIONAL_TRANSPORT_AND_MAP_SYMBOLS = list(map(chr, range(0x1F681, 0x1F6C5 + 1)))
OTHER_ADDITIONAL_SYMBOLS = list(map(chr, range(0x1F30D, 0x1F567 + 1)))
UNCATEGORIZED = list(map(chr, [0xa9, 0xae, 0x203c, 0x2049, 0x2122, 0x2139, *range(0x2194, 0x2199), 0x21a9, 0x21aa, 0x231a, 0x231b, *range(0x23e9, 0x23ec), 0x23f0, 0x23f3, 0x25aa, 0x25ab, 0x25b6, 0x25c0, *range(0x25fb, 0x25fe), 0x2600, 0x2601, 0x260e, 0x2611, 0x2614, 0x2615, 0x261d, 0x263a, *range(0x2648, 0x2653), 0x2660, 0x2663, 0x2665, 0x2666, 0x2668, 0x267b, 0x267f, 0x2693, 0x26a0, 0x26a1, 0x26aa, 0x26ab, 0x26bd, 0x26be, 0x26c4, 0x26c5, 0x26ce, 0x26d4, 0x26ea, 0x26f2, 0x26f3, 0x26f5, 0x26fa, 0x26fd, 0x2934, 0x2935, *range(0x2b05, 0x2b07), 0x2b1b, 0x2b1c, 0x2b50, 0x2b55, 0x3030, 0x303d, 0x3297, 0x3299, 0x1f004, 0x1f0cf, *range(0x1f300, 0x1f30c), 0x1f30f, 0x1f311, *range(0x1f313, 0x1f315), 0x1f319, 0x1f31b, 0x1f31f, 0x1f320, 0x1f330, 0x1f331, 0x1f334, 0x1f335, *range(0x1f337, 0x1f34a), *range(0x1f34c, 0x1f34f), *range(0x1f351, 0x1f37b), *range(0x1f380, 0x1f393), *range(0x1f3a0, 0x1f3c4), 0x1f3c6, 0x1f3c8, 0x1f3ca, *range(0x1f3e0, 0x1f3e3), *range(0x1f3e5, 0x1f3f0), *range(0x1f40c, 0x1f40e), 0x1f411, 0x1f412, 0x1f414, *range(0x1f417, 0x1f429), *range(0x1f42b, 0x1f43e), 0x1f440, *range(0x1f442, 0x1f464), *range(0x1f466, 0x1f46b), *range(0x1f46e, 0x1f4ac), *range(0x1f4ae, 0x1f4b5), *range(0x1f4b8, 0x1f4eb), 0x1f4ee, *range(0x1f4f0, 0x1f4f4), 0x1f4f6, 0x1f4f7, *range(0x1f4f9, 0x1f4fc), 0x1f503, *range(0x1f50a, 0x1f514), *range(0x1f516, 0x1f52b), *range(0x1f52e, 0x1f53d), *range(0x1f550, 0x1f55b), *range(0x1f5fb, 0x1f600), 0x1f607, 0x1f608, 0x1f60e, 0x1f610, 0x1f611, 0x1f615, 0x1f617, 0x1f619, 0x1f61b, 0x1f61f, 0x1f626, 0x1f627, 0x1f62c, 0x1f62e, 0x1f62f, 0x1f634]))
ALL_EMOJI = (
    EMOTICONS
    + DINGBATS
    + TRANSPORT_SYMBOLS
    + ENCLOSED_CHARACTERS
    + UNCATEGORIZED
    + ADDITIONAL_EMOTICONS
    + ADDITIONAL_TRANSPORT_AND_MAP_SYMBOLS
    + OTHER_ADDITIONAL_SYMBOLS
)

"""
NLTK provides detailed word tags (superlative and comaprative adjectives).
To simplify, as we will be intersting only with 4 tags : verb, adverb, adjectif and noun, we construct   
a more general dcitionnay that maps between the general tags and specific ones

"""
WORD_TAG_DICT ={
    # base form, past tens, gerund or present participle, past participle, non-3rd person singular present, 3rd person singular present 
    "verb" : [ "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    # Adjective, comparative, superlative
    "adjectif" :["JJ", "JJR", "JJS"],
    # Adverb, comparative, superlative
    "adverb" : ["RB", "RBR", "RBS"],
    # Noun Singular, Plural
    "noun" : ["NN", "NNS"] 
}

MISSPELLED_FILE = FRAMEWORK_PATH + "transformation/" +"misspelled_en.txt"


TOP_N = 10  # he number of word similars to be retrieved 

"""
The number of sub-transformation in each transormation
"""
WORD_INSERT_N_TRANS = 4
WORD_DROP_N_TRANS = 1
WORD_REPLACE_N_TRANS = 4
CHAR_INSERT_N_TRANS = 3
CHAR_DROP_N_TRANS = 4
CHAR_REPLACE_N_TRANS = 3


"""
max valid number of transfromations 
"""

WORD_INSERT_MAX_TRANS = 7
WORD_DROP_MAX_TRANS = 4
WORD_REPLACE_MAX_TRANS = 6
CHAR_INSERT_MAX_TRANS = 10
CHAR_DROP_MAX_TRANS= 9
CHAR_REPLACE_MAX_TRANS= 7


""""
Vector sizes for each transformation
"""
WORD_INSERT_VECTOR_SIZE = 3
WORD_DROP_VECTOR_SIZE = 1
WORD_REPLACE_VECTOR_SIZE = 3
CHAR_INSERT_VECTOR_SIZE = 3
CHAR_DROP_VECTOR_SIZE = 3 
CHAR_REPLACE_VECTOR_SIZE = 3

MAX_WORDS = 1000
VALID_RATE = 0.25