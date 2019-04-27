import numpy as np
import cortecx.constrcution.session as session


class HELP:

    def __init__(self):
        self.pos_codes = {
            'CC': 'Coordinating Conjunction',
            'CD': 'Cardinal number',
            'DT': 'Determiner',
            'EX': 'Existential "there"',
            'FW': 'Foreign Word',
            'IN': 'Preposition/Subordinating Conjunction',
            'JJ': 'Adjective',
            'JJR': 'Adjective, comparative',
            'JJS': 'Adjective, superlative',
            'LS': 'List item maker',
            'MD': 'Modal',
            'NN': 'Noun, singular or mass',
            'NNS': 'Noun, plural',
            'NNP': 'Proper noun, singular',
            'NNPS': 'Proper noun, plural',
            'PDT': 'Predeterminer',
            'POS': 'Possessive ending',
            'PRP': 'Personal pronoun',
            'PP$': 'Possessive pronoun',
            'RB': 'Adverb',
            'RBR': 'Adverb, comparative',
            'RBS': 'Adverb, superlative',
            'RP': 'Particle',
            'SYM': 'Symbol',
            'TO': 'To',
            'UH': 'Interjection',
            'VB': 'Verb, base form',
            'VBD': 'Verb, past tense',
            'VBG': 'Verb, gerund/present participle',
            'VBN': 'Verb, past principle',
            'VBP': 'Verb, non-3rd ps. sing. present',
            'VBZ': 'Verb, 3rd ps. sing. present',
            'WDT': 'wh-determiner',
            'WP': 'wh-pronoun',
            'WP$': 'Possessive wh-pronoun',
            'WRB': 'wh-adverb',
            '#': 'Pound sign',
            '$': 'Dollar sign',
            '.': 'Sentence-final punctuation',
            ',': 'Comma',
            ':': 'Colon',
            '(': 'Left Bracket',
            ')': 'Right Bracket',
        }

        self.chunk_codes = {
            'NP': 'Noun Phrase',
            'VP': 'Verb Phrase',
            'PP': 'Prepositional Phrase',
            'ADVP': 'Adverb Phrase',
            'SBAR': 'Subordinate Clause',
            'ADJP': 'Adjective Phrase',
            'PRT': 'Particles',
            'CONJP': 'Conjunction Phrase',
            'INTJ': 'Interjection',
            'LST': 'List Maker',
            'UCP': 'Unlike Coordinate Phrase',
            'O': 'Other'
        }

        self.ner_codes = {

        }


class Tokenizer:

    def __init__(self, text: str):
        self.tokens = list
        self.text = text
        self.map = {}
        self.special = ['?', '.', ',', '!', ':', ';', "'", '"', '+', '-', '=',
                        '_', '@', '#', '$', '%', '^', '&', '*', '(', ')', '<', '>']

        self.alphabet = 'abcdefghijklmnopABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def tokenize(self) -> [list, dict]:
        text = self.text

        for letter in self.alphabet:
            for char in self.special:
                text = text.replace('{}{}'.format(letter, char), '{} {}'.format(letter, char))
                text = text.replace('{}{}'.format(char, letter), '{} {}'.format(char, letter))

        for char in self.special:
            text = text.replace('{}'.format(char), ' {}'.format(char))

        tokens = list(text.split(' '))

        try:
            tokens.remove('')
        except ValueError:
            pass

        value = 0
        for token in tokens:
            self.map.update({token: value})
            value += 1
        self.tokens = tokens
        return self

    def sentence_tokenize(self) -> list:
        text = self.text
        return text.split('. ')

    def reconstruct(self) -> str:
        print(self.text)
        if type(self.text) is not list:
            raise TypeError('Reconstruct reconstructs word tokens therefore input must be a list')

        self.tokens = list(self.text)
        new_string = ''

        for element in self.tokens:
            new_string = new_string + str(element) + ' '

        return new_string


def to_data_obj(self, data: list) -> dict:
    worker = {}
    for i in data:
        if isinstance(i, list):
            worker.update({str(data.index(i)): self.to_data_obj(data[data.index(i)])})
        else:
            worker.update({str(data.index(i)): data[data.index(i)]})
    return worker


def summer(values: list) -> int:
    total = 0
    for value in values:
        total += value
    return total


def subtract(values: list) -> int:
    total = 0
    for value in values:
        total -= value
    return total


def factorial(values: list) -> int:
    total = 1
    for value in values:
        total = total * value
    return total


def exponent(value: int, power: int) -> int:
    total = value
    for num in range(power - 1):
        total = value * total
    return total


def flatten(data: list) -> list:
    limit = 1000
    counter = 0
    for i in range(len(data)):
        if (isinstance(data[i], (list, tuple)) and
                counter < limit):
            for a in data.pop(i):
                data.insert(i, a)
                i += 1
            counter += 1
            return flatten(data)
    return data


def reverse(data: list) -> list:
    return data[::-1]


def depth_finder(data: list) -> int:
    depth = 0

    while True:
        try:
            data = data[0]
            depth += 1
        except TypeError:
            break
    return depth


def dim_finder(data: list) -> list:
    try:
        data[0]
    except IndexError:
        return list(np.array(data).shape)


def sort(data: list, sort_by='ltg'):
    pass


def find(data: dict, coords: list):
    for index in coords:
        data = data[index]
    return data


def replace(data: dict, coords: list):
    pass  # Not Finished


def reverse_dictionary(dictionary: dict) -> dict:
    new = {}
    for key in dictionary.keys():
        new.update({dictionary[key]: key})
    return new


def to_gigabyte(nbbytes: int) -> float:
    return nbbytes/1e9


def tokenize(text: str) -> list:
    special = ['?', '.', ',', '!', ':', ';', "'", '"', '+', '-', '=',
               '_', '@', '#', '$', '%', '^', '&', '*', '(', ')', '<', '>']

    alphabet = 'abcdefghijklmnopABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for letter in alphabet:
        for char in special:
            text = text.replace('{}{}'.format(letter, char), '{} {}'.format(letter, char))
            text = text.replace('{}{}'.format(char, letter), '{} {}'.format(char, letter))

    for char in special:
        text = text.replace('{}'.format(char), ' {}'.format(char))

    tokens = list(text.split(' '))

    try:
        tokens.remove('')
    except ValueError:
        pass

    return tokens


def sentence_tokenizer(text: str) -> list:
    return text.split('. ')


def reconstruct(tokens: list) -> str:
    if type(tokens) is not list:
        raise TypeError('Reconstruct reconstructs word tokens therefore input must be a list')
    new_string = ''

    for element in tokens:
        new_string = new_string + str(element) + ' '

    return new_string


def infer_pad_char(data):
    pass  # Not Finished


def padding(data, pad_len, pad_char=' ', pad_type='after') -> list:
    tokens = list
    if isinstance(data, str):
        tokens = tokenize(data)
    if isinstance(data, list):
        tokens = data

    pad_amount = pad_len - len(tokens)

    if pad_type == 'after':
        if pad_amount > 0:
            tokens.extend([pad_char for num in range(pad_amount)])
        elif pad_amount == 0:
            pass
        elif pad_amount < 0:
            tokens = tokens[:pad_len]
        return tokens
    if pad_type == 'before':
        tokens = [pad_char for num in range(pad_amount)] + tokens
        return tokens


def reshape(data, new_shape):
    return np.reshape(np.array(data), newshape=new_shape).tolist()


def vectorize(text) -> list:
    tokens = list

    if isinstance(text, str):
        tokens = tokenize(text)
    if isinstance(text, list):
        tokens = text

    for token in tokens:
        try:
            tokens[tokens.index(token)] = session.cortecx_sess.word_embeddings[token]
        except KeyError:
            tokens[tokens.index(token)] = np.zeros(300)
    return tokens


def clean(self, include_punctuation=False, include_numbers=False, filters=None, custom_filter=None) -> str:
    deconstructed_text = []
    standard_filter = '`~@#$%^&*()_-+=}{][|\\/><'
    num_filter = '1234567890'
    punc_filter = ',.?":;!'
    apostraphe_filter = "'"
    punc_filter = punc_filter + apostraphe_filter
    reconstructed_text = ''
    final_filter = standard_filter

    if include_punctuation is True:
        final_filter = final_filter + punc_filter
    if include_numbers is True:
        final_filter = final_filter + num_filter
    if filters is not None:
        final_filter = final_filter + filters
    if custom_filter is not None:
        final_filter = custom_filter

    for element in self.text:
        if ord(element) < 128:
            if element not in final_filter:
                deconstructed_text.append(element)

    for element in deconstructed_text:
        reconstructed_text = reconstructed_text + str(element)

    alphabet = 'abscdefghijklmnopqrstuvwxyz'

    for element in alphabet:
        reconstructed_text = reconstructed_text.replace('.' + element, '.' + ' ' + element)

    i = 0

    while i < 6:
        spaces = ['  ', '   ', '     ', '       ', '         ', '           ']
        for element in spaces:
            reconstructed_text = reconstructed_text.replace(element, ' ')
        i += 1

    return reconstructed_text


"""
def remove_stop(text: (str, list)):
    to_reconstructor = []
    for token in tokenize(text):
        if Word(token).is_stop() is True:
            pass
        else:
            to_reconstructor.append(token)
    text = reconstruct(to_reconstructor)
    return text
"""


def chop(text: (str, list), chop_size, pad_char=' ') -> list:
    tokens = list
    if isinstance(text, list):
        pass
    if isinstance(text, str):
        tokens = tokenize(text)

    chopped_text = []
    i = 0
    x = 0
    y = chop_size
    while i <= (len(tokens) / chop_size):
        text = tokens[x:y]
        padding(text, pad_len=chop_size, pad_char=pad_char)
        chopped_text.append(list(text))
        x += chop_size
        y += chop_size
        i += 1

    for element in chopped_text:
        if element == [pad_char for num in range(chop_size)]:
            chopped_text.remove(element)
        else:
            continue
    return chopped_text


ner_pos_tags = {'-X-': 1, 'NNP': 2, 'VBZ': 3, 'JJ': 4, 'NN': 5, 'TO': 6, 'VB': 7, '.': 8, 'CD': 9, 'DT': 10,
                'VBD': 11, 'IN': 12, 'PRP': 13, 'NNS': 14, 'VBP': 15, 'MD': 16, 'VBN': 17, 'POS': 18, 'JJR': 19, '"': 20,
                'RB': 21, ',': 22, 'FW': 23, 'CC': 24, 'WDT': 25, '(': 26, ')': 27, ':': 28, 'PRP$': 29, 'RBR': 30,
                'VBG': 31, 'EX': 32, 'WP': 33, 'WRB': 34, '$': 35, 'RP': 36, 'NNPS': 37, 'SYM': 38, 'RBS': 39, 'UH': 40,
                'PDT': 41, "''": 42, 'LS': 43, 'JJS': 44, 'WP$': 45, 'NN|SYM': 46}

pos_pos_tags = {'NN': 1, 'IN': 2, 'DT': 3, 'VBZ': 4, 'RB': 5, 'VBN': 6, 'TO': 7, 'VB': 8, 'JJ': 9, 'NNS': 10,
                'NNP': 11, ',': 12, 'CC': 13, 'POS': 14, '.': 15, 'VBP': 16, 'VBG': 17, 'PRP$': 18, 'CD': 19, '``': 20,
                "''": 21, 'VBD': 22, 'EX': 23, 'MD': 24, '#': 25, '(': 26, '$': 27, ')': 28, 'NNPS': 29, 'PRP': 30,
                'JJS': 31, 'WP': 32, 'RBR': 33, 'JJR': 34, 'WDT': 35, 'WRB': 36, 'RBS': 37, 'PDT': 38, 'RP': 39, ':': 40,
                'FW': 41, 'WP$': 42, 'SYM': 43, 'UH': 44}

ner_tags = {'O': 1, 'B-ORG': 2, 'B-MISC': 3, 'B-PER': 4, 'I-PER': 5, 'B-LOC': 6, 'I-ORG': 7, 'I-MISC': 8, 'I-LOC': 9}


noun_tags = {'B-NP': 1, 'B-PP': 2, 'I-NP': 3, 'B-VP': 4, 'I-VP': 5, 'B-SBAR': 6, 'O': 7, 'B-ADJP': 8, 'B-ADVP': 9, 'I-ADVP': 10,
             'I-ADJP': 11, 'I-SBAR': 12, 'I-PP': 13, 'B-PRT': 14, 'B-LST': 15, 'B-INTJ': 16, 'I-INTJ': 17, 'B-CONJP': 18, 'I-CONJP': 19, 'I-PRT': 20,
             'B-UCP': 21, 'I-UCP': 22}


def convert_encoding_pos_to_ner(predictions: list) -> list:
    repred = reverse_dictionary(pos_pos_tags)
    converted = []
    for pred in predictions:
        try:
            converted.append(ner_pos_tags[repred[pred]])
        except KeyError:
            converted.append(0)
    return converted


def for_models(sentence: str) -> tuple:
    vectorized = []
    tokens = Tokenizer(sentence.lower()).tokenize()
    tokens = tokens.tokens
    for token in tokens:
        try:
            vector = session.cortecx_sess.word_embeddings[token]
        except KeyError:
            vector = np.zeros(300).tolist()
        vector = list(vector)
        vector.append(0)
        vectorized.append(vector)
    return vectorized, tokens


def combine_dicts(tree: list, branch: list):
    for n in range(len(tree)):
        combine_value = []
        word_key = list(tree[n].keys())[0]
        word_value = list(tree[n].values())
        branch_value = list(branch[n].values())
        combine_value.append(word_value)
        combine_value.append(branch_value)
        combine_value = flatten(combine_value)

        tree[n].update({word_key: combine_value})
    return tree
