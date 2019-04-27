import os
import fileinput
import numpy as np
from pprint import pprint
from cortecx.constrcution.foundation import DataObj, TextObj, Word
from cortecx.constrcution.tools import Tokenizer
from cortecx.constrcution import tools as t
from cortecx.constrcution import session


class Parser(Word):

    possible_types = [
        'dictionary',
        'list',
        'tuple'
    ]

    possible_params = [
        'original',
        'pos',
        'chunks',
        'stem',
        'lemma',
        'vector',
        'entity',
        'definition',
        'synonyms',
        'is_stop'
    ]

    def __init__(self, as_type='dictionary'):
        super(Word, self).__init__()
        self.text = str
        self.params = []
        self.processed = []
        self.all = {}

        if as_type not in Parser.possible_types:
            raise KeyError('Type specified not in possible types: {}'.format(Parser.possible_types))

    def include(self, parameter: str):
        self.params.append(parameter.lower())

    def reset(self):
        self.params = []

    def _run_noun_pred(self):
        noun_model = session.cortecx_sess.noun_model
        in_data = t.padding(self.processed[0], 60, pad_char=np.zeros(301))

        pred = noun_model.predict([[in_data]])[0]
        pred = [np.argmax(element) for element in pred]
        return pred

    def _decode_noun_pred(self, noun_prediction):
        decoded = []

        n_map = t.noun_tags
        n_map = t.reverse_dictionary(n_map)
        for n, pred in enumerate(noun_prediction):
            try:
                decoded.append({self.processed[1][n]: {'chunk': n_map[pred]}})
            except IndexError or ValueError:
                try:
                    decoded.append({self.processed[1][n]: {'chunk': 'N/A'}})
                except IndexError:
                    break
        self.all.update({'chunks': decoded})
        return decoded

    def _run_pos_pred(self):
        n_pred = self._run_noun_pred()
        for n, pred in enumerate(self.processed[0]):
            self.processed[0][n][-1] = n_pred[n]

        pos = session.cortecx_sess.pos_model
        inpt = t.padding(self.processed[0], 60, pad_char=np.zeros(301))

        pos_pred = pos.predict([[inpt]])[0]
        pos_pred = [np.argmax(element) for element in pos_pred]
        return pos_pred

    def _decode_pos_pred(self, pos_prediction):
        decoded = []

        p_map = t.pos_pos_tags
        p_map = t.reverse_dictionary(p_map)
        for n, pred in enumerate(pos_prediction):
            try:
                decoded.append({self.processed[1][n]: {'pos': p_map[pred]}})
            except IndexError or ValueError:
                try:
                    decoded.append({self.processed[1][n]: {'pos': 'N/A'}})
                except IndexError:
                    break
        self.all.update({'pos': decoded})
        return decoded

    def _run_ner_pred(self):
        p_pred = self._run_pos_pred()
        p_pred = t.convert_encoding_pos_to_ner(p_pred)

        for i, element in enumerate(self.processed[0]):
            self.processed[0][i][-1] = p_pred[i]

        ner = session.cortecx_sess.ner_model
        inpt = t.padding(self.processed[0], 60, pad_char=np.zeros(301))

        ner_pred = ner.predict([[inpt]])[0]
        ner_pred = [np.argmax(element) for element in ner_pred]
        return ner_pred

    def _decode_ner_pred(self, ner_prediction):
        decoded = []

        ne_map = t.ner_tags
        ne_map = t.reverse_dictionary(ne_map)
        for n, pred in enumerate(ner_prediction):
            try:
                decoded.append({self.processed[1][n]: {'entity': ne_map[pred]}})
            except IndexError or ValueError:
                try:
                    decoded.append({self.processed[1][n]: {'entity': 'N/A'}})
                except IndexError:
                    break
        self.all.update({'entity': decoded})
        return decoded

    def _vector(self):
        vectors = []
        for word in t.tokenize(str(self.text)):
            word = Word(word)
            vectors.append({str(word): {'vector': list(word.vector)}})
        self.all.update({'vector': list(vectors)})
        return vectors

    def _definition(self):
        definitions = []
        for word in t.tokenize(str(self.text)):
            word = Word(word)
            definitions.append({str(word): {'definition': word.definition}})
        self.all.update({'definition': list(definitions)})
        return definitions

    def _synonyms(self):
        synonyms = []
        for word in t.tokenize(str(self.text)):
            word = Word(word)
            synonyms.append({str(word): {'synonyms': word.synonyms}})
        self.all.update({'synonyms': list(synonyms)})
        return synonyms

    def _is_stop(self):
        stop_words = []
        for word in t.tokenize(str(self.text)):
            word = Word(word)
            stop_words.append({str(word): {'is_stop': word.is_stop}})
        self.all.update({'is_stop': list(stop_words)})
        return stop_words

    @staticmethod
    def _further_combine_dicts(tree: dict):
        master = {}
        for i, dictionary in enumerate(tree):
            for key in dictionary.keys():
                for thing in dictionary[key]:
                    master.update(thing)
                dictionary.update({key: master})
                master = {}
                tree[i] = dictionary
        return tree

    def __call__(self, *args, **kwargs):
        text = args[0]
        answer = {}

        if isinstance(text, TextObj):
            text = text.text
        elif isinstance(text, str):
            pass
        else:
            raise TypeError('Parser takes either string or TextObj data types')

        self.text = text
        self.all.update({'original': str(self.text)})

        self.processed = t.for_models(self.text)

        self._decode_noun_pred(self._run_noun_pred())
        self._decode_pos_pred(self._run_pos_pred())
        self._decode_ner_pred(self._run_ner_pred())
        self._vector()
        self._definition()
        self._synonyms()
        self._is_stop()

        temp = {}

        if 'original' in self.params:
            answer.update({'text': self.all['original']})
            self.params.remove('original')

        if len(self.params) > 1:
            for param in self.params:
                if isinstance(temp, dict):
                    temp = self.all[param]
                else:
                    temp = t.combine_dicts(temp, self.all[param])
            temp = self._further_combine_dicts(temp)
            answer.update({'analysis': list(temp)})
        else:
            answer.update({'analysis': list(self.all[self.params[0]])})

        return answer

    def __str__(self):
        return str(self.text)

    def __getitem__(self, item):
        pass


class Encoder(Tokenizer):

    def __init__(self, data, **kwargs):
        super(Tokenizer, self).__init__()
        self.data = data
        self.encode_type = str
        self.token_map = {}
        self.transformed_data = []
        self.kwargs = kwargs

    def encode(self):
        if isinstance(self.data, DataObj):
            self.data = self.data.data
        if isinstance(self.data, Encoder):
            self.data = self.data.data
        if isinstance(self.data, str):
            self.data = t.Tokenizer(self.data).tokenize().tokens
        else:
            self.data = self.data

        self.encode_type = self.kwargs['encode_type']

        worker = []
        for element in self.data:
            worker.append(element) if element not in worker else None
        if self.encode_type not in ['integer', 'binary', 'frequency', 'conll']:
            raise ValueError('Encoding type must be either "integer", "binary", or "conll"')

        if self.encode_type == 'binary':
            keys = {}
            initializer = [0 for num in range(len(worker))]
            for i in range(len(worker)):
                initializer[i] = 1
                keys.update({worker[i]: initializer})
                initializer = [0 for num in range(len(worker))]
            self.token_map = keys
            for element in self.data:
                self.transformed_data.append(keys[element])
            return self.transformed_data

        if self.encode_type == 'integer':
            keys = {}
            initializer = 1
            for i in range(len(worker)):
                keys.update({worker[i]: initializer})
                initializer += 1
            self.token_map = keys
            for element in self.data:
                self.transformed_data.append(keys[element])
            return self.transformed_data

        if self.encode_type == 'conll':
            max_length = 0
            worker = []
            ner = []

            for element in self.data[1]:
                for value in element:
                    worker.append(value)

            encoder = Encoder(data=worker, encode_type='integer')
            encoder.encode()
            pos_map = encoder.token_map

            for element in self.data[1]:
                element = [pos_map[value] for value in element]
                element = t.padding(element, pad_len=60, pad_char=0)
                ner.append(element)
            self.token_map = pos_map

            worker = []

            self.data = self.data[0]

            for sentence in self.data:
                for word in sentence:
                    if worker.count(word[1]) >= 1:
                        pass
                    else:
                        worker.append(word[1])

            encoder = Encoder(data=worker, encode_type='integer')
            encoder.encode()
            pos_map = encoder.token_map
            # print(pos_map)

            for sentence in self.data:
                if len(sentence) > max_length:
                    max_length = len(sentence)
                else:
                    pass

            for sentence in self.data:
                worker = []
                for word in sentence:
                    new_word = list(Word(word[0]).vector)
                    new_word.append(float(pos_map[word[1]]))
                    worker.append(new_word)
                self.transformed_data.append(t.padding(worker, pad_len=60, pad_char=list(np.zeros(301))))
            self.data = self.transformed_data
            return self.data, ner

        if self.encode_type == 'frequency':
            tokens = self.data
            for token in tokens:
                tokens[tokens.index(token)] = tokens.count(token)
            self.transformed_data = tokens
            return self.transformed_data

    def wipe(self):
        self.data = []
        self.token_map = {}
        self.transformed_data = []


class Read(TextObj):

    def __init__(self, text: TextObj):
        super(TextObj, self).__init__()
        self.text = text
        pass

    def ask(self, question: str) -> str:
        pass

    def question_type(self) -> dict:
        pass

    def pos(self) -> dict:
        pass

    def sentiment(self) -> dict:
        pass

    def extract_ner(self) -> dict:
        pass

    def chunks(self) -> dict:
        pass
