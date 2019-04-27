import os
import fileinput
import pprint
import numpy as np
from cortecx.constrcution import tools as t
from cortecx.constrcution import session


class DataObj:

    def __init__(self):
        self.data = None
        self.depth = 0
        self.dim = []

    def _list_to_dict_iterator(self, data: list) -> dict:
        worker = {}
        for i in data:
            if isinstance(i, list):
                worker.update({str(data.index(i)): self._list_to_dict_iterator(data[data.index(i)])})
            else:
                worker.update({str(data.index(i)): data[data.index(i)]})
        return worker

    def _dict_to_list_iterator(self, data: dict) -> list:
        worker = []
        for i in data.keys():
            if isinstance(data[i], dict):
                worker.append(self._dict_to_list_iterator(data[i]))
            else:
                worker.append(data[i])
        return worker

    def construct(self) -> dict:
        self.depth = t.depth_finder(list(self.data))
        self.dim = t.dim_finder(self.data)
        self.data = self._list_to_dict_iterator(list(self.data))
        return self.data

    def destruct(self) -> list:
        self.data = self._dict_to_list_iterator(dict(self.data))
        return self.data

    def wipe(self):
        self.data = None
        self.depth = None
        self.dim = None

    def feed(self, data: list):
        if isinstance(data, np.ndarray):
            self.dim = list(np.array(data).shape)
            self.data = np.array(data).tolist()
        elif isinstance(data, list):
            self.data = data
        elif isinstance(data, dict):
            self.data = self._dict_to_list_iterator(data)
        else:
            raise TypeError('Given data is not the data type expected: list, dict, ndarray')

    def retrieve(self, **kwargs):
        path = kwargs['path']
        path = os.path.abspath(path)

        dformat = kwargs['format']

        if dformat == 'csv':
            delimiter = kwargs['delimiter']
            self.data = []

            for line in fileinput.input([path]):
                self.data.append(line.replace('\n', '').split(delimiter))

            worker = {}
            rw = 1

            for value in self.data[0]:
                worker.update({value: {}})

            for key in worker.keys():
                for num in range(len(self.data)):
                    for row in self.data[num]:
                        worker[key].update({rw: self.data[num][list(worker.keys()).index(key)]})
                        rw += 1
                rw = 1
            self.data = worker
            return self  # Needs Fixing

        if dformat == 'json':
            file = open(path)

            self.data = eval(file.read())
            file.close()

            return self

        if dformat == 'conll-2003-ner':
            to_ignore = ['-DOCSTART- -X- -X- O', '\n']

            lines = []
            sentence = []
            temp = []
            ner = []
            for line in fileinput.input([path]):
                line = str(line)
                line = line.split(' ')
                if line[0] not in to_ignore:
                    sentence.append([line[0], line[1]])
                    temp.append(line[3].replace('\n', ''))
                else:
                    lines.append(sentence)
                    ner.append(temp)
                    sentence = []
                    temp = []
            temp, sentence = [], []
            self.data = lines, ner
            return self

        if dformat == 'conll-2000-pos':
            to_ignore = ['-DOCSTART- -X- -X- O', '\n']

            lines = []
            sentence = []
            temp = []
            pos = []
            for line in fileinput.input([path]):
                line = str(line)
                line = line.split(' ')
                if line[0] not in to_ignore:
                    sentence.append([line[0], line[2].replace('\n', '')])
                    temp.append(line[1].replace('\n', ''))
                else:
                    lines.append(sentence)
                    pos.append(temp)
                    sentence = []
                    temp = []
            temp, sentence = [], []
            self.data = lines, pos
            return self

        if dformat == 'conll-2000-noun':
            to_ignore = ['-DOCSTART- -X- -X- O', '\n']

            lines = []
            sentence = []
            temp = []
            pos = []
            for line in fileinput.input([path]):
                line = str(line)
                line = line.split(' ')
                if line[0] not in to_ignore:
                    sentence.append([line[0], 0])
                    temp.append(line[2].replace('\n', ''))
                else:
                    lines.append(sentence)
                    pos.append(temp)
                    sentence = []
                    temp = []
            temp, sentence = [], []
            self.data = lines, pos
            return self

        if dformat == 'text':
            f = open(path)
            self.data = str(f.read())
            f.close()
            return self


class Word:

    def __init__(self, word: str):
        self.word = word.lower()

    @property
    def stem(self):
        return None  # Not Finished

    @property
    def syllables(self):
        count = 0
        vowels = "aeiouy"
        if self.word[0] in vowels:
            count += 1
        for index in range(1, len(self.word)):
            if self.word[index] in vowels and self.word[index - 1] not in vowels:
                count += 1
        if self.word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1

        return count

    @property
    def prefix(self):
        return self.prefix  # Not Finished

    @property
    def vector(self):
        try:
            return session.cortecx_sess.word_embeddings[self.word]
        except KeyError:
            return np.zeros(300)

    @property
    def suffix(self):
        return self.suffix  # Not Finished

    @property
    def is_stop(self) -> bool:
        try:
            return session.cortecx_sess.stop_words[self.word]
        except KeyError:
            return False

    def similarity(self, second_word):
        pass  # Not Finished

    @property
    def definition(self):
        try:
            return session.cortecx_sess.word_dictionary[self.word]
        except KeyError:
            return KeyError('Word is not supported')

    @property
    def synonyms(self):
        try:
            return session.cortecx_sess.word_thesaurus[self.word]
        except KeyError:
            return KeyError('Word is not supported')

    def __str__(self) -> str:
        return str(self.word)

    def __add__(self, other):
        if isinstance(other, Word):
            other = str(other)
        if isinstance(other, str):
            other = other
        return self.word + other

    def __len__(self) -> int:
        return len(self.word)

    def __getitem__(self, item):
        return self.word[item]

    def __eq__(self, other) -> bool:
        if isinstance(other, Word):
            return self.word == other.word

        if isinstance(other, str):
            return self.word == other


class TextObj:

    def __init__(self, text):
        self.text = text
        if isinstance(self.text, DataObj):
            self.text = self.text.data
            if not isinstance(self.text, str):
                raise TypeError('TextObj can only take text data from DataObj')

    def wipe(self):
        self.text = None

    def __str__(self):
        return str(self.text)
