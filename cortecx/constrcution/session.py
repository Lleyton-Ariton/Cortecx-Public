import os
import sys
import time
import psutil
import threading
import platform
import fileinput
import tensorflow as tf
import requests
import zipfile
import io
import ssl
import pickle
import gzip
import cortecx as c
from keras.models import load_model


class CortecxSessionError(UserWarning):
    pass


universal_path = str(c.__path__[0])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context


cortecx_sess = CortecxSessionError()


class Session:

    def __init__(self, suppress_warnings=False):
        self.word_dictionary = {}
        self.word_thesaurus = {}
        self.stop_words = {}
        self.word_embeddings = {}
        self.pos_model = None
        self.noun_model = None
        self.ner_model = None
        self.cpu_use = 0
        self.cpu_num = 0
        self.ram = 0
        self.ram_used = 0
        self.ram_free = 0
        self.swap = 0
        self.swap_used = 0
        self.swap_free = 0
        self.system_os = platform.system()
        self.sess_thread = None
        self.suppress_warnings = suppress_warnings
        self.embed_lim = 40000
        self.terminate = False
        self.pickled = True

    @staticmethod
    def _download_embeddings():
        r = requests.get('http://nlp.stanford.edu/data/glove.6B.zip')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('data/')

    def _load_embeddings(self):
        with gzip.open('embeddings', 'rb') as f:
            word_embeddings = pickle.load(f)
            self.word_embeddings = word_embeddings

    @staticmethod
    def to_gigabyte(nbbytes: int) -> float:
        return nbbytes / 1e9

    def build_dict(self):
        if self.pickled is False:
            try:
                letter = 97
                while letter <= 122:
                    file = open('{}/data/dictionary-words/{}.json'.format(universal_path, chr(letter)))
                    definitions = file.read()
                    definitions = list(eval(definitions))
                    for chunk in definitions:
                        word = str(chunk['word']).lower()
                        definition = str(chunk['description'])
                        if word in self.word_dictionary.keys():
                            self.word_dictionary.update(
                                {word: self.word_dictionary[word] + '\n\n{}'.format(definition)})
                        else:
                            self.word_dictionary.update({word: definition})
                    letter += 1
            except FileNotFoundError:
                letter = 97
                while letter <= 122:
                    file = open('{}/data/dictionary-words/{}.json'.format(universal_path, chr(letter)))
                    definitions = file.read()
                    definitions = list(eval(definitions))
                    for chunk in definitions:
                        word = str(chunk['word']).lower()
                        definition = str(chunk['description'])
                        if word in self.word_dictionary.keys():
                            self.word_dictionary.update(
                                {word: self.word_dictionary[word] + '\n\n{}'.format(definition)})
                        else:
                            self.word_dictionary.update({word: definition})
                    letter += 1
        elif self.pickled is True:
            file = gzip.GzipFile('{}/data/dictionary-words'.format(universal_path), 'rb')
            self.word_dictionary = pickle.load(file)

    def build_theo(self):
        if self.pickled is False:
            try:
                file_p = '{}/data/thesaurus-words.txt'.format(universal_path)
                file = open(file_p, 'r')
                for line in file.readlines():
                    wrds = line.replace('\n', '').split(',')
                    self.word_thesaurus.update({wrds[0]: wrds[1:]})
                file.close()
            except FileNotFoundError:
                file_p = '{}/data/thesaurus-words.txt'.format(universal_path)
                file = open(file_p, 'r')
                for line in file.readlines():
                    wrds = line.replace('\n', '').split(',')
                    self.word_thesaurus.update({wrds[0]: wrds[1:]})
                file.close()
        elif self.pickled is True:
            file = gzip.GzipFile('{}/data/thesaurus-words'.format(universal_path))
            self.word_thesaurus = pickle.load(file)
            file.close()

    def build_stop(self):
        try:
            file_p = os.path.abspath('{}/data/englishstopwords.txt'.format(universal_path))
            file = open(file_p, 'r')
            for line in file.readlines():
                self.stop_words.update({line.replace('\n', ''): True})
            file.close()
        except FileNotFoundError:
            file_p = os.path.abspath('{}/data/englishstopwords.txt'.format(universal_path))
            file = open(file_p, 'r')
            for line in file.readlines():
                self.stop_words.update({line.replace('\n', ''): True})
            file.close()

    def build_embeddings(self):
        if self.pickled is False:
            try:
                file_p = os.path.abspath('{}/data/word_vectors.txt'.format(universal_path))
                line_tracker = 0
                for line in fileinput.input([file_p]):
                    if line_tracker > self.embed_lim:
                        fileinput.close()
                        break
                    else:
                        splits = str(line).replace('\n', '').split(' ')
                        wrd = splits[0]
                        vector = [float(num) for num in splits[1:]]
                        self.word_embeddings.update({wrd: vector})
                        line_tracker += 1
                        continue
            except FileNotFoundError:
                file_p = os.path.abspath('{}/data/word_vectors.txt'.format(universal_path))
                line_tracker = 0
                for line in fileinput.input([file_p]):
                    if line_tracker > self.embed_lim:
                        fileinput.close()
                        break
                    else:
                        splits = str(line).replace('\n', '').split(' ')
                        wrd = splits[0]
                        vector = [float(num) for num in splits[1:]]
                        self.word_embeddings.update({wrd: vector})
                        line_tracker += 1
                        continue

        elif self.pickled is True:
            file = gzip.GzipFile('{}/data/embeddings'.format(universal_path), 'rb')
            self.word_embeddings = pickle.load(file)  # FIX ASAP
            file.close()

    def build_pos_model(self):
        if self.pickled is False:
            if self.pos_model is None:
                try:
                    self.pos_model = load_model('{}/models/trained_models/IsaacPOSModel_v1.h5'.format(universal_path))
                except OSError:
                    self.pos_model = load_model('{}/models/trained_models/IsaacPOSModel_v1.h5'.format(universal_path))
            else:
                self.pos_model = self.pos_model
        elif self.pickled is True:
            self.pos_model = self.pos_model
            pos_file = open('{}/models/trained_models/IsaacPOSModel_v1'.format(universal_path), 'rb')
            self.pos_model = pickle.load(pos_file)
            pos_file.close()

    def build_ner_model(self):
        if self.pickled is False:
            if self.ner_model is None:
                try:
                    self.ner_model = load_model('{}/models/trained_models/IsaacNERModel_v1.h5'.format(universal_path))
                except OSError:
                    self.ner_model = load_model('{}/models/trained_models/IsaacNERModel_v1.h5'.format(universal_path))
            else:
                self.ner_model = self.ner_model
        elif self.pickled is True:
            self.ner_model = self.ner_model
            ner_file = open('{}/models/trained_models/IsaacNERModel_v1'.format(universal_path), 'rb')
            self.ner_model = pickle.load(ner_file)
            ner_file.close()

    def build_noun_model(self):
        if self.pickled is False:
            if self.noun_model is None:
                try:
                    self.noun_model = load_model('{}/models/trained_models/IsaacNOUNModel_v1.h5'.format(universal_path))
                except OSError:
                    self.noun_model = load_model('{}/models/trained_models/IsaacNOUNModel_v1.h5'.format(universal_path))
            else:
                self.noun_model = self.noun_model
        elif self.pickled is True:
            noun_file = open('{}/models/trained_models/IsaacNOUNModel_v1'.format(universal_path), 'rb')
            self.noun_model = pickle.load(noun_file)
            noun_file.close()

    def check(self):

        red = '\033[31m'
        end = '\033[0m'

        if self.ram < 6:
            print(red + 'CortecxSysWarning: Your system has less than 6GB of RAM, '
                  'which is not recommended. Ideal is 8GB or above. ''Total RAM: {}'.format(self.ram) + end)

        if self.swap_used > 4:
            print(red + 'CortecxSysWarning: Your system has began to use swap, which is not recommended. '
                  'SWAP usage: {}'.format(self.swap_used))

        if self.ram - self.ram_used < 2:
            print(red + 'CortecxSysWarning: Your system is using too much RAM. '
                  'Your are approaching your maximum physical RAM. RAM usage: {}'.format(self.ram_used) + end)

        sys.stdout.write("\033[0m")

    def system(self):
        while self.terminate is False:
            mem = psutil.virtual_memory()

            self.cpu_num = psutil.cpu_count()
            self.cpu_use = psutil.cpu_percent(interval=None)

            self.ram = self.to_gigabyte(mem[0])
            self.ram_free = self.to_gigabyte(int(mem[1]))
            self.ram_used = self.to_gigabyte(mem[3])

            swap = psutil.swap_memory()
            self.swap = self.to_gigabyte(swap[0])
            self.swap_free = self.to_gigabyte(swap[3])
            self.swap_used = self.to_gigabyte(swap[1])

            if self.suppress_warnings is False:
                self.check()
            else:
                pass

            time.sleep(60)

    def set_params(self, embed_lim=40000, pickled=True):
        self.embed_lim = embed_lim
        self.pickled = pickled

    def interface(self, fp, **kwargs):
        if kwargs['embeddings'] is not True:
            pass
        else:
            file_p = os.path.abspath(fp)
            line_tracker = 0
            limit = kwargs['limit']
            for line in fileinput.input([file_p]):
                if line_tracker > limit:
                    fileinput.close()
                    break
                else:
                    splits = str(line).replace('\n', '').split(' ')
                    wrd = splits[0]
                    vector = [float(num) for num in splits[1:]]
                    self.word_embeddings.update({wrd: vector})
                    line_tracker += 1
                    continue

    def start(self, create_threads=True):

        self.build_noun_model()
        self.build_pos_model()
        self.build_ner_model()

        if create_threads is True:
            self.sess_thread = threading.Thread(target=self.system, args=())
            self.sess_thread.start()

            dict_thread = threading.Thread(target=self.build_dict, args=())
            dict_thread.start()
            dict_thread.join()

            theo_thread = threading.Thread(target=self.build_theo, args=())
            theo_thread.start()
            theo_thread.join()

            sword_thread = threading.Thread(target=self.build_stop, args=())
            sword_thread.start()
            sword_thread.join()

            if len(self.word_embeddings) == 0:
                word_thread = threading.Thread(target=self.build_embeddings, args=())
                word_thread.start()
                word_thread.join()

        else:
            pass

        global cortecx_sess
        cortecx_sess = self
        return cortecx_sess

    def end(self):
        self.terminate = True
        self.sess_thread._is_running = False


class MultiCore:

    def __init__(self):
        pass  # Not Finished
