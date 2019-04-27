import os
import sys
import time
import psutil
import threading
import platform
import fileinput
import zipfile
import io
import ssl
import pickle
import gzip
import cortecx as c
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool


class CortecxSessionError(UserWarning):
    pass


universal_path = str(c.__path__[0])

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
    def to_gigabyte(nbbytes: int) -> float:
        return nbbytes / 1e9

    @staticmethod
    def build_dict(path):
        file = gzip.GzipFile(path, 'rb')
        word_dictionary = pickle.load(file)
        file.close()
        return word_dictionary

    @staticmethod
    def build_theo(path):
        file = gzip.GzipFile(path, 'rb')
        word_thesaurus = pickle.load(file)
        file.close()
        return word_thesaurus

    @staticmethod
    def build_stop(path):
        file = gzip.GzipFile(path)
        stop_words = pickle.load(file)
        file.close()
        return stop_words

    @staticmethod
    def build_embeddings(path):
        file = gzip.GzipFile(path, 'rb')
        word_embeddings = pickle.load(file)
        file.close()
        return word_embeddings

    @staticmethod
    def build(f_path):
        file = gzip.GzipFile(f_path, 'rb')
        loaded = pickle.load(file)
        file.close()
        return loaded

    def build_pos_model(self):
        pos_file = open('{}/models/trained_models/IsaacPOSModel_v1'.format(universal_path), 'rb')
        self.pos_model = pickle.load(pos_file)
        pos_file.close()

    def build_ner_model(self):
        ner_file = open('{}/models/trained_models/IsaacNERModel_v1'.format(universal_path), 'rb')
        self.ner_model = pickle.load(ner_file)
        ner_file.close()

    def build_noun_model(self):
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

        paths = [
            '{}/data/embeddings'.format(universal_path),
            '{}/data/dictionary-words'.format(universal_path),
            '{}/data/english-stop-words'.format(universal_path),
            '{}/data/thesaurus-words'.format(universal_path),
        ]

        if create_threads is True:

            pool = ThreadPool(16)

            results = pool.map(self.build, paths)

            pool.close()
            pool.join()

            self.word_embeddings = results[0]
            self.word_dictionary = results[1]
            self.stop_words = results[2]
            self.word_thesaurus = results[3]

        else:
            results = []
            for path in paths:
                results.append(self.build(path))

            self.word_embeddings = results[0]
            self.word_dictionary = results[1]
            self.stop_words = results[2]
            self.word_thesaurus = results[3]

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.build_noun_model()
        self.build_pos_model()
        self.build_ner_model()

        global cortecx_sess
        cortecx_sess = self
        return cortecx_sess

    @staticmethod
    def end():
        global cortecx_sess
        cortecx_sess = CortecxSessionError()
        return cortecx_sess


class MultiCore:

    def __init__(self):
        pass  # Not Finished
