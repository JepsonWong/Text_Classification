#encoding:utf-8

class word2vec_zh(object):
    def __init__(self):
        pass

class glove_zh(object):
    def __init__(self):
        pass

class polyglot_zh(object):
    def __init__(self):
        pass

    def load_pkl_file(self, file_name):
        import cPickle as pickle
        with open(file_name, 'rb') as fr:
            inf = pickle.load(fr)
        print inf

if __name__ == '__main__':
    polyglot_zh_example = polyglot_zh()
    polyglot_zh_example.load_pkl_file("../../../data/word_vectors/polyglot/polyglot-zh.pkl")