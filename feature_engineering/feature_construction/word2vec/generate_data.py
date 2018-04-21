#encoding:utf-8

import jieba.posseg as pseg
import collections

import os,sys
sys.path.append("../../../")
# print(sys.path)

from feature_engineering.preprocess.get_corpus import get_corpus

class word_id():
    def __init__(self, get_corpus):
        self.corpus_files = get_corpus.get_files()
        self.vocabs = {}
        self.stop_words = []
        self.maxlen = 0

        self.content = []
        self.label = []

    def add_corpus(self, get_corpus):
        self.corpus_files = self.corpus_files + get_corpus.get_files()

    def set_stopwords(self, stopwords_path):
        _get_abs_path = lambda xpath: os.path.normpath(os.path.join(os.getcwd(), xpath))
        abs_path = _get_abs_path(stopwords_path)
        if not os.path.isfile(abs_path):
            raise Exception("tfidf: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.append(line)

    def file_to_word_ids(self, content):
        return [self.vocabs[word] for word in content if word in self.vocabs]

    def __cut1_no_stop(self, text):
        words = pseg.cut(text)
        tags = []
        for item in words:
            tags.append(item.word)
        return tags

    def __cut1_stop(self, text):
        words = pseg.cut(text)
        tags = []
        for item in words:
            if item.word in self.stop_words:
                continue
            if item.word.isdigit():
                continue
            tags.append(item.word)
        return tags

    def generate_vocab(self):
        all_words = []
        for file in self.corpus_files:
            # print file.get_name()
            content_text = file.get_content()
            context = self.__cut1_stop(content_text)
            if (len(context) > self.maxlen):
                self.maxlen = len(context)
            all_words = all_words + context
        counter = collections.Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        self.vocabs = word_to_id

    def generate_content_and_label(self):
        content = []
        label = []
        for file in self.corpus_files:
            content_text = file.get_content()
            context = self.__cut1_stop(content_text)
            content.append(self.file_to_word_ids(context))
            label.append(file.get_label())
        self.content = content
        self.label = label


    def get_vocabs(self):
        if not self.vocabs:
            self.generate_vocab()
        return self.vocabs

if __name__ == "__main__":
    corpus = get_corpus('../../../data/复旦大学中文语料库/corpus_train', 'gb18030')
    word_id_example = word_id(corpus)
    word_id_example.set_stopwords('../../../data/stop_words_zh.utf8.txt')
    vocabs = word_id_example.get_vocabs()
    word_id_example.generate_content_and_label()
    print len(word_id_example.content)
    #for i in word_id_example.content:
    #    print len(i)
    print len(word_id_example.label)
    #for i in word_id_example.label:
    #    print i

    print len(vocabs)
    print word_id_example.maxlen