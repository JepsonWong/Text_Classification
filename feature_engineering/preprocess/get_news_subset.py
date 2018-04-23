#encoding:utf-8

import codecs
import collections
# import jieba.posseg as pseg
import os
# import tensorflow.contrib.keras as kr

class get_news_subset(object):
    def __init__(self, news_train_path, encoding, suffix_accepted='txt,TXT', words = {}, characters = {}):
        self.news_train_path = news_train_path
        self.encoding = encoding
        self.suffix_accepted = tuple(suffix_accepted.split(','))

        self.news_train_contents = []
        self.news_train_labels = []
        self.news_train_word_ids = []
        self.news_train_character_ids = []
        self.news_train_contents_words = []
        # self.news_train_labels_id = []

        self.words = words
        self.characters = characters
        self.stop_words = []
        self.words_maxlen = 0
        self.characters_maxlen = 0

    def set_stopwords(self, stopwords_path):
        _get_abs_path = lambda xpath: os.path.normpath(os.path.join(os.getcwd(), xpath))
        abs_path = _get_abs_path(stopwords_path)
        if not os.path.isfile(abs_path):
            raise Exception("stop_words file: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.append(line)

    def add_content_and_label(self, path, encoding='utf-8'):
        categories = {u'体育': 1, u'财经': 2, u'房产': 3, u'家居': 4, u'教育': 5, u'科技': 6, u'时尚': 7, u'时政': 8, u'游戏': 9,
                      u'娱乐': 0}
        with codecs.open(path, encoding=encoding, errors='ignore') as m_f:
            for line in m_f:
                try:
                    label, content = line.strip().split("\t")
                    if content:
                        self.news_train_contents.append(content)
                        self.news_train_labels.append(categories[label])
                except:
                    pass

    def to_words(self):
        all_words = []
        if self.news_train_contents:
            self.generate_content_and_label()
        if self.words:
            self.generate_words()
        for content in self.news_train_contents:
            context = self.__cut1_stop(content)
            result = []
            for i in context:
                if i in self.words:
                    result.append(i)
            all_words.append(result)
        self.news_train_contents_words = all_words

    def to_words_ids(self):
        data_id = []
        if not self.news_train_contents:
            self.generate_content_and_label()
        if not self.words:
            self.generate_words()
        for i in range(len(self.news_train_contents)):
            data_id.append([self.words[x] for x in self.news_train_contents[i] if x in self.words])
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        # x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.words_maxlen)
        x_pad = []
        for i in data_id:
            if len(i) <= 600:
                x_pad.append([0]*(600-len(i)) + i)
            else:
                x_pad.append(i[(len(i)-1-599):(len(i))])
        self.news_train_word_ids = x_pad

    def to_characters_ids(self):
        data_id = []
        if not self.news_train_contents:
            self.generate_content_and_label()
        if not self.characters:
            self.generate_characters()
        for i in range(len(self.news_train_contents)):
            data_id.append([self.characters[x] for x in self.news_train_contents[i] if x in self.characters])
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        #x_pad = kr.preprocessing.sequence.pad_sequences(data_id, 600) # self.characters_maxlen = 600
        x_pad = []
        for i in data_id:
            if len(i) <= 600:
                x_pad.append([0]*(600-len(i)) + i)
            else:
                x_pad.append(i[(len(i)-1-599):(len(i)-1)])
        self.news_train_character_ids = x_pad

    def generate_content_and_label(self):
        contents = []
        labels = []
        categories = {u'体育': 1, u'财经': 2, u'房产': 3, u'家居': 4, u'教育': 5, u'科技': 6, u'时尚': 7, u'时政': 8, u'游戏': 9,
                      u'娱乐': 0}
        with codecs.open(self.news_train_path, encoding=self.encoding, errors='ignore') as m_f:
            for line in m_f:
                try:
                    label, content = line.strip().split("\t")
                    if content:
                        contents.append(content)
                        labels.append(categories[label])
                except:
                    pass
        self.news_train_contents = contents
        self.news_train_labels = labels

    def get_content_and_label(self):
        if not self.news_train_contents:
            self.generate_content_and_label()
        return self.news_train_contents, self.news_train_labels

    def get_word_ids_and_labels(self):
        if not self.news_train_word_ids:
            self.to_words_ids()
        return self.news_train_word_ids, self.news_train_labels

    def get_character_ids_and_labels(self):
        if not self.news_train_character_ids:
            self.to_characters_ids()
        return self.news_train_character_ids, self.news_train_labels

    def get_contents_words_labels(self):
        if not self.news_train_contents_words:
            self.to_words()
        return self.news_train_contents_words, self.news_train_labels

    def __cut1_stop(self, text):
        # words = pseg.cut(text)
        words = []
        tags = []
        for item in words:
            if item.word in self.stop_words:
                continue
            if item.word.isdigit():
                continue
            tags.append(item.word)
        return tags

    def generate_characters(self):
        """构建词汇表"""
        all_data = []
        for content in self.news_train_contents:
            if (len(content) > self.characters_maxlen):
                self.characters_maxlen = len(content)
            all_data.extend(content)
        counter = collections.Counter(all_data)
        count_pairs = counter.most_common(4999) # (vocab_size - 1) character表的个数
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)
        word_to_id = dict(zip(words, range(len(words))))
        self.characters = word_to_id

    def generate_words(self):
        all_words = []
        for content in self.news_train_contents:
            context = self.__cut1_stop(content)
            if (len(context) > self.words_maxlen):
                self.words_maxlen = len(context)
                # print self.words_maxlen
            all_words = all_words + context
        counter = collections.Counter(all_words)
        count_pairs = counter.most_common(9999) # (words_size - 1) word表的个数
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)
        word_to_id = dict(zip(words, range(len(words))))
        self.words = word_to_id

    def get_words(self):
        if not self.words:
            self.generate_words()
        return self.words

    def get_characters(self):
        if not self.characters:
            self.generate_characters()
        return self.characters

if __name__ == '__main__':
    news = get_news_subset("../../data/THUCNews的一个子集/cnews.train.txt", "utf-8")
    news.set_stopwords('../../data/stop_words_zh.utf8.txt')

    contents, label = news.get_content_and_label()

    # 下面两行一定在get_content_and_label下面
    news.add_content_and_label("../../data/THUCNews的一个子集/cnews.test.txt")
    news.add_content_and_label("../../data/THUCNews的一个子集/cnews.val.txt")
    print len(contents)
    print len(label)
    print contents[0]
    print label[0]
    print contents[64999]
    print label[64999]
    '''
    print "words:"
    words = news.get_words()
    print "characters:"
    characters = news.get_characters()
    print len(words)
    print len(characters)

    news.to_characters_ids()
    print "character_size: "
    print news.characters_maxlen
    print "content: "
    print len(contents[0])
    print len(news.news_train_character_ids[0])
    print contents[0]
    print news.news_train_character_ids[0]

    news.to_words()
    print news.news_train_contents_words[0][0]
    '''
    '''
    train
    words:161224
    characters:5451
    '''