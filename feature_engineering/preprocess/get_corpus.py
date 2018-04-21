# encoding:utf-8

import os
import codecs


class get_corpus(object):
    def __init__(self, corpus_path, encoding, suffix_accepted='txt,TXT'):
        if os.path.isdir(corpus_path):
            self.corpus_path = corpus_path
            self.encoding = encoding
        else:
            raise Exception(str(corpus_path) + "does not exist!")

        self.corpus_files = []
        self.suffix_accepted = tuple(suffix_accepted.split(','))
        self.filenames = []
        self.labels = []

    def get_files(self):
        if not self.corpus_files:
            ret_files = []
            for root, _, files in os.walk(self.corpus_path, topdown=False):
                for name in files:
                    if name[0] == '.' or not name.endswith(self.suffix_accepted):
                        continue
                    if '-' in name:
                        label = name.split('-')[0][1:]
                    else:
                        label = -1
                    # os.path.join(root, name)生成相对目录，os.path.abspath根据os.getcwd()补充获得绝对路径
                    file_path = os.path.abspath(os.path.join(root, name))
                    ret_files.append(corpus_file(file_path, label, self.encoding))
            self.corpus_files = ret_files
        return self.corpus_files

    def get_filenames_and_labels(self):
        if not self.corpus_files:
            self.get_files()
        if self.filenames == [] or self.labels == []:
            self.filenames = []
            self.labels = []
            for file in self.corpus_files:
                self.filenames.append(file.get_name())
                self.labels.append(file.get_label())
        return self.filenames, self.labels


class corpus_file(object):
    def __init__(self, file_path, label, encoding):
        self.file_path = file_path
        self.label = label
        self.encoding = encoding
        with codecs.open(self.file_path, encoding=self.encoding, errors='ignore') as m_f:
            self.content = "".join(m_f.readlines())

    def get_content(self):
        return self.content

    def get_path(self):
        return self.file_path

    def get_name(self):
        return self.file_path.split('/')[-1]

    def get_label(self):
        return self.label


if __name__ == '__main__':
    corpus = get_corpus('../../data/复旦大学中文语料库/corpus_big', 'gb18030')
    files = corpus.get_files()
    print(files[0].get_path())
    print(files[0].get_name())
    print(files[0].get_label())
    #print(files[0].get_content())