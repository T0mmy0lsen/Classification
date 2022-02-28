import os
import re
import warnings

import lemmy
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize


class Text:

    stopwords = None
    lemmatizer = None

    def __init__(self):
        self.stopwords = self.ready_stop_words()
        self.lemmatizer = lemmy.load("da")

    @staticmethod
    def get_str_from_tokens(tokens):
        return " ".join(str(x) for x in tokens)

    @staticmethod
    def get_tokens_from_str(string):
        return string.split(" ")

    @staticmethod
    def get_stopwords_removed(tokens, stopwords=None):
        return [token for token in tokens if token not in stopwords]

    @staticmethod
    def get_lemma(lemmatizer, tokens):
        return [lemmatizer.lemmatize("", token)[0] for token in tokens]

    @staticmethod
    def get_tokenized_text(line, language="danish"):
        return [token.lower() for token in word_tokenize(line, language=language) if token.isalnum()]

    @staticmethod
    def get_beautiful_text(line):
        text = BeautifulSoup(line, "lxml").text
        text = re.sub('[\n.]', ' ', text)
        return text

    # --------------------------------------------------------------------------

    def ready_stop_words(
            self,
            language='danish',
            file_path_input='{}/input/stopwords.txt'.format(os.path.dirname(__file__)),
    ):
        print(os.path.dirname(__file__))

        """:return array of stopwords in :arg language"""
        if os.path.isfile(file_path_input):
            stopwords = []
            with open(file_path_input, 'r') as file_handle:
                for line in file_handle:
                    currentPlace = line[:-1]
                    stopwords.append(currentPlace)
            return stopwords

        url = "http://snowball.tartarus.org/algorithms/%s/stop.txt" % language
        text = requests.get(url).text
        stopwords = re.findall('^(\w+)', text, flags=re.MULTILINE | re.UNICODE)

        url_en = "http://snowball.tartarus.org/algorithms/english/stop.txt"
        text_en = requests.get(url_en).text
        stopwords_en = re.findall('^(\w+)', text_en, flags=re.MULTILINE | re.UNICODE)

        with open(file_path_input, 'w') as file_handle:
            for list_item in stopwords + stopwords_en:
                file_handle.write('%s\n' % list_item)

        return stopwords

    def get_process_text(self, text):
        text = self.get_beautiful_text(text)
        tokens = self.get_tokenized_text(text)
        tokens = self.get_lemma(tokens=tokens, lemmatizer=self.lemmatizer)
        tokens = self.get_stopwords_removed(tokens=tokens, stopwords=self.stopwords)
        return self.get_str_from_tokens(tokens)