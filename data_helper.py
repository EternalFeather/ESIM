import pandas as pd
import numpy as np
from string import punctuation as p
from config import Parameters as pm
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
import re, jieba
if pm.use_owndict:
    jieba.load_userdict(pm.jieba_dictionary)

jieba.suggest_freq(('亲', '工'), True)
jieba.suggest_freq(('对', '子'), True)


class Dataloader(object):
    def __init__(self):
        self.q1_data, self.q2_data, self.label = self.read_dataset(pm.train_data_path)
        self.embedding_index = self.load_pretrain_embedding(pm.embedding_path)
        if pm.clean_data:
            if pm.remove_stopwords:
                self.ignored_word = self.load_clean_words(pm.clean_path)
            self.cleaned_q1_data, self.cleaned_q2_data = [], []
            for text in self.q1_data:
                self.cleaned_q1_data.append(self.clean_data(text))
            for text in self.q2_data:
                self.cleaned_q2_data.append(self.clean_data(text))
        self.q1_sequences, self.q2_sequences, self.word_index = self.tokenizer()
        self.nb_words, self.embedding_matrix = self.prepare_embedding_matrix()

    def read_dataset(self, train_path):
        train = pd.read_csv(train_path)

        q1_data = train['Q1'].values
        q2_data = train['Q2'].values
        label = train['label'].values

        return q1_data, q2_data, label

    def load_pretrain_embedding(self, file):
        print('Indexing word vector...')
        embedding_index = word2vec.Word2Vec.load(file)

        return embedding_index

    def load_clean_words(self, file):
        clean_word_dict = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                # typo, correct = line.split(',')
                # clean_word_dict[typo] = correct
                clean_word_dict[line] = ','

        return clean_word_dict

    def clean_data(self, text):
        replace_numbers = re.compile(r'\d+', re.IGNORECASE)

        text = text.lower()
        text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
        text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"i’m", "i am", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r" +", "", text)

        stop_p = p + "~·！@#￥%……&*（）——=+-{}【】：；“”‘’《》，。？、|、"

        if pm.keep_punctuation:
            text = re.sub(r"”", "\"", text)
            text = re.sub(r"“", "\"", text)
            text = re.sub(r"´", "'", text)
            text = re.sub(r"—", " ", text)
            text = re.sub(r"’", "'", text)
            text = re.sub(r"‘", "'", text)
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
            text = re.sub(r"!", " ! ", text)
            text = re.sub(r"\/", " ", text)
            text = re.sub(r"\^", " ^ ", text)
            text = re.sub(r"\+", " + ", text)
            text = re.sub(r"\-", " - ", text)
            text = re.sub(r"\=", " = ", text)
            text = re.sub(r"'", " ", text)
            text = re.sub(r":", " : ", text)
            text = re.sub(r"−", " ", text)
            text = re.sub(r"\?", " ? ", text)
            text = re.sub(r"\^", " ^ ", text)
            text = re.sub(r"#", " # ", text)
            text = re.sub(r"￥", "$", text)
        else:
            for token in stop_p:
                text = re.sub(token, "", text)

        text = replace_numbers.sub('', text)

        if pm.remove_stopwords:
            text = "".join([word for word in text if word not in self.ignored_word])

        return text

    def tokenizer(self):
        tokenizer = Tokenizer(num_words=pm.MAX_NB_WORDS, filters='"#$%&()+,-./:;<=>@[\\]^_`{|}~\t\n')
        q1_cutted_data = self.segmentation(self.cleaned_q1_data)
        q2_cutted_data = self.segmentation(self.cleaned_q2_data)

        tokenizer.fit_on_texts(q1_cutted_data + q2_cutted_data)
        q1_sequences = tokenizer.texts_to_sequences(q1_cutted_data)
        q2_sequences = tokenizer.texts_to_sequences(q2_cutted_data)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        # Padding
        q1_data = pad_sequences(q1_sequences, maxlen=pm.MAX_SEQUENCE_LENGTH)
        print('Shape of q1_data tensor: ', q1_data.shape)
        q2_data = pad_sequences(q2_sequences, maxlen=pm.MAX_SEQUENCE_LENGTH)
        print('Shape of q2_data tensor: ', q2_data.shape)
        print('Shape of label tensor: ', self.label.shape)

        return q1_data, q2_data, word_index

    def segmentation(self, data):
        data_cutted = []
        for sentence in tqdm(data):
            seg_list = jieba.cut(sentence, cut_all=False)
            data_cutted.append(" ".join(seg_list))
        print('Finished segment for dataset.')

        return data_cutted

    def prepare_embedding_matrix(self):
        nb_words = min(pm.MAX_NB_WORDS, len(self.word_index))
        embedding_matrix = np.zeros((nb_words + 1, pm.EMBEDDING_DIM))

        print('Creating embedding matrix ...')
        for word, idx in self.word_index.items():
            if idx >= pm.MAX_NB_WORDS:
                continue
            if word in self.embedding_index.wv.vocab:
                embedding_vector = self.embedding_index.wv[word]
                embedding_matrix[idx] = embedding_vector

        return nb_words, embedding_matrix
