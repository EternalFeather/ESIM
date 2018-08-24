import pandas as pd
import numpy as np
from string import punctuation as p
from config import Parameters as pm
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re, jieba
if pm.use_owndict:
    jieba.load_userdict(pm.jieba_dictionary)


jieba.add_word('关联关系', tag='n')
jieba.add_word('工单', tag='n')
jieba.suggest_freq(('亲', '工'), True)
jieba.add_word('云平台', tag='n')
jieba.add_word('全速云', tag='n')
jieba.add_word('云cas', tag='eng')
jieba.add_word('子菜单', tag='n')
jieba.suggest_freq(('对', '子'), True)
jieba.add_word('外网', tag='n')
jieba.add_word('内网', tag='n')
jieba.add_word('功能码', tag='n')
jieba.add_word('线上', tag='a')
jieba.add_word('清楚点', tag='a')
jieba.add_word('一级菜单', tag='n')
jieba.add_word('某一步', tag='m')
jieba.add_word('下一步', tag='m')
jieba.add_word('入职', tag='v')
jieba.add_word('找你', tag='v')
jieba.add_word('找谁', tag='v')
jieba.add_word('找不到', tag='v')
jieba.add_word('通用岗位', tag='v')
jieba.add_word('走流程', tag='v')
jieba.add_word('登录不了', tag='v')
jieba.add_word('被锁', tag='v')
jieba.add_word('没反应', tag='v')
jieba.add_word('挂了', tag='v')
jieba.add_word('登录不上', tag='v')
jieba.add_word('不存在', tag='v')
jieba.add_word('新员工', tag='n')
jieba.add_word('新用户', tag='n')
jieba.add_word('新组件', tag='n')
jieba.add_word('菜单名', tag='n')
jieba.add_word('用户名', tag='n')
jieba.add_word('岗位变动', tag='n')
jieba.add_word('人事变动', tag='n')
jieba.add_word('dns解析', tag='n')
jieba.add_word('wiki地址', tag='n')


class Dataloader(object):
    def __init__(self):
        self.q1_data, self.q2_data, self.label = self.read_dataset(pm.train_data_path)
        # self.embedding_index = self.load_pretrain_embedding(pm.embedding_path)
        if pm.clean_data:
            if pm.remove_stopwords:
                self.ignored_word = self.load_clean_words(pm.clean_path)
            self.cleaned_q1_data, self.cleaned_q2_data = [], []
            for text in self.q1_data:
                self.cleaned_q1_data.append(self.clean_data(text))
            for text in self.q2_data:
                self.cleaned_q2_data.append(self.clean_data(text))
        self.q1_sequences, self.q2_sequences, self.word_index = self.tokenizer()
        # self.embedding_matrix = self.prepare_embedding_matrix()

    def read_dataset(self, train_path):
        train = pd.read_csv(train_path)

        q1_data = train['Q1'].values
        q2_data = train['Q2'].values
        label = train['label'].values

        return q1_data, q2_data, label

    def load_pretrain_embedding(self, file):
        print('Indexing word vector...')
        embedding_index = {}
        f = open(file, 'r', encoding='utf-8')
        for line in f:
            values = line.strip()
            try:
                word = values[0]
                coefs = np.array(values[1:], dtype='float32')
                embedding_index[word] = coefs
            except:
                print('Error on : ', values[:3])
        f.close()
        print('Done! Total %s word vectors' % len(embedding_index))

        return embedding_index

    def load_clean_words(self, file):
        clean_word_dict = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                typo, correct = line.split(',')
                clean_word_dict[typo] = correct

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
        text = re.sub(r"登陆", "登录", text)
        text = re.sub(r"原件", "组件", text)
        text = re.sub(r"新人", "新员工", text)
        text = re.sub(r"变更", "变动", text)
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
        embedding_matrix = np.zeros((nb_words, pm.EMBEDDING_DIM))

        print('Creating embedding matrix ...')
        for word, idx in self.word_index.items():
            if idx >= pm.MAX_NB_WORDS:
                continue
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector

        return embedding_matrix
