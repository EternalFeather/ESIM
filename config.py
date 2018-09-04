
class Parameters(object):

    jieba_dictionary = './dataset/segdict.txt'
    embedding_path = '/home/linhx/word2vec/model/word2vec_wx'
    clean_path = './dataset/stopwords.txt'
    train_data_path = './dataset/dataset.csv'
    model_path = './checkpoint/pre_emb/'

    MAX_NB_WORDS = 30000
    BATCH_SIZE = 128
    EMBEDDING_DIM = 256
    MAX_SEQUENCE_LENGTH = 50
    RECURRENT_UNITS = 300
    DENSE_UNITS = 300
    DROPOUT_RATE = 0.5

    keep_punctuation = True
    clean_data = True
    remove_stopwords = False
    use_owndict = False



