import warnings, os
import tensorflow as tf
import numpy as np
from data_helper import Dataloader
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from config import Parameters as pm
from models import get_ESIM_model
warnings.filterwarnings('ignore')


# Init settings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def train_model_by_logloss(model, batch_size, train_q1, train_q2, train_y, val_q1, val_q2, val_y, fold_id):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    best_model_path = pm.model_path + 'ESIM_' + str(fold_id) + '.h5'
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    hist = model.fit([train_q1, train_q2], train_y, validation_data=([val_q1, val_q2], val_y),
                     epochs=50, batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])
    best_val_score = min(hist.history['val_loss'])
    predictions = model.predict([val_q1, val_q2])
    auc = roc_auc_score(val_y, predictions)
    print('AUC Score : ', auc)

    return model, best_val_score, auc, predictions


def train_folds(q1, q2, y, fold_count, batch_size, get_model_func):
    fold_size = len(q1) // fold_count
    models, fold_predictions = [], []
    score, total_auc = 0, 0
    write_file = open('./log/Logger.txt', 'w', encoding='utf-8')
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_count - 1:
            fold_end = len(q1)

        train_q1 = np.concatenate([q1[:fold_start], q1[fold_end:]])
        train_q2 = np.concatenate([q2[:fold_start], q2[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_q1 = q1[fold_start: fold_end]
        val_q2 = q2[fold_start: fold_end]
        val_y = y[fold_start: fold_end]

        print('In fold {}'.format(fold_id + 1))
        model, best_val_score, auc, fold_prediction = train_model_by_logloss(get_model_func, batch_size,
                                                                             train_q1, train_q2, train_y,
                                                                             val_q1, val_q2, val_y, fold_id)
        score += best_val_score
        total_auc += auc
        fold_predictions.append(fold_prediction)
        models.append(model)
        write_file.write('Fold {}\tLoss {}\tAUC {}\n'.format(fold_id + 1, best_val_score, auc))
        write_file.flush()
    
    write_file.close()

    return models, score / fold_count, total_auc / fold_count, fold_predictions


def train():
    # q1 & q2 sequences (after tokenize operation) + label + embedding_matrix
    data_loader = Dataloader()
    if not os.path.exists(pm.model_path):
        os.makedirs(pm.model_path)

<<<<<<< HEAD
    model = get_ESIM_model(data_loader.nb_words + 1, pm.EMBEDDING_DIM, data_loader.embedding_matrix,
=======
    model = get_ESIM_model(pm.MAX_NB_WORDS, pm.EMBEDDING_DIM, data_loader.embedding_matrix,
>>>>>>> 7fd62157b89495c374c043a213b00878f518b046
                           pm.RECURRENT_UNITS, pm.DENSE_UNITS, pm.DROPOUT_RATE,
                           pm.MAX_SEQUENCE_LENGTH, 1)
    # model = get_ESIM_model(pm.MAX_NB_WORDS, pm.EMBEDDING_DIM, None,
    #                        pm.RECURRENT_UNITS, pm.DENSE_UNITS, pm.DROPOUT_RATE,
    #                        pm.MAX_SEQUENCE_LENGTH, 1)
    print(model.summary())

    models, val_loss, total_auc, fold_predictions = train_folds(data_loader.q1_sequences,
                                                                data_loader.q2_sequences,
                                                                data_loader.label,
                                                                10,
                                                                pm.BATCH_SIZE,
                                                                model)

    print('Overall val-loss: {}, AUC {}'.format(val_loss, total_auc))


def evaluate():
    '''
    For training OOB(out-of-bag) Evaluation.
    '''
    data_loader = Dataloader()
    eval_predicts_list = []
    for fold_id in range(0, 10):
        model = get_ESIM_model(data_loader.nb_words + 1, pm.EMBEDDING_DIM, data_loader.embedding_matrix,
                           pm.RECURRENT_UNITS, pm.DENSE_UNITS, pm.DROPOUT_RATE,
                           pm.MAX_SEQUENCE_LENGTH, 1)
        model.load_weights(pm.model_path + 'ESIM_' + str(fold_id) + '.h5')
        eval_predict = model.predict([data_loader.q1_sequences, data_loader.q2_sequences], 
                                     batch_size=pm.BATCH_SIZE, verbose=1)
        eval_predicts_list.append(eval_predict)
    
        train_auc = roc_auc_score(data_loader.label, eval_predict)
        train_loss = log_loss(data_loader.label, eval_predict)
        train_acc = accuracy_score(data_loader.label, eval_predict.round())
        train_precision = precision_score(data_loader.label, eval_predict.round())
        train_recall = recall_score(data_loader.label, eval_predict.round())
        train_f1_score = f1_score(data_loader.label, eval_predict.round())
        print('Training AUC:{}\tLOSS:{}\tACCURACY:{}\tPRECISION:{}\tRECALL:{}\tF1_SCORE:{}'.format(
            train_auc, train_loss, train_acc, train_precision, train_recall, train_f1_score))

    
    train_fold_predictions = np.zeros(eval_predicts_list[0].shape)
    for fold_predict in eval_predicts_list:
        train_fold_predictions += fold_predict
    train_fold_predictions /= len(eval_predicts_list)

    train_auc = roc_auc_score(data_loader.label, train_fold_predictions)
    train_loss = log_loss(data_loader.label, train_fold_predictions)
    train_acc = accuracy_score(data_loader.label, train_fold_predictions.round())
    train_precision = precision_score(data_loader.label, train_fold_predictions.round())
    train_recall = recall_score(data_loader.label, train_fold_predictions.round())
    train_f1_score = f1_score(data_loader.label, train_fold_predictions.round())
    print('Training AUC:{}\tLOSS:{}\tACCURACY:{}\tPRECISION:{}\tRECALL:{}\tF1_SCORE:{}'.format(
        train_auc, train_loss, train_acc, train_precision, train_recall, train_f1_score))


if __name__ == '__main__':
    # train()
    evaluate()



