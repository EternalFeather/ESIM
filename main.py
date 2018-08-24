import warnings, os
import tensorflow as tf
import numpy as np
from data_helper import Dataloader
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import Parameters as pm
from models import get_ESIM_model
warnings.filterwarnings('ignore')


# Init settings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5


def train_model_by_logloss(model, batch_size, train_x, train_y, val_x, val_y, fold_id):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    best_model_path = pm.model_path + str(fold_id) + '.h5'
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    hist = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                     epochs=50, batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])
    best_val_score = min(hist.history['val_loss'])
    predictions = model.predict(val_x)
    auc = roc_auc_score(val_y, predictions)
    print('AUC Score : ', auc)

    return model, best_val_score, auc, predictions


def train_folds(x, y, fold_count, batch_size, get_model_func):
    fold_size = len(x) // fold_count
    models, fold_predictions = [], []
    score, total_auc = 0, 0
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_count - 1:
            fold_end = len(x)

        train_x = np.concatenate([x[:fold_start], x[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = x[fold_start: fold_end]
        val_y = y[fold_start: fold_end]

        print('In fold {}'.format(fold_id + 1))
        model, best_val_score, auc, fold_prediction = train_model_by_logloss(get_model_func, batch_size,
                                                                             train_x, train_y, val_x, val_y,
                                                                             fold_id)
        score += best_val_score
        total_auc += auc
        fold_predictions.append(fold_prediction)
        models.append(model)

    return models, score / fold_count, total_auc / fold_count, fold_predictions


def train():
    # q1 & q2 sequences (after tokenize operation) + label + embedding_matrix
    data_loader = Dataloader()
    if os.path.exists(pm.model_path):
        os.makedirs(pm.model_path)

    model = get_ESIM_model(pm.MAX_NB_WORDS, pm.EMBEDDING_DIM, data_loader.embedding_matrix,
                           pm.RECURRENT_UNITS, pm.DENSE_UNITS, pm.DROPOUT_RATE,
                           pm.MAX_SEQUENCE_LENGTH, 1)
    model.summary()

    models, val_loss, total_auc, fold_predictions = train_folds(data_loader.q1_sequences,
                                                                data_loader.q2_sequences,
                                                                10,
                                                                pm.BATCH_SIZE,
                                                                model)

    print('Overall val-loss: {}, AUC {}'.format(val_loss, total_auc))


if __name__ == '__main__':
    train()



