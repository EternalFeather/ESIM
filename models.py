import keras
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model


def get_ESIM_model(nb_words, embedding_dim, embedding_matrix, recurrent_units, dense_units, dropout_rate, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                # embeddings_initializer='uniform',
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    input_q1_layer = Input(shape=(max_sequence_length,), dtype='int32', name='q1')
    input_q2_layer = Input(shape=(max_sequence_length,), dtype='int32', name='q2')

    embedding_sequence_q1 = BatchNormalization(axis=2)(embedding_layer(input_q1_layer))
    embedding_sequence_q2 = BatchNormalization(axis=2)(embedding_layer(input_q2_layer))

    final_embedding_sequence_q1 = SpatialDropout1D(0.25)(embedding_sequence_q1)
    final_embedding_sequence_q2 = SpatialDropout1D(0.25)(embedding_sequence_q2)

    rnn_layer_q1 = Bidirectional(LSTM(recurrent_units, return_sequences=True))(final_embedding_sequence_q1)
    rnn_layer_q2 = Bidirectional(LSTM(recurrent_units, return_sequences=True))(final_embedding_sequence_q2)

    attention = Dot(axes=-1)([rnn_layer_q1, rnn_layer_q2])
    w_attn_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_attn_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    align_layer_1 = Dot(axes=1)([w_attn_1, rnn_layer_q1])
    align_layer_2 = Dot(axes=1)([w_attn_2, rnn_layer_q2])

    subtract_layer_1 = subtract([rnn_layer_q1, align_layer_1])
    subtract_layer_2 = subtract([rnn_layer_q2, align_layer_2])

    multiply_layer_1 = multiply([rnn_layer_q1, align_layer_1])
    multiply_layer_2 = multiply([rnn_layer_q2, align_layer_2])

    m_q1 = concatenate([rnn_layer_q1, align_layer_1, subtract_layer_1, multiply_layer_1])
    m_q2 = concatenate([rnn_layer_q2, align_layer_2, subtract_layer_2, multiply_layer_2])

    v_q1_i = Bidirectional(LSTM(recurrent_units, return_sequences=True))(m_q1)
    v_q2_i = Bidirectional(LSTM(recurrent_units, return_sequences=True))(m_q2)

    avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
    avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
    maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
    maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)

    merged_q1 = concatenate([avgpool_q1, maxpool_q1])
    merged_q2 = concatenate([avgpool_q2, maxpool_q2])

    final_v = BatchNormalization()(concatenate([merged_q1, merged_q2]))
    output = Dense(units=dense_units, activation='relu')(final_v)
    output = BatchNormalization()(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(units=out_size, activation='sigmoid')(output)

    model = Model(inputs=[input_q1_layer, input_q2_layer], output=output)
    adam_optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    parallel_model = multi_gpu_model(model, gpus=2)

    parallel_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['binary_crossentropy', 'accuracy'])

    return parallel_model
