import os
#os.environ["THEANO_FLAGS"] = "device=gpu0,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"
os.environ["THEANO_FLAGS"] = "floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"


import numpy as np
np.random.seed(1000)
from scipy.spatial.distance import cdist
from collections import namedtuple
from keras.models import Model
from keras.layers import Dense, Activation, Input, merge, Lambda
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
import keras
import sys
import re
import Score
from collections import defaultdict
from sklearn import linear_model
from keras.models import load_model


vec_dim=300
sent_vec_dim=300
ans_len_cut_off=40
max_qs_l = 23
max_ans_l = 40
word_vecs = {}
stop_words=[]
idf = defaultdict(float)
data_folder=""
qtype_map, qtype_invmap = {}, {}
QASample = namedtuple("QASample", "Qs Ans QsWords AnsWords Label")

def load_word2vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, vec_dim)
    return word_vecs

def load_vocab(vocab_file):
    file_reader = open(vocab_file)
    lines = file_reader.readlines()
    file_reader.close()
    vocab = {}
    for line in lines:
        parts = line.split('\t')
        qs = parts[0]
        ans = parts[1]
        qwords = qs.split()
        for word in qwords:
            if vocab.has_key(word):
                vocab[word] += 1
            else:
                vocab[word] = 1
        answords = ans.split()
        for word in answords:
            if vocab.has_key(word):
                vocab[word] += 1
            else:
                vocab[word] = 1
    return vocab

def load_samples(file):
    file_reader = open(file)
    lines = file_reader.readlines()
    file_reader.close()
    samples = []
    for line in lines:
        parts = line.split('\t')
        qs = parts[0]
        qs=qs.replace('\n','')
        ans = parts[1]
        ans=ans.replace('\n','')
        qwords = qs.split()
        answords = ans.split()
        label = int(parts[2].replace('\n', ''))
        sample = QASample(Qs=qs, Ans=ans, QsWords=qwords, AnsWords=answords, Label=label)
        samples.append(sample)
    return samples

def load_stop_words(stop_file):
    file_reader=open(stop_file)
    lines=file_reader.readlines()
    for line in lines:
        line=line.replace('\n','')
        stop_words.append(line)
    return stop_words

#def get_max_len(train_samples,dev_samples,test_samples):
#     max_qs_l = len(train_samples[0].QsWords)
#     for i in range(1, len(train_samples)):
#         if len(train_samples[i].QsWords) > max_qs_l:
#             max_qs_l = len(train_samples[i].QsWords)
#
#     for i in range(0, len(dev_samples)):
#         if len(dev_samples[i].QsWords) > max_qs_l:
#             max_qs_l = len(dev_samples[i].QsWords)
#
#     for i in range(0, len(test_samples)):
#         if len(test_samples[i].QsWords) > max_qs_l:
#             max_qs_l = len(test_samples[i].QsWords)
#
#     max_ans_l = len(train_samples[0].AnsWords)
#     for i in range(1, len(train_samples)):
#         if len(train_samples[i].AnsWords) > max_ans_l:
#             max_ans_l = len(train_samples[i].AnsWords)
#
#     for i in range(0, len(dev_samples)):
#         if len(dev_samples[i].AnsWords) > max_ans_l:
#             max_ans_l = len(dev_samples[i].AnsWords)
#
#     for i in range(0, len(test_samples)):
#         if len(test_samples[i].AnsWords) > max_ans_l:
#             max_ans_l = len(test_samples[i].AnsWords)
#     return max_qs_l, max_ans_l
#
# Implementation of Wang model

def compose_decompose(qmatrix, amatrix):
    qhatmatrix, ahatmatrix = f_match(qmatrix, amatrix, window_size=3)
    qplus, qminus = f_decompose(qmatrix, qhatmatrix)
    aplus, aminus = f_decompose(amatrix, ahatmatrix)
    return qplus, qminus, aplus, aminus

def f_match(qmatrix, amatrix, window_size=3):
    A = 1 - cdist(qmatrix, amatrix, metric='cosine')  # Similarity matrix
    Atranspose = np.transpose(A)
    qa_max_indices = np.argmax(A,
                               axis=1)  # 1-d array: for each question word, the index of the answer word which is most similar
    # Selecting answer word vectors in a window surrounding the most closest answer word
    qa_window = [range(max(0, max_idx - window_size), min(amatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 qa_max_indices]
    # Selecting question word vectors in a window surrounding the most closest answer word
    # Finding weights and its sum (for normalization) to find f_match for question for the corresponding window of answer words
    qa_weights = [(np.sum(A[qword_idx][aword_indices]), A[qword_idx][aword_indices]) for qword_idx, aword_indices in
                  enumerate(qa_window)]
    # Then multiply each vector in the window with the weights, sum up the vectors and normalize it with the sum of weights
    # This will give the local-w vecotrs for the Question sentence words and Answer sentence words.
    qhatmatrix = np.array([np.sum(weights.reshape(-1, 1) * amatrix[aword_indices], axis=0) / weight_sum for
                           ((qword_idx, aword_indices), (weight_sum, weights)) in
                           zip(enumerate(qa_window), qa_weights)])

    # Doing similar stuff for answer words
    aq_max_indices = np.argmax(A,
                               axis=0)  # 1-d array: for each   answer word, the index of the question word which is most similar
    aq_window = [range(max(0, max_idx - window_size), min(qmatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 aq_max_indices]
    aq_weights = [(np.sum(Atranspose[aword_idx][qword_indices]), Atranspose[aword_idx][qword_indices]) for
                  aword_idx, qword_indices in enumerate(aq_window)]
    ahatmatrix = np.array([np.sum(weights.reshape(-1, 1) * qmatrix[qword_indices], axis=0) / weight_sum for
                           ((aword_idx, qword_indices), (weight_sum, weights)) in
                           zip(enumerate(aq_window), aq_weights)])
    return qhatmatrix, ahatmatrix

def f_decompose(matrix, hatmatrix):
    # finding magnitude of parallel vector
    mag = np.sum(hatmatrix * matrix, axis=1) / np.sum(hatmatrix * hatmatrix, axis=1)
    # multiplying magnitude with hatmatrix vector
    plus = mag.reshape(-1, 1) * hatmatrix
    minus = matrix - plus
    return plus, minus

def get_wang_conv_model_input(samples_sent_matrix, max_qs_l, max_ans_l):
    token = np.zeros((2, vec_dim), dtype='float')
    qsamples = samples_sent_matrix[0]
    asamples = samples_sent_matrix[1]
    q_list = []
    a_list = []
    for qmatrix, amatrix in zip(qsamples, asamples):
        qplus, qminus, aplus, aminus = compose_decompose(qmatrix, amatrix)
        # Padding questions
        qpad_width = ((0, max_qs_l - qplus.shape[0]), (0, 0))
        qplus_pad = np.pad(qplus, pad_width=qpad_width, mode='constant', constant_values=0.0)
        qminus_pad = np.pad(qminus, pad_width=qpad_width, mode='constant', constant_values=0.0)
        # Padding answers
        apad_width = ((0, max_ans_l - aplus.shape[0]), (0, 0))
        aplus_pad = np.pad(aplus, pad_width=apad_width, mode='constant', constant_values=0.0)
        aminus_pad = np.pad(aminus, pad_width=apad_width, mode='constant', constant_values=0.0)
        qplusminus=np.concatenate((qplus_pad, token, qminus_pad))
        aplusminus = np.concatenate((aplus_pad, token, aminus_pad))
        # Adding these padded matrices to list
        q_list.append(qplusminus)
        a_list.append(aplusminus)
    q_tensor = np.array(q_list)
    a_tensor = np.array(a_list)
    return q_tensor, a_tensor

def get_wang_model_input(samples, max_qs_l, max_ans_l):
    """
    Returns the training samples and labels as numpy array
    """
    s1samples_list = []
    s2samples_list = []
    labels_list = []

    for sample in samples:
        q_len=len(sample.QsWords)
        if q_len > max_qs_l:
            q_len=max_qs_l
        a_len=len(sample.AnsWords)
        if a_len > max_ans_l:
            a_len=max_ans_l
        s1samples_list.append(get_sent_matrix(sample.QsWords[0:q_len]))
        s2samples_list.append(get_sent_matrix(sample.AnsWords[0:a_len]))
        labels_list.append(sample.Label)

    samples_sent_matrix = [s1samples_list, s2samples_list]
    labels = labels_list
    return samples_sent_matrix, labels

def get_sent_matrix(words):
    """
    Given a sentence, gets the input in the required format.
    """
    vecs = []
    vec = np.zeros(vec_dim, dtype='float32')
    for word in words:
        if word_vecs.has_key(word):
            vec = word_vecs[word]
        else:
            vec = word_vecs["<unk>"]
        vecs.append(np.array(vec))
    return np.array(vecs)

def run_wang_cnn_model(train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines):
    max_qs_l, max_ans_l = 23,40

    if max_ans_l > ans_len_cut_off:
        max_ans_l = ans_len_cut_off
    train_samples_sent_matrix, train_labels = get_wang_model_input(train_samples, max_qs_l, max_ans_l)
    train_labels_np=np.array(train_labels)
    train_q_tensor, train_a_tensor = get_wang_conv_model_input(train_samples_sent_matrix, max_qs_l, max_ans_l)
    max_qs_l = 2 * max_qs_l + 2
    max_ans_l = 2 * max_ans_l + 2
    Reduce = Lambda(lambda x: x[:, 0, :], output_shape=lambda shape: (shape[0], shape[-1]))

    nb_filter = 500

    qs_input = Input(shape=(max_qs_l, vec_dim,), dtype='float32', name='qs_input')
    qs_convmodel_3 = Convolution1D(nb_filter=nb_filter, filter_length=3, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_3 = MaxPooling1D(pool_length=max_qs_l - 2)(qs_convmodel_3)
    qs_convmodel_3 = Reduce(qs_convmodel_3)
    qs_convmodel_2 = Convolution1D(nb_filter=nb_filter, filter_length=2, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_2 = MaxPooling1D(pool_length=max_qs_l - 1)(qs_convmodel_2)
    qs_convmodel_2 = Reduce(qs_convmodel_2)
    qs_convmodel_1 = Convolution1D(nb_filter=nb_filter, filter_length=1, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_1 = MaxPooling1D(pool_length=max_qs_l)(qs_convmodel_1)
    qs_convmodel_1 = Reduce(qs_convmodel_1)
    qs_concat = merge([qs_convmodel_1, qs_convmodel_2, qs_convmodel_3], mode='concat', concat_axis=-1)

    ans_input = Input(shape=(max_ans_l, vec_dim,), dtype='float32', name='ans_input')
    ans_convmodel_3 = Convolution1D(nb_filter=nb_filter, filter_length=3, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_3 = MaxPooling1D(pool_length=max_ans_l - 2)(ans_convmodel_3)
    ans_convmodel_3 = Reduce(ans_convmodel_3)
    ans_convmodel_2 = Convolution1D(nb_filter=nb_filter, filter_length=2, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_2 = MaxPooling1D(pool_length=max_ans_l - 1)(ans_convmodel_2)
    ans_convmodel_2 = Reduce(ans_convmodel_2)
    ans_convmodel_1 = Convolution1D(nb_filter=nb_filter, filter_length=1, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_1 = MaxPooling1D(pool_length=max_ans_l)(ans_convmodel_1)
    ans_convmodel_1 = Reduce(ans_convmodel_1)
    ans_concat = merge([ans_convmodel_1, ans_convmodel_2, ans_convmodel_3], mode='concat', concat_axis=-1)

    q_a_model=merge([qs_concat, ans_concat], mode='concat', concat_axis=-1)
    sim_model = Dense(output_dim=1, activation = 'linear')(q_a_model)
    labels = Activation('sigmoid', name='labels')(sim_model)

    wang_model = Model(input=[qs_input, ans_input], output=[labels])

    wang_model.compile(loss={'labels': 'binary_crossentropy'},
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

    batch_size = 10
    epoch = 25

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoint-{epoch:02d}-{acc:.2f}.hdf5', monitor='acc',
                                                 save_best_only=True, save_weights_only=False, mode='auto', verbose=2)

    wang_model.fit({'qs_input': train_q_tensor, 'ans_input': train_a_tensor}, {'labels': train_labels_np}, nb_epoch=epoch,
                   batch_size=batch_size, verbose=2, validation_split=0.30, shuffle='True', callbacks=[checkpoint])

if __name__=="__main__":

    model_name = sys.argv[1]
    data_folder = os.path.join("data")
    word_vec_file = os.path.join(data_folder, sys.argv[2])
    stop_words_file = os.path.join(data_folder, sys.argv[3])
    train_file = os.path.join(data_folder, sys.argv[4])
    dev_file = os.path.join(data_folder, sys.argv[5])
    dev_ref_file = os.path.join(data_folder, sys.argv[6])
    test_file = os.path.join(data_folder, sys.argv[7])
    test_ref_file = os.path.join(data_folder, sys.argv[8])

    QASample=namedtuple("QASample","Qs Ans QsWords AnsWords Label")

    word_vecs=load_word2vec(word_vec_file)
    stop_words=load_stop_words(stop_file=stop_words_file)

    files=[]
    files.append(train_file)
    files.append(dev_file)
    files.append(test_file)

    train_samples = load_samples(train_file)
    dev_samples=load_samples(dev_file)
    test_samples = load_samples(test_file)

    file_reader = open(dev_ref_file)
    dev_ref_lines = file_reader.readlines()
    file_reader.close()

    file_reader=open(test_ref_file)
    test_ref_lines=file_reader.readlines()
    file_reader.close()


    if model_name=="DecompCompCNN":
        #Decomposition and Composition based CNN model
        print "Decomposition and Composition based CNN model started......"
        run_wang_cnn_model(train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines)


















































