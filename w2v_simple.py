
# --- IMPORT DEPENDENCIES ------------------------------------------------------+
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import combinations

# --- CONSTANTS ----------------------------------------------------------------+

class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window']
        self.v_count = settings['vocab_count']
        pass

    # GENERATE TRAINING DATA
    def generate_training_data(self, imgs):
        # load image vectors from text file

        training_data = []
        # CYCLE through the vector, considering each as a target and all other images as it's context
        for idx1, w_t in enumerate(imgs):
            w_target = w_t.tolist()
            context = (combinations(np.delete(imgs, 0, axis=0), 2))

            for w_c in context :
                w_context = []
                for j in w_c:
                    w_context.append(j.tolist())

                training_data.append([w_target, w_context])


        return training_data

    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass

    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))  # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))  # context matrix

        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:
                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)

                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                #self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                # self.loss += -np.sum([u[label == 1] for label in np.sum(context, axis = 1)]) + np.sum(context) * np.log(np.sum(np.exp(u)))

            print('EPOCH:', i )#, 'LOSS:', self.loss)
        pass

    # input an image vector, returns a embedding vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    #print all the words and their embeddings
    def print_vec(self):
        for word in self.word_index.keys():
            w_index = self.word_index[word]
            v_w = self.w1[w_index]
            print(word, v_w)

    #plot the embeddings in a 2 dimensional space using TSNE.
    def plot_vec(self):
        labels = []
        tokens = []
        for w in self.word_index.keys():
            labels.append(w)
            w_index = self.word_index[w]
            tokens.append(self.w1[w_index])

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()


# --- EXAMPLE RUN --------------------------------------------------------------+

settings = {}
settings['n'] = 64 # dimension of word embeddings
settings['epochs'] = 300  # number of training epochs
settings['learning_rate'] = 0.01  # learning rate
settings['window'] = 3
settings['vocab_count'] = 2048
np.random.seed(0)  # set the seed for reproducibility


# INITIALIZE W2V MODEL
w2v = word2vec()

vocab = {}
#process images
inputfolder = 'ImgVecstest'

assert os.path.isdir(inputfolder), "Specified Image Vectors Folder does not exist"

subfolders = os.listdir(inputfolder)

training_data = []

for idx, subfolder in enumerate(subfolders):
    assert os.path.isfile(inputfolder + "/" + subfolder + "/Names.npy"), "Files for subfolder " + subfolder + " incomplete"
    assert os.path.isfile(inputfolder + "/" + subfolder + "/ImgVectors.npy"), "Files for subfolder " + subfolder + " incomplete"

    names = np.load(inputfolder + "/" + subfolder + "/Names.npy")

    if len(names) <  settings['window']  + 1:
        continue

    vectors = np.load(inputfolder + "/" + subfolder + "/ImgVectors.npy")

    vocab.update(dict(zip(names,vectors)))

    training_data.extend(w2v.generate_training_data(vectors))

training_data = np.array(training_data)

# train word2vec model
w2v.train(training_data)

# -------- END ----------------------------------------------------------------------+