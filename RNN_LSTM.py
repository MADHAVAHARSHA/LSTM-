
from collections import Counter
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
import matplotlib as mplt
mplt.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')



def pre_process():
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    words = []
    temp_post_text = []
    print(len(newsgroups_data.data))

    for post in newsgroups_data.data:

        all_text = ''.join([text for text in post if text not in punctuation])
        all_text = all_text.split('\n')
        all_text = ''.join(all_text)
        temp_text = all_text.split(" ")

        for word in temp_text:
            if word.isalpha():
                temp_text[temp_text.index(word)] = word.lower()

        # temp_text = [word for word in temp_text if word not in stopwords.words('english')]
        temp_text = list(filter(None, temp_text))
        temp_text = ' '.join([i for i in temp_text if not i.isdigit()])
        words += temp_text.split(" ")
        temp_post_text.append(temp_text)

    # temp_post_text = list(filter(None, temp_post_text))

    dictionary = Counter(words)
    # deleting spaces
    # del dictionary[""]
    sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
    vocab_to_int = {c: i for i, c in enumerate(sorted_split_words,1)}

    message_ints = []
    for message in temp_post_text:
        temp_message = message.split(" ")
        message_ints.append([vocab_to_int[i] for i in temp_message])


    # maximum message length = 6577

    # message_lens = Counter([len(x) for x in message_ints])AAA

    seq_length = 6577
    num_messages = len(temp_post_text)
    features = np.zeros([num_messages, seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        # print(features[i, -len(row):])
        # features[i, -len(row):] = np.array(row)[:seq_length]
        features[i, :len(row)] = np.array(row)[:seq_length]
        # print(features[i])

    lb = LabelBinarizer()
    lbl = newsgroups_data.target
    labels = np.reshape(lbl, [-1])
    labels = lb.fit_transform(labels)

    sequence_lengths = [len(msg) for msg in message_ints]
    return features, labels, len(sorted_split_words)+1, sequence_lengths


def get_batches(x, y, sql, batch_size=1):
    for ii in range(0, len(y), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size], sql[ii:ii+batch_size]


def plot(noOfWrongPred, dataPoints):
    font_size = 14
    fig = plt.figure(dpi=100,figsize=(10, 6))
    mplt.rcParams.update({'font.size': font_size})
    plt.title("Distribution of wrong predictions", fontsize=font_size)
    plt.ylabel('Error rate', fontsize=font_size)
    plt.xlabel('Number of data points', fontsize=font_size)

    plt.plot(dataPoints, noOfWrongPred, label='Prediction', color='blue', linewidth=1.8)
    # plt.legend(loc='upper right', fontsize=14)

    plt.savefig('distribution of wrong predictions.png')
    # plt.show()



def train_test():
    features, labels, n_words, sequence_length = pre_process()

    print(features.shape)
    print(labels.shape)

    # Defining Hyperparameters

    lstm_layers = 1
    batch_size = 1
    lstm_size = 200
    learning_rate = 0.01

    # --------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
        # labels_ = tf.placeholder(dtype= tf.int32)
        labels_ = tf.placeholder(tf.float32, [None, None], name="labels")
        sql_in = tf.placeholder(tf.int32, [None], name= 'sql_in')

        # output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size = 300

        # generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1),trainable=True)
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        print(embedding.shape)
        print(embed.shape)
        print(embed[0])

        # Your basic LSTM cell
        lstm =  tf.contrib.rnn.BasicLSTMCell(lstm_size)


        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state, sequence_length=sql_in)

        # hidden layer
        hidden = tf.layers.dense(outputs[:, -1], units=25, activation=tf.nn.relu)

        print(hidden.shape)

        logit = tf.contrib.layers.fully_connected(hidden, num_outputs=20, activation_fn=None)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        saver = tf.train.Saver()

    # ----------------------------online training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        state = sess.run(initial_state)
        wrongPred = 0
        noOfWrongPreds = []
        dataPoints = []

        for ii, (x, y, sql) in enumerate(get_batches(features, labels, sequence_length, batch_size), 1):

            feed = {inputs_: x,
                    labels_: y,
                    sql_in : sql,
                    keep_prob: 0.5,
                    initial_state: state}

            predictions = tf.nn.softmax(logit).eval(feed_dict=feed)

            print("----------------------------------------------------------")
            print("sez: ",sql)
            print("Iteration: {}".format(iteration))

            isequal = np.equal(np.argmax(predictions[0], 0), np.argmax(y[0], 0))

            print(np.argmax(predictions[0], 0))
            print(np.argmax(y[0], 0))

            if not (isequal):
                wrongPred += 1

            print("nummber of wrong preds: ",wrongPred)

            if iteration%50 == 0:
                noOfWrongPreds.append(wrongPred/iteration)
                dataPoints.append(iteration)

            loss, states, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            print("Train loss: {:.3f}".format(loss))
            iteration += 1

        saver.save(sess, "checkpoints/sentiment.ckpt")
        errorRate = wrongPred / len(labels)
        print("ERRORS: ", wrongPred)
        print("ERROR RATE: ", errorRate)
        plot(noOfWrongPreds, dataPoints)


if __name__ == '__main__':
    train_test()
