from dataProcessing import read_file, model_builder
import numpy as np
import tensorflow as tf
import sys
from time import time

### load parameters from console
## model parameter
batch_size = 1000
LAYERS = 1
hidden_rapr_Bi = 60

# Data summary
max_length_sentences = 144
dim_feature = 2

# place holders dimentions
dimension_embeddings = 100
dimension_POS = 48

# learning rate
lr = 0.01

for el in sys.argv[1:]:
    com, val = el.split("=")
    if com == "batch":
        batch_size = int(val)
        print("batch size is: ", batch_size)
    elif com == "layers":
        LAYERS = int(val)
        print("layers are: ", LAYERS)
    elif com == "hidden_Bi_lstm":
        hidden_rapr_Bi = int(val)
        print("Hidden for Bi-lstm is: ", hidden_rapr_Bi)
    elif com == "max_length_sentences":
        max_length_sentences = int(val)
        print("max len for seq in the train is: ", max_length_sentences)
    elif com=="lr":
        lr = int(val)
        print("the learning rate is: ", lr)

        
file = "../data/CoNLL2009-ST-English-train.txt"
reader = read_file(file)


pred_to_fit, pred_not = reader.find_all_predicates(occ_greater_than=50)  # I fit on my model just the predicate seen more than 50 times
print("total numeber of fittable predicates are: ", len(pred_to_fit))
pos, pos_pos = reader.find_all_POS() # ritorna la lista di possibili POS e il contatore di quanti predicati si trovano in quel POS
print("possible POS have length: ", len(pos_pos))
max_len_in_data = reader.max_length_sentence()
print("maximum length of sentence not cutted is ", max_len_in_data)


### il modello serve a creare i dati input per il DNN
model = model_builder(pos_pos)
list_of_words = model.load_model("../data/glove.6B.100d.txt")
#model("the")

#X_emb, sent_len, pred, X_POS = model.creation(sents, max_length= 50)





tf.reset_default_graph()
g1 = tf.Graph()
with g1.as_default():
    ## Inputs
    one_hot_POS = tf.placeholder(tf.float32, shape = (batch_size, max_length_sentences, dimension_POS), name= "POS")
    embeddings = tf.placeholder(tf.float32, shape=(batch_size, max_length_sentences , dimension_embeddings), name="embeddings")
    sequence_lengths = tf.placeholder(tf.int32, shape = [batch_size])
    
    ## randomized embeddings for POS
    W_emb_pos = tf.get_variable("W_emb_pos", shape = [dimension_POS , POS_emb_dim], dtype = tf.float32)
    POS = tf.reshape(one_hot_POS, shape = [-1, dimension_POS ])
    POS_emb_flat = POS @ W_emb_pos    
    POS_emb = tf.reshape(POS_emb_flat, shape = [ batch_size , max_length_sentences, POS_emb_dim])
    
    # inputs to the Bi-lstm
    inputs = tf.concat([embeddings, POS_emb], axis = -1)

    inputs = tf.concat([embeddings, one_hot_POS], axis = -1)

    ## Bi-Lstm for cuda
    bi_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = LAYERS, num_units = hidden_rapr_Bi,
                                             direction = 'bidirectional',
                                             dropout = 0.2,
                                             seed = 123, dtype = tf.float32, name="Bi-Lstm")

    outputs_Bi_Lstm, output_states = bi_lstm(inputs)

    context_flat = tf.reshape(outputs_Bi_Lstm, shape = [-1,2*hidden_rapr_Bi ])


    ## Last layer weigths
    W_out = tf.get_variable("W_out", shape = [2*hidden_rapr_Bi , dim_feature], dtype = tf.float32)
    b_out = tf.get_variable("b_out", shape = [dim_feature], dtype = tf.float32, initializer=tf.zeros_initializer())


    pred_flat = tf.matmul(context_flat, W_out) + b_out
    pred = tf.reshape(pred_flat, shape= [-1, max_length_sentences, dim_feature])


    ###### Losses and Optimizer
    labels = tf.placeholder(tf.int32, shape=[batch_size, max_length_sentences], name= "labels")

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels=labels)

    #mask
    mask = tf.sequence_mask(sequence_lengths, max_length_sentences)
    losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses)

    ### Optimizer
    
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)

    ### labels pred
    labels_pred = tf.cast(tf.argmax(pred, axis=-1), dtype= tf.int32, name="pred_labels")

    
start = time()
sents = reader.read_sentences(batch_size)
X_emb, sent_len, pred, X_POS = model.creation(sents, max_length= max_length_sentences)


## fare qualcosa che randomizza il sampling
feed_dict={one_hot_POS: X_POS[:batch_size],
           embeddings: X_emb[:batch_size],
           sequence_lengths:sent_len[:batch_size],
           labels: pred[:batch_size]
          }

print("start to train #########\n")
with g1.as_default():
    for ite in range(100):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            s.run(tf.initializers.global_variables())
            l , _ = s.run([loss, train_op], feed_dict=feed_dict)
            if ite % 100==1:
                print(l)
                print("exec: ", time()-start)
            
print("exec: ", time()-start)