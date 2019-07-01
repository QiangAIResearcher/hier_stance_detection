from score import *
from util_by_model import *
import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# Prompt for mode
mode = input('mode (load / train)? ')


# Set file names
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "competition_test_stances.csv"
file_test_bodies = "competition_test_bodies.csv"
file_predictions = 'predictions_test.csv'


# Initialise hyperparameters
r = random.Random()
lim_unigram = 3000
hidden_size = 100
mmd_size = 10
train_keep_prob = 1
clip_ratio = 5
batch_size_train = 200
epochs = 10000
relatedness_size = 2
agreement_size = 3
classes_size = 4

alpha = 1.5
beta = 1e-2
l2_alpha = 1e-5
learn_rate = 1e-5
early_epoch = 300

# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)


# Process data sets
train_set,train_mean,train_stance,train_stance_idx,\
train_relatedness,train_relatedness_false,\
bow_vectorizer,tfreq_vectorizer,tfidf_vectorizer,mmd_symbol,mmd_symbol_ = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)

test_set,test_stance,test_stance_idx,\
test_relatedness,test_relatedness_false,\
test_mmd_sym, test_mmd_sym_ = \
    pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, train_mean)

n_train = len(train_set)
input_size = len(train_set[0])


# Define model
# Create placeholders
with tf.name_scope('inputs'):
    input_pl = tf.placeholder(tf.float32, [None, input_size], name='unigram_features')
    relatedness_true_pl = tf.placeholder(tf.int32, shape=[None,relatedness_size], name='true_relatedness')
    stance_true_pl = tf.placeholder(tf.int32, [None,classes_size], name='true_stances')
    mmd_symbol_pl = tf.placeholder(tf.float32,[None],name='mmd_symbol')
    mmd_symbol_pl_ = tf.placeholder(tf.float32, [None], name='mmd_symbol_')
    keep_prob_pl = tf.placeholder(tf.float32,name='keep_prob')

# Define multi-layer perceptron
with tf.name_scope('layers'):
    with tf.name_scope('hidden_layer'):
        wx_b = tf.contrib.layers.linear(input_pl, hidden_size)
        tf.summary.histogram('wx_b', wx_b)

        # batch normalization
        wx_b_mean,wx_b_var = tf.nn.moments(wx_b,axes=[0])
        scale = tf.Variable(tf.ones([hidden_size]))
        shift = tf.Variable(tf.zeros([hidden_size]))
        epilson = 0.001
        wx_b_normalization = tf.nn.batch_normalization(wx_b,wx_b_mean,wx_b_var,shift,scale,epilson)
        hidden_layer = tf.nn.dropout(tf.nn.relu(wx_b_normalization), keep_prob=keep_prob_pl)

    with tf.name_scope('mmd_layer'):
        theta_d = tf.contrib.layers.linear(hidden_layer, mmd_size)

        # batch normalization
        theta_d_mean,theta_d_var = tf.nn.moments(theta_d,axes=[0])
        scale_d = tf.Variable(tf.ones([mmd_size]))
        shift_d = tf.Variable(tf.zeros([mmd_size]))
        epilson_d = 0.001
        theta_d_normalization = tf.nn.batch_normalization(theta_d,theta_d_mean,theta_d_var,shift_d,scale_d,epilson_d)

        theta_d_layer = tf.nn.dropout(tf.nn.relu(theta_d_normalization), keep_prob=keep_prob_pl)#
        n1 = tf.reduce_sum(mmd_symbol_pl,  axis = 0)
        n2 = tf.reduce_sum(mmd_symbol_pl_, axis = 0)
        aa = tf.reshape(mmd_symbol_pl,[-1,1])
        bb = tf.reshape(mmd_symbol_pl_,[-1,1])
        d1 = tf.divide(tf.reduce_sum(tf.multiply(theta_d_layer, aa),axis=1), n1)
        d2 = tf.divide(tf.reduce_sum(tf.multiply(theta_d_layer, bb),axis=1),n2)

        mmd_loss = d1 - d2

    with tf.name_scope('probability_layer'):
        relatedness_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, relatedness_size),
                                         keep_prob=keep_prob_pl)
        P_relatedness = tf.nn.softmax(tf.reshape(relatedness_flat, [-1, relatedness_size]),axis=1)
        P_related = tf.reshape(P_relatedness[:, 0], [-1, 1], name='related')
        P_unrelated = tf.reshape(P_relatedness[:, 1], [-1, 1], name='unrelated')

        concat_fea = tf.concat([hidden_layer,P_related],axis=1)
        stance_flat = tf.nn.dropout(tf.contrib.layers.linear(concat_fea, classes_size),
                                       keep_prob=keep_prob_pl)
        P_stance = tf.nn.softmax(tf.reshape(stance_flat, [-1, classes_size]),axis=1)

# Define overall loss
with tf.name_scope('loss'):
    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    relatedness_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=relatedness_true_pl,logits=P_relatedness)
    stance_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=stance_true_pl, logits=P_stance)

    rank_loss = relatedness_loss + alpha * stance_loss + beta * mmd_loss
    loss = tf.reduce_sum(rank_loss+ l2_loss)
tf.summary.scalar('loss', loss)

# Define prediction
with tf.name_scope('prediction'):
    tmp1 = P_stance[:,:3]
    tmp2 = tf.reshape(tf.reduce_sum(tmp1,axis=1),[-1,1])
    tmp3 = tf.concat([tmp2,tmp2,tmp2],axis=1)
    tmp4 = tf.concat([P_related, P_related, P_related],axis=1)
    tmp5 = tf.divide(tmp1,tmp3)
    tmp6 = tf.multiply(tmp5,tmp4)
    prob = tf.concat([tmp6,P_unrelated],1)#tmp6

    predict = tf.argmax(prob, 1)
tf.summary.histogram('prediction', predict)

if mode == 'train':
    # Train model
    early_stop = True
    best_loss = 100000000
    best_epoch = 0
    stopping_step = 0

    val_loss_list = []

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/',sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        train_loss_list = []
        for epoch in range(epochs):
            print(str(epoch)+'/'+str(epochs))
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_input = [train_set[i] for i in batch_indices]
                batch_relatedness = [train_relatedness[i] for i in batch_indices]
                # batch_relatedness_false = [train_relatedness_false[i] for i in batch_indices]
                batch_stance = [train_stance[i] for i in batch_indices]
                batch_mmd_symbol = [mmd_symbol[i] for i in batch_indices]
                batch_mmd_symbol_ = [mmd_symbol_[i] for i in batch_indices]
                batch_feed_dict = {input_pl:batch_input, \
                                   relatedness_true_pl:batch_relatedness,\
                                   # relatedness_false_pl:batch_relatedness_false,\
                                   stance_true_pl: batch_stance, \
                                   mmd_symbol_pl: batch_mmd_symbol, \
                                   mmd_symbol_pl_: batch_mmd_symbol_, \
                                   keep_prob_pl: train_keep_prob}

                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss

                results = sess.run(merged,feed_dict=batch_feed_dict)
                writer.add_summary(results,i+epoch*(n_train // batch_size_train))

            train_loss_list.append(total_loss)

            # early stopping
            if epoch>early_epoch:
                print("Early stopping is trigger.")
                break

        # Predict on training data
        train_feed_dict = {input_pl: train_set, keep_prob_pl: 1.0}
        train_pred = sess.run(predict, feed_dict=train_feed_dict)

        # Predict on test data
        test_feed_dict = {input_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

# Load model
if mode == 'load':

    print("\nloading model...")
    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, './model/best-hier-mmd-2')

        # Predict on training data
        train_feed_dict = {input_pl: train_set, keep_prob_pl: 1.0}
        train_pred = sess.run(predict, feed_dict=train_feed_dict)

        #Predict on test data
        test_feed_dict = {input_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

report_score([LABELS[e] for e in train_stance_idx], [LABELS[e] for e in train_pred])
report_score([LABELS[e] for e in test_stance_idx], [LABELS[e] for e in test_pred])
# Save predictions
#save_predictions(test_pred, file_predictions)
