import random
random.seed(123456789)
import numpy as np
np.random.seed(123456789)
import subprocess
import shlex
import sys
import math
from optparse import OptionParser
from collections import defaultdict
import pprint
import pdb

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K, optimizers

from utils import invert_dict, unsplit_query, merge_two_dicts
from data_preprocess import gen_data, load_data, save_data, construct_vocab_emb, sample_val_set
from default_args import get_best_args
from attention_model import create_attention_model, add_embed_layer

if K.backend() == "tensorflow":
    import tensorflow as tf
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) #, device_count={"GPU": 0}
    tf.set_random_seed(1234)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
else:
    from theano.tensor.shared_randomstreams import RandomStreams
    # from theano import function
    srng = RandomStreams(seed=123456789)
    srng.seed(123456789)  # seeds rv_u and rv_n with different seeds each


def evaluate(predictions_file, qrels_file):
    pargs = shlex.split("/bin/sh run_eval.sh '{}' '{}'".format(qrels_file, predictions_file))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()

    if sys.version_info[0] < 3:
        lines = pout.split(b'\n')
    else:
        lines = pout.split(b'\n')
    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    p30 = float(lines[2].strip().split()[-1])
    return map, mrr, p30


def print_dataset(mode, dataset, vocab_inv):
    for key in dataset:
        print(key, dataset[key].shape)
        if mode == 'deep_twitter':
            if "weight" in key:
                # try print this info for debugging
                # print(key, dataset[key][0][:2])
                None
            elif "mask" in key:
                # try print this info for debugging
                # print(key, dataset[key][0])
                None
            elif "word" in key:
                print(dataset[key][0])
                print(key, unsplit_query(dataset[key][0], "word", vocab_inv["word"]))
            elif "url" in key:
                print(key, unsplit_query(dataset[key][0], "3gram", vocab_inv["url"]))
            elif "3gram" in key:
                print(key, unsplit_query(dataset[key][0], "3gram", vocab_inv["3gram"]))
            elif "feat" in key:
                print(key, dataset[key][0:5])


def get_model_weights(model):
    model_weights = 0.0
    for layer in model.layers:
        for weights in layer.get_weights():
            model_weights += np.sum(weights)
    return model_weights


def batch_generator(dataset, batch_size):
    while True:
        num_batches = int(math.ceil(dataset['sim'].shape[0] * 1.0 / batch_size))
        for i in range(num_batches):
            start_idx, end_idx = i*batch_size, min((i+1)*batch_size, dataset['sim'].shape[0])
            x, y = {}, dataset['sim'][start_idx:end_idx]
            for key in dataset:
                x[key] = dataset[key][start_idx:end_idx]
            yield (x, y)


def get_default_args():
    return {
        "load_data": False, # load previously created data to save the time for data conversion
        "load_model": False, # load previously trained model for testing
        "trainable": True, # whether to train the word embedding or not
        "nb_filters": 256, # number of filters in the CNN model
        "dropout": 0.1,
        "batch_size": 64,
        "mode": 'deep_twitter',
        'weighting': 'query', # "query" or "doc":
        # "query" -> only weight query words and ngrams,
        # "doc" -> weight both query and document words/ngrams
        # whether to mask padded word or not -- this param seems have a big impact on model performance
        "mask": False,
        "val_split": 0.15,
        # "word_url" -> query-doc (word) + query-url; "complete" -> "query-doc (word+ngram) + query-url"
        "model_option": "complete",
        # "normal" -> conv connection layer by layer; can also be "ResNet";
        "conv_option": "normal",
        # path to save intermediate outputs, i.e., logs, models, etc.
        "experimental_data": "./experimental",
        "raw_data": "data/twitter-v0",
        "qrels_file": "data/twitter-v0/qrels.microblog2011-2014.txt",
        "base_embed_path": "data/word2vec/GoogleNews-vectors-negative300.bin.gz",
        # "base_embed_path": "../data/word2vec/tweet_vector_0401.bin",
        "split": {"train": "train_all", "test": "trec-2011"},
        "external_feat": False,
        "norm_weight": False,
        "cos": False,
        "epochs": 15,
        "optimizer": "sgd",
        "learning_rate": 0.05,
        "verbose": 1 # 0 for silent, 1 for interactive, 2 for only showing logs per epoch
    }


def print_args(args):
    print('------------------------------------------------------------')
    pprint.pprint(args)
    print('------------------------------------------------------------')


def load_best_args(args, options, best_args):
    test_set = options.test if options.test else args['test']
    print('load default args for test_set %s' % test_set)
    for param in best_args[test_set]:
        args[param] = best_args[test_set][param]


def set_args(args, options):
    print(type(options))
    for arg in dir(options):
        if arg in args and getattr(options, arg) is not None:
            args[arg] = getattr(options, arg)
    args['split']['test'] = options.test


def create_option_parser():
    parser = OptionParser()
    parser.add_option("-n", "--nb_filters", action="store", type=int, dest="nb_filters")
    parser.add_option("-d", "--dropout", action="store", type=float, dest="dropout")
    parser.add_option("-b", "--batch_size", action="store", type=int, dest="batch_size")
    parser.add_option("-w", "--weighting", action="store", type=str, dest="weighting")
    parser.add_option("-m", "--mask", action="store_true", dest="mask")
    parser.add_option("-t", "--test", action="store", type="string", dest="test")
    parser.add_option("-c", "--conv_option", action="store", type="string", dest="conv_option")
    parser.add_option("-e", "--external_feat", action="store_true", dest="external_feat")
    parser.add_option("-l", "--load_data", action="store_true", dest="load_data")
    parser.add_option("-o", "--optimizer", action="store", type=str, dest="optimizer")
    parser.add_option("-v", "--verbose", action="store", type=int, dest="verbose")
    parser.add_option("--norm", "--norm_weight", action="store_true", dest="norm_weight")
    parser.add_option("--mode", "--mode", action="store", type=str, dest="mode")
    parser.add_option("--cos", "--cos", action="store_true", dest="cos")
    parser.add_option("--epochs", "--epochs", action="store", type=int, dest="epochs")
    parser.add_option("--trainable", "--trainable", action="store_true", dest="trainable")
    parser.add_option("--model_option", "--model_option", action="store", dest="model_option")
    parser.add_option("--lr", "--learning_rate", action="store", type=float, dest="learning_rate")
    parser.add_option("--load_model", "--load_model", action="store_true", dest="load_model")
    parser.add_option("--val_split", "--val_split", action="store", type=float, dest="val_split")
    return parser


def main(options):
    args = get_default_args()
    load_best_args(args, options, get_best_args())
    set_args(args, options)
    print_args(args)
    mode = args['mode']
    train_name, test_name = args['split']['train'], args['split']['test']
    if train_name == 'train_all':
        train_set = ['trec-2011', 'trec-2012', 'trec-2013', 'trec-2014']
        train_set.remove(test_name)
    else:
        train_set = [train_name]
    test_set = test_name
    print('train_set: {}, test_set: {}'.format(train_set, test_set))
    max_query_len, max_doc_len, max_url_len = defaultdict(int), defaultdict(int), defaultdict(int)
    vocab = {'word': {}, '3gram': {}, 'url': {}}
    test_vocab = {'word': {}, '3gram': {}, 'url': {}}

    ############################# LOAD DATA ##################################
    data_name = ("data_m%s_%s_%s" % (mode, train_name, test_name)).lower()
    if args["load_data"]:
        train_dataset, vocab, train_vocab_emb, max_query_len, max_doc_len, max_url_len = load_data(
            "%s/%s/%s" % (args["experimental_data"], data_name, train_name), True)
        test_dataset, test_vocab, test_vocab_emb, _, _, _ = load_data(
            "%s/%s/%s" % (args["experimental_data"], data_name, test_name), False)
        print('load dataset successfully')
    else:
        train_dataset = gen_data(args["raw_data"], train_set, vocab, test_vocab, True, max_query_len, max_doc_len,
                                 max_url_len, args)
        print("create training set successfully...")
        test_dataset = gen_data(args["raw_data"], [test_set], vocab, test_vocab, False, max_query_len, max_doc_len,
                                max_url_len, args)
        train_vocab_emb, test_vocab_emb = construct_vocab_emb(
            "%s/%s" % (args["experimental_data"], data_name), vocab['word'], test_vocab['word'], 300, "word",
            base_embed_path=args["base_embed_path"])
        save_data("%s/%s/%s" % (args["experimental_data"], data_name, train_name), True, train_dataset, max_query_len,
                  max_doc_len, max_url_len, vocab, train_vocab_emb)
        print("save training set successfully...")
        save_data("%s/%s/%s" % (args["experimental_data"], data_name, test_name), False, test_dataset,
                  vocab=test_vocab, vocab_emb=test_vocab_emb)
        print("save test set successfully...")

    val_split = args['val_split']
    num_samples, _ = train_dataset["query_word_input"].shape
    # randomly sample queries and all their documents if query_random is True
    # otherwise, query-doc pairs are randomly sampled
    query_random = True
    if query_random:
        val_indices = sample_val_set(args["raw_data"], train_set, val_split)
    else:
        val_indices, val_set = [], set()
        for i in range(int(num_samples * val_split)):
            val_index = np.random.randint(num_samples)
            while val_index in val_set:
                val_index = np.random.randint(num_samples)
            val_indices.append(val_index)
            val_set.add(val_index)

    val_dataset = {}
    for key in train_dataset:
        val_dataset[key] = train_dataset[key][val_indices]
        train_dataset[key] = np.delete(train_dataset[key], val_indices, 0)

    # shuffle the train dataset explicitly to make results reproducible
    # whether the performance will be affected remains a question
    keys, values = [], []
    for key in train_dataset:
        keys.append(key)
        values.append(train_dataset[key])
    zipped_values = list(zip(*values))
    random.shuffle(zipped_values)
    shuffled_values = list(zip(*zipped_values))
    for i, key in enumerate(keys):
        train_dataset[key] = np.array(shuffled_values[i])
    print('after shuffle: id {}, sim {}, query_word_input'.format(train_dataset['id'][:3], train_dataset['sim'][:3],
          train_dataset['query_word_input'][:3]))

    # merge the vocabulory of train and test set
    merged_vocab = {'url': vocab['url'], '3gram': vocab['3gram']}
    merged_vocab['word'] = merge_two_dicts(vocab['word'], test_vocab['word'])
    print("merged vocab: word(%d) 3gram(%d)" % (len(merged_vocab['word']), len(test_vocab['3gram'])))
    vocab_inv, vocab_size = {}, {}
    vocab['char'] = merge_two_dicts(vocab['3gram'], vocab['url'])
    test_vocab['char'] = merge_two_dicts(test_vocab['3gram'], test_vocab['url'])
    merged_vocab['char'] = merge_two_dicts(vocab['char'], test_vocab['char'])

    for key in vocab:
        vocab_inv[key] = invert_dict(merged_vocab[key])
        vocab_size[key] = len(vocab[key])
    print(vocab_size)

    # Print data samples for debug purpose
    print_dataset(mode, train_dataset, vocab_inv)
    print_dataset(mode, test_dataset, vocab_inv)

    ############################ TRAIN MODEL #################################
    model = None
    if mode == 'deep_twitter':
        model = create_attention_model(max_query_len, max_doc_len, max_url_len, vocab_size, train_vocab_emb,
                                       args["nb_filters"], embed_size=300, dropout_rate=args['dropout'],
                                       trainable=args["trainable"], weighting=args['weighting'],
                                       mask=args["mask"], conv_option=args['conv_option'],
                                       model_option=args['model_option'])
    model_name = ("model_N%s_data%s_mo%s_c%s_NumFilter%d_T%s_D%.1f_W%s_M%s_B%d_Val%.2f" % (
        mode, train_name, args['model_option'], args['conv_option'], args["nb_filters"], args["trainable"],
        args['dropout'], args['weighting'], args['mask'], args['batch_size'], args['val_split'])).lower()
    model_path = "%s/%s/%s" % (args['experimental_data'], data_name, model_name)
    print(model_path)

    if args['optimizer'] == "adam":
        opt = optimizers.Adam(lr=args["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        print('use Adam optimizer')
    elif args['optimizer'] == "sgd":
        opt = optimizers.SGD(lr=args["learning_rate"], decay=1e-6, momentum=0.9, nesterov=True)
        print('use SGD optimizer')
    elif args['optimizer'] == 'rmsprop':
        opt = optimizers.RMSprop(lr=args["learning_rate"], rho=0.9, epsilon=None, decay=0.0)
        print('use RMSprop optimizer')

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    print('model init weights sum: %.4f' % get_model_weights(model))
    if not args['load_model']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        checkpoint = ModelCheckpoint(filepath=model_path + ".best.weights",
                                     monitor='val_loss', save_best_only=True, verbose=1)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
        #print(train_dataset['id'][:3], val_dataset['id'][:3], val_dataset['id'][-3:])
        model.fit(train_dataset, train_dataset['sim'],
                  validation_data=(val_dataset, val_dataset['sim']),
                  batch_size=args['batch_size'],
                  epochs=args['epochs'], shuffle=False,
                  callbacks=[checkpoint, lr_reducer, early_stopping],
                  verbose=args['verbose'])


    ############################ TEST MODEL #################################
    print('load best model from %s.best.weights' % model_path)
    model.load_weights("%s.best.weights" % model_path)
    if mode == 'deep_twitter':
        # load trained vocab embedding.
        trained_vocab_emb = model.get_layer('sequential_2').get_weights()[0]
        # merge trained vocab embedding with test OOV word embeddings
        merged_vocab_emb = np.zeros(shape=(len(merged_vocab['word']), 300))
        merged_vocab_emb[0:len(vocab['word']), :] = trained_vocab_emb
        merged_vocab_emb[len(vocab['word']):len(merged_vocab['word']), :] = test_vocab_emb
        for key in vocab:
            vocab_size[key] = len(merged_vocab[key])
        print(vocab_size)

        new_model = create_attention_model(max_query_len, max_doc_len, max_url_len, vocab_size, merged_vocab_emb,
                                           args["nb_filters"], embed_size=300, dropout_rate=args['dropout'],
                                           trainable=args["trainable"], weighting=args['weighting'],
                                           mask=args["mask"], conv_option=args['conv_option'],
                                           model_option=args['model_option'])
        new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(new_model.summary())
        num_layers = 0
        for layer in model.layers:
            num_layers += 1
        for layer_id in range(num_layers):
            layer = model.layers[layer_id]
            if layer.name != 'sequential_2':
                new_model.layers[layer_id].set_weights(layer.get_weights())
        print('copy weight done.')
        predictions = new_model.predict(test_dataset)

    print(predictions[:10])
    predictions_file = "%s/%s/predictions_%s.txt" % (args["experimental_data"], data_name, model_name)
    with open(predictions_file, 'w') as f:
        for i in range(test_dataset['id'].shape[0]):
            f.write("%s %.4f %s\n" % (test_dataset['id'][i], predictions[i], args['mode']))
    print('write predictions with trec format to %s' % predictions_file)
    map, mrr, p30 = evaluate(predictions_file, args["qrels_file"])
    print('MAP: %.4f P30: %.4f MRR: %.4f' % (map, p30, mrr))


if __name__ == "__main__":
    parser = create_option_parser()
    options, arguments = parser.parse_args()
    main(options)
