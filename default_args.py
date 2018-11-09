# This file contains parameters we found to be effective in our setting.
# It might be different in your case, which is affected by many factors,
# # like random seed, hardware and environment.


def get_best_args():
    return {
        'trec-2011': {
            'dropout': 0.1,
            'batch_size': 64,
            'nb_filters': 256
        },
        'trec-2012': {
            'dropout': 0.5,
            'batch_size': 64,
            'nb_filters': 256
        },
        'trec-2013': {
            'dropout': 0.4,
            'batch_size': 64,
            'nb_filters': 256
        },
        'trec-2014': {
            'dropout': 0.3,
            'batch_size': 128,
            'nb_filters': 256
        },
    }