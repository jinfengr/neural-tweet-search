# Multi-Perspective Relevance Matching with Hierarchical ConvNets for Social Media Search
This repo contains code and data for our neural tweet search paper published in [AAAI'19](https://arxiv.org/abs/1805.08159).

In social media search, the scenario is different as standard ad-hoc retrieval: shorter document length, less formal languages and multiple relevance source signals (e.g., URL, hashtag). We propose a hierarchical convolutional model to approach the hetergeneous relevance signals (tweet, URL, hashtag) at multiple perspectives, including character-, word-, phrase- and sentence-level modeling. Our model demonstrated significant gains on multiple twitter datasets against state-of-the-art neural ranking models. More details can be found in our paper.


## Requirements
- Python 2.7
- Tensorflow or Theano (tested on TF 1.4.1)
- Keras (tested on 2.0.5)

## Install
- Download our repo:
```
git clone https://github.com/Jeffyrao/neural-tweet-search.git
cd neural-tweet-search
```
- Install [gdrive](https://github.com/prasmussen/gdrive)
- Download required data and word2vec:
```
$ chmod +x download.sh; ./download.sh
```
- Install Tensorflow and Keras dependency:
```
$ pip install -r requirements.txt
```

## Run
- Train and test on GPU:
```
CUDA_VISIBLE_DEVICES=0 python -u train.py -t trec-2013
```
The path of best model and output predictions will be shown in the log. Default parameters should work reasonably well.
- Parameter sweep to find the best parameter set:
```
chmod +x param_sweep.sh; ./param_sweep.sh trec-2013 &
```
This command will save all the outputs under tune-logs folder. 
## Evaluate with trec_eval
```
$ ./trec_eval.8.1/trec_eval data/twitter-v0/qrels.microblog2011-2014.txt \
                            best_run/mphcnn_trec_2013_pred.txt
```
This should return the exact MPHCNN score on TREC 2013 dataset (MAP: 0.2818, P30: 0.5222) we reported in our paper.

## Command line parameters
| option                   | input format |   default   | description |
|--------------------------|--------------|-------------|-------------|
| `-t`   | [trec-2011, trec-2012, trec-2013, trec-2014] | trec-2011 | test set |
| `-l`   | [true, false]       | false     | whether to load pre-created dataset (set to true when data is ready) |
| `--load_model`     | [true, false]       | false     | whether to load pre-trained model |
| `-b`   | [1, n)    | 64 | batch size | 
| `-n`    | [1, n)    | 256 | number of convolutional filters |
| `-d`    | [0, 1]    | 0.1 | dropout rate | 
| `-o`    | [sgd, adam, rmsprop] | sgd | optimization method | 
| `--lr`  | [0, 1]    | 0.05 | learning rate |
| `--epochs`| [1, n)  | 15   | number of training epochs | 
| `--trainable` | [true, false] | true | whether to train word embeddings | 
| `--val_split` | (0, 1) | 0.15 | percentage of validation set sampled from training set | 
| `-v`| [0, 1, 2] | 1 | verbose (for logging), 0 for silent, 1 for interactive, 2 for per-epoch logging |
| `--conv_option` | [normal, ResNet]       | normal     | convolutional model, normal or ResNet |
| `--model_option`| [complete, word-url]       | complete | what input sources to use, complete for MP-HCNN, word-url for only modeling query-tweet (word) and query-url (char)  |

## Reference
If you are using this code or dataset, please kindly cite the paper below:
```
@article{rao2018multi,
  title={Multi-Perspective Relevance Matching with Hierarchical ConvNets for Social Media Search},
  author={Rao, Jinfeng and Yang, Wei and Zhang, Yuhao and Ture, Ferhan and Lin, Jimmy},
  journal={arXiv preprint arXiv:1805.08159},
  year={2018}
}
```
