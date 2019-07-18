# dialog-eval &middot; [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://ctt.ac/E_jP6)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Presented%20at-ACL%202019-yellow.svg)](https://arxiv.org/abs/1905.05471) [![Code1](https://img.shields.io/badge/code-chatbot%20training-green.svg)](https://github.com/ricsinaruto/Seq2seqChatbots) [![Code2](https://img.shields.io/badge/code-filtering-green.svg)](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering) [![documentation](https://img.shields.io/badge/documentation-on%20wiki-red.svg)](https://github.com/ricsinaruto/dialog-eval/wiki)  
A lightweight repo for automatic evaluation of dialog models using **17 metrics**.

## Features
  :twisted_rightwards_arrows: &nbsp; Choose which metrics you want to be computed  
  :rocket: &nbsp;&nbsp; Evaluation can automatically run either on a response file or a directory containing multiple files  
  :floppy_disk: &nbsp; Metrics are saved in a pre-defined easy to process format  
  :warning: &nbsp; The program warns you if some files required to compute specific metrics are missing  
  
### Metrics
* Response length: Number of words in the response.
* Per-word entropy: Probabilities of words are calculated based on frequencies observed in the training data. Entropy at the bigram level is also computed.
* Utterance entropy: The product of per-word entropy and the response length. Also computed at the bigram level.
* KL divergence: Measures how well the word distribution of the model responses approximates the ground truth distribution. Also computed at the bigram level (with bigram distributions).
* Embedding: Embedding *average*, *extrema*, and *greedy* are measured. *average* measure the cosine similarity between the averages of word vectors of response and target utterances. *extrema* constructs a representation by taking the greatest absolute value for each dimension among the word vectors in the response and target utterances and measures the cosine similarity between them. *greedy* matches each response token to a target token (and vica versa) based on the cosine similarity between their ebeddings and averages the total score across all words. 
* Coherence: Cosine similarity of input and response representations (constructed with the average word embedding method).
* Distinct: Distinct-1 and distinct-2 measure the ratio of unique unigrams/bigrams to the total number of unigrams/bigrams in a set of responses.
* BLEU: Measures n-gram overlap between response and target. n = [1,2,3,4]. Smoothing method can be choosen in the arguments.



## Setup
Run this command to install required packages:
```
pip install -r requirements.txt
```

## Usage
The main file can be called from anywhere, but when specifying paths to directories you should give them from the root of the repository.
```
python main.py -h
```
<a><img src="https://github.com/ricsinaruto/dialog-eval/blob/master/docs/help.png" align="top" height="500" ></a>

For the complete documentation visit the [wiki](https://github.com/ricsinaruto/dialog-eval/wiki).

### Input format

### Saving format


## Results & Examples


## Contributing

## Authors

## License

## Acknowledgments


##### If you require any help with running the code or if you want the files of the trained models, write to this e-mail address. (ricsinaruto@hotmail.com)
