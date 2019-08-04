# dialog-eval &middot; [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://ctt.ac/a16pa)
[![Paper](https://img.shields.io/badge/Presented%20at-ACL%202019-yellow.svg)](https://www.aclweb.org/anthology/P19-1567) [![Poster](https://img.shields.io/badge/The-Poster-yellow.svg)](https://ricsinaruto.github.io/website/docs/acl_poster_h.pdf) [![Code1](https://img.shields.io/badge/code-chatbot%20training-green.svg)](https://github.com/ricsinaruto/Seq2seqChatbots) [![Code2](https://img.shields.io/badge/code-filtering-green.svg)](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering) [![documentation](https://img.shields.io/badge/documentation-on%20wiki-red.svg)](https://github.com/ricsinaruto/dialog-eval/wiki) [![blog](https://img.shields.io/badge/Blog-post-black.svg)](https://medium.com/@richardcsaky/neural-chatbots-are-dumb-65b6b40e9bd4)  
A lightweight repo for automatic evaluation of dialog models using **17 metrics**.

## Features
  :twisted_rightwards_arrows: &nbsp; Choose which metrics you want to be computed  
  :rocket: &nbsp;&nbsp; Evaluation can automatically run either on a response file or a directory containing multiple files  
  :floppy_disk: &nbsp; Metrics are saved in a pre-defined easy to process format  
  :warning: &nbsp; The program warns you if some files required to compute specific metrics are missing  
  
### Metrics
* **Response length**: Number of words in the response.
* **[Per-word entropy](http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf)**: Probabilities of words are calculated based on frequencies observed in the training data. Entropy at the bigram level is also computed.
* **[Utterance entropy](http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf)**: The product of per-word entropy and the response length. Also computed at the bigram level.
* **[KL divergence](https://www.aclweb.org/anthology/P19-1567)**: Measures how well the word distribution of the model responses approximates the ground truth distribution. Also computed at the bigram level (with bigram distributions).
* **[Embedding](https://aclweb.org/anthology/D16-1230)**: Embedding *average*, *extrema*, and *greedy* are measured. *average* measure the cosine similarity between the averages of word vectors of response and target utterances. *extrema* constructs a representation by taking the greatest absolute value for each dimension among the word vectors in the response and target utterances and measures the cosine similarity between them. *greedy* matches each response token to a target token (and vica versa) based on the cosine similarity between their ebeddings and averages the total score across all words. 
* **[Coherence](https://arxiv.org/pdf/1809.06873.pdf)**: Cosine similarity of input and response representations (constructed with the average word embedding method).
* **[Distinct](https://www.aclweb.org/anthology/N16-1014)**: Distinct-1 and distinct-2 measure the ratio of unique unigrams/bigrams to the total number of unigrams/bigrams in a set of responses.
* **[BLEU](https://www.aclweb.org/anthology/P02-1040)**: Measures n-gram overlap between response and target (n = [1,2,3,4]). Smoothing method can be choosen in the arguments.



## Setup
Run this command to install required packages:
```
pip install -r requirements.txt
```

## Usage
The main file can be called from anywhere, but when specifying paths to directories you should give them from the root of the repository.
```
python code/main.py -h
```
<a><img src="https://github.com/ricsinaruto/dialog-eval/blob/master/docs/help.png" align="top" height="500" ></a>

For the complete documentation visit the [wiki](https://github.com/ricsinaruto/dialog-eval/wiki).

### Input format
You should provide as many of the argument paths required (image above) as possible. If you miss some the program will still run, but it will not compute some metrics which require those files (it will print these metrics). If you have a training data file the program can automatically generate a vocabulary and download fastText embeddings.  
  
If you don't want to compute all the metrics you can set which metrics should be computed in the [config](https://github.com/ricsinaruto/dialog-eval/blob/master/utils/config.py) file very easily.

### Saving format
A file will be saved to the directory where the response file(s) is. The first row contains the names of the metrics, then each row contains the metrics for one file. The name of the file is followed by the individual metric values separated by spaces. Each metric consists of three numbers separated by commas: the mean, standard deviation, and confidence interval. You can set the t value of the confidence interval in the arguments, the default is for 95% confidence.

## Results & Examples
### [Transformer](https://arxiv.org/abs/1706.03762) trained on [DailyDialog](https://arxiv.org/abs/1710.03957)
Interestingly all 17 metrics improve until a certain point and then stagnate with no overfitting occuring during the training of a Transformer model on DailyDialog. Check the appendix of the [paper](https://arxiv.org/pdf/1905.05471.pdf) for figures.  
<a><img src="https://github.com/ricsinaruto/dialog-eval/blob/master/docs/dailydialog_metrics.png" align="top" height="110" ></a>  
TRF is the Transformer model evaluated at the validation loss minimum and TRF-O is the Transformer model evaluated after 150 epochs of training, where the metrics start stagnating. RT means randomly selected responses from the training set and GT means ground truth responses.  

### [Transformer](https://arxiv.org/abs/1706.03762) trained on [Cornell](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
<a><img src="https://github.com/ricsinaruto/dialog-eval/blob/master/docs/cornell_metrics.png" align="top" height="100" ></a>  
TRF is the Transformer model, while RT means randomly selected responses from the training set and GT means ground truth responses. These results are on measured on the test set at a checkpoint where the validation loss was minimal.  

### [Transformer](https://arxiv.org/abs/1706.03762) trained on [Twitter](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/twitter)
<a><img src="https://github.com/ricsinaruto/dialog-eval/blob/master/docs/twitter_metrics.png" align="top" height="90" ></a>  
TRF is the Transformer model, while RT means randomly selected responses from the training set and GT means ground truth responses. These results are on measured on the test set at a checkpoint where the validation loss was minimal.  

## Contributing
##### Check the [issues](https://github.com/ricsinaruto/dialog-eval/issues) for some additions where help is appreciated. Any contributions are welcome :heart:
##### Please try to follow the code syntax style used in the repo (flake8, 2 spaces indent, 80 char lines, commenting a lot, etc.)

**New metrics** can be added by making a class for the metric, which handles the computation of the metric given data. Check [BLEU metrics](https://github.com/ricsinaruto/dialog-eval/blob/master/code/metrics/bleu_metrics.py) for an example. Normally the init function handles any data setup which is needed later, and the update_metrics updates the metrics dict using the current example from the arguments. Inside the class you should define the self.metrics dict, which stores lists of metric values for a given test file. The names of these metrics (keys of the dictionary) should also be added in the config file to self.metrics. Finally you need to add an instance of your metric class to [self.objects](https://github.com/ricsinaruto/dialog-eval/blob/master/code/metrics/metrics.py#L97). Here at initialization you can make use of paths to data files if your metric requires any setup. After this your metric should be automatically computed and saved.  
  
However, you should also add some constraints to your metric, e.g. if a file required for the computation of the metric is missing the user should be notified, as [here](https://github.com/ricsinaruto/dialog-eval/blob/master/code/metrics/metrics.py#L64).

## Authors
* **[Richard Csaky](ricsinaruto.github.io)** (If you need any help with running the code: ricsinaruto@hotmail.com)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ricsinaruto/dialog-eval/blob/master/LICENSE) file for details.  
Please include a link to this repo if you use it in your work and consider citing the following paper:
```
@inproceedings{Csaky:2019,
    title = "Improving Neural Conversational Models with Entropy-Based Data Filtering",
    author = "Cs{\'a}ky, Rich{\'a}rd and Purgai, Patrik and Recski, G{\'a}bor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1567",
    pages = "5650--5669",
}
```
