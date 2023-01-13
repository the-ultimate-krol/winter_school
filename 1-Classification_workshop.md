# Natural language processing

## Introduction to Transformers Architecture
Currently, the most popular models for Natural Language Processing use the Transformer Architecture. There are several libraries implementing this architecture. However, in the context of NLP Huggingface transformers are most commonly used.

Apart from the source code itself, this library contains a number of other elements. Among the most important of these are:

[models](https://huggingface.co/models) - a huge and growing number of ready-made models that we can use to solve many problems in NLP (but also in speech recognition or image processing),
[datasets](https://huggingface.co/datasets) - a very large catalogue of useful datasets that we can easily use to train our own NLP models (and other models).

## Environment preparation - How to start with Google Colab

Training NLP models requires access to hardware accelerators to accelerate the learning of neural networks. If our computer is not equipped with a GPU, we can use the Google Colab environment.

In this environment, we can choose an accelerator from GPU and TPU. Let us check if we have access to an environment equipped with an NVidia accelerator by executing the following commend:

```
!nvidia-smi
```

If the accelerator is unavailable (the command ended with an error), we change the execution environment by selecting from the "Execution environment" menu -> "Change execution environment type" -> GPU.

We will then install all the necessary libraries. In addition to the `transformers` library itself, we also install the `datasets` management library datasets, a library that defines many metrics used in AI `evaluate` algorithms, and additional tools such as `sacremoses` and `sentencepiece`.

```
!pip install transformers sacremoses datasets evaluate sentencepiece
```

With the necessary libraries installed, we can use all the models and datasets registered in the catalogue.

A typical way of using the available models is:

- using a ready-made model that performs a specific task, e.g. [sentiment analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) - a model of this kind does not need to be trained, it is enough to run it in order to obtain a classification result (this can be seen in the demo at the indicated link),
- using a base model that is trained for a specific task; an example of such a model is the [HerBERT base](https://huggingface.co/allegro/herbert-base-cased), which was taught as a masked language model. To use it for a specific task, we need to select a 'classification head' for it and retrain it on our own dataset.
Models of this kind are different, and can be loaded using a common interface, but it is best to use one of the specialised classes, tailored to the task at hand. We will start by loading the BERT base model - one of the most popular models, for English. We will use it to guess missing words in the text. We will use the `AutoModelForMaskedLM` call to do this.
Use the code to see the outcome:

```
from transformers import AutoModelForMaskedLM, AutoTokenizer
```
```
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
```

## Connecting Google Drive
The final element of preparation, which is optional, is to attach your own Google Drive to the Colab environment. This makes it possible to save trained models, during the training process, to an "external" drive. If Google Colab leads to an interruption of the training process, the files that were successfully saved during the training will nevertheless not be lost. It will be possible to resume training already on a partially trained model.


To do this, we mount the Google Drive in Colab. This requires authorisation of the Colab tool in Google Drive.

```
from google.colab import drive
drive.mount('/content/gdrive') 
```
Once the drive is mounted, we have access to the entire contents of Google Drive. When indicating where to save data during a workout, indicate a path starting with `/content/gdrive`, but indicate some subdirectory within our drive space. The full path could be `/content/gdrive/MyDrive/output`. It is a good idea to check that the data writes to the drive before running the workout.

## Text tokenization
Loading the model itself, however, is not enough to start using it. We must have a mechanism for converting text (a string of characters), into a sequence of tokens, belonging to a specific dictionary. During the training of the model, this dictionary is determined (selected algorithmically) before the actual training of the neural network. Although it is possible to extend it later (training on the training data, it also allows to obtain a representation of missing tokens), usually the dictionary in the form that was defined before the neural network training is used. Therefore, it is important to specify the correct dictionary for the tokeniser performing the text splitting.

The library has an `AutoTokenizer` class that accepts the model name, which allows the dictionary corresponding to the selected neural network model to be automatically loaded. However, it is important to remember that if you are using 2 models, each will most likely have a different dictionary, and therefore they must have their own instances of the `Tokenizer` class.

```
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

The Tokenizer uses a fixed-size dictionary. This, of course, subordinates to the fact that not all words occurring in the text will be included. Furthermore, if we use the tokenizer to split text in a language other than the one for which it was created, such text will be split into a larger number of tokens.
