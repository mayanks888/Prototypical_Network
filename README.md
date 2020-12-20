# Prototypical Network

A re-implementation of [Prototypical Network](https://arxiv.org/abs/1703.05175).

`implementation and improvement is still in progress`

##Datasets:
The model is modified for  training with the following datasets:
1. Omniglot :self download through script
2. Mini_imagenet (Download the images: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)
3. Custom datasets : Create a unique folder for data classes and place respective image inside it

    `goto parser_util.py for selecting datasets`

### Results

1-shot: to be calculated

5-shot:  to be calculated
## Environment
## Installing the right version of PyTorch 

This project is updated to be compatible with pytorch 1.0.1 and requires python 3.6

You can find other project requirements in `requirements.txt` , which you can install using `pip install -r requirements.txt

## Instructions

`update information at "parser_util.py`

### 1-shot Train

`python train.py`

### 1-shot Test

`python eval.py`


### 1-shot inference 

In progress..