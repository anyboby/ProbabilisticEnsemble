# Probabilistic Ensemble w. epistemic and aleatoric uncertainty.
This repository contains a probabilistic ensemble for aleatoric and epistemic uncertainty quantification.

## Prerequisites
This repository was tested on Ubuntu 16 LTS and Ubuntu 18 LTS. The main prerequisites are python 3.6 and tensorflow/tensorflow-gpu 14. 

## Install
We typically prefer installs via anaconda, even though installs through other virtual environment managers should pose no problem.


``` bash
conda create -n pe_env python=3.6
conda activate pe_env
pip install -r req.txt
```

## Usage
run.py provides an example for how to build the model. The pe_factory serves as an easy builder utility.

The main purpose of this model is to provide scale-invariant epistemic uncertainty estimates in terms of the average KL divergence between ensemble models. It will take some training for these estimates to take place, but a modified 'MSPE' loss (as opposed to Negative Log-Likelihood MLE) provides more stable variance predictions in unknown regions. 

![Results of the example run.py](https://photos.google.com/share/AF1QipPUuXNyrVrEddhFCwQc-LRfW-C1ACDS6ObUbbUUBccaxb1CwLo6P7yODtEKY78WUw/photo/AF1QipOya1e6P0MKhCh2Fci9Bbm3fAZgP9VumiFs8oBt?key=bXFPTzN4NEYxSkdMdnBnYi1BcWdiUDUwVnJpQXV3)

## Parameters
Most parameters in this model will follow the common understanding, there are, however a number of special things to consider:

1. 
2. Loss Type: You can choose the loss type per string among a selection of ['MSPE', 'NLL', 'MSE', 'Huber', 'CE']. The CE loss will require a slightly different formatting for inputs and outputs. 
3. use_scaler: The scaler can be used for both inputs and outputs. It normalized training samples to mean 0 and std. 0 with a running mean and std. Scaling outputs should be done with caution and has only been tested for MSE, NLL and MSPE.
4. The 'MSPE' loss is the main distinction of this paper and serves as an alternative loss for probabilistic predictions. Instead of performing MLE (as NLL does), it directly approximates the MSE of a prediction. We find this to be more well-behaved in certaint situations which is a crucial factor in computing the average Kullback-Leibler divergence between predictions. 


## Acknowledgement
This code is based largely on previous work by Michael Janner and Kurtland Chua
