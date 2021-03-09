# Probabilistic Ensemble w. epistemic and aleatoric uncertainty.
This repository contains a probabilistic ensemble for aleatoric and epistemic uncertainty quantification.

## Prerequisites
This repository was tested on Ubuntu 16 LTS and Ubuntu 18 LTS. The main prerequisites are python 3.6 and tensorflow/tensorflow-gpu 14. 

## Install
We typically prefer installs via anaconda, even though installs through other virtual environment managers should pose no problem.


``` bash
conda create -n pe_env python=3.6
conda activate pe_env
pip install -r requirements.txt
```

## Usage
run.py provides an example for how to build the model. The pe_factory serves as an easy builder utility.

The main purpose of this model is to provide scale-invariant epistemic uncertainty estimates in terms of the average KL divergence between ensemble models. It will take some training for these estimates to take place, but a modified 'MSPE' loss (as opposed to Negative Log-Likelihood MLE) provides more stable variance predictions in unknown regions. 

![Results of the example run.py](https://lh3.googleusercontent.com/pw/ACtC-3cWhZQNPlDDqQC-YtzXaFgA1cDyIt0AyxFBdkj_lNdcQxwXMBdJH2IHKbrk9LfvCDykJa7Qwf7gEiPP-hkor-cLuousEae3jipKl9JGqeil8wh7yrbO-HKSPV1aWxwcjrvnYlMlvUxW6wVT68gzu2gFfw=w1549-h911-no?authuser=0)

## Parameters
Most parameters in this model will follow the common understanding. There are, however a number of special things to consider:
 
1. Loss Type: You can choose the loss type per string among a selection of ['MSPE', 'NLL', 'MSE', 'Huber', 'CE']. The CE loss will require a slightly different formatting for inputs and outputs. 
2. use_scaler: The scaler can be used for both inputs and outputs. It normalizes training samples to mean 0 and std 1 with a running mean and std. Scaling outputs should be done with caution and has only been tested for MSE, NLL and MSPE. The variance prediction of the MSPE loss is based on output scaling and should thus also be used in junction with output scaling.
3. The 'MSPE' loss is the main distinctive feature of this repo and serves as an alternative loss for probabilistic predictions. Instead of performing MLE (as NLL does), it directly approximates the MSE of a prediction. Empirically, this appears to be more well-behaved for variance predictions in unseen regions which is a crucial factor in computing the average Kullback-Leibler divergence between predictions.


## Acknowledgement
This code is based largely on previous work by Michael Janner and Kurtland Chua
