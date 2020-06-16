# Vanilla Neural Network 

Automatic bike share rental systems have become commonplace in many cities. Ridership is tightly correlated to weather conditions, date, and time of day. This repository is an attempt to illustrate the building blocks of a basic neural network through the lense of a model that predicts bike rental ridership. This project was part of Udacity's Deep Learning Nanodegree and uses a dataset<sup>[[1]](#1)</sup> from the University of Porto, Portugal. 

**Note:** This model does not use Pytorch or other frameworks. Feel free to see if you can improve on the predictions using more depth or other methods.   

### Navigation
* [Running Notebook Locally](#installing_locally)
* [Data Preparation](#data_prep)
* [Model Architecture](#model_arch)
* [Results](#results)
* [References](#referances)

<a id='installing_locally'></a>
## Installing Locally
If would like to tinker feel free to install locally and make it your own.

1. Install dependencies - I generally use Conda for my environment and package management. 

	>`conda install -c conda-forge jupyterlab`  

	>`pip install requirments.txt` 

2. Dataset - Hourly and daily data in `csv` format is located in the `/datset` folder. A `Readme.txt` file explains how the data is formated. 

3. The following Jupyter notebook uses the `vanilla_n_class.py` file:
    * `Predicting_bike_sharing.ipynb` Jupyter notebook 
 
<a id='data_prep'></a>
## Data Preparation

Bla Bla

<a id='model_arch'></a>
## Model Architecture

<p align="center">
<img width="280" src = "imgs/2_layer_nn.png">
</p>

- Sigmoid activation function

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

- Output (prediction) formula

$$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$

- The function that updates the weights

$$ w_i \longrightarrow w_i + \alpha (y - \hat{y}) x_i$$

$$ b \longrightarrow b + \alpha (y - \hat{y})$$

<a id='results'></a>
## Results

<p align="center">
<img width="400" src = "imgs/loss_01.jpg">
</p>


<p align="center">
<img width="800" src = "imgs/pred_01.jpg">
</p>

## References

[1]<a id='1'></a> Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.