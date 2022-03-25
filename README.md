# MasterThesis-SLDTW
For Master Thesis docs and code.
This python scripts were made as a part of my master thesis work for the University of Hildesheim which seeks to develop a surrogate loss network for Dynamic Time Warping. Dynamic Time Warping is non-differentiable beacuse of its 'min' cost alignment function. Hence, we use a neural network to aprroximate the DTW distance. The proposed method is applicable to time series datasets and used different types of forecasting models and levarages different losss functions in its training. We used soft-DTW [1] as our baseline and compared the performance of surrogate loss network.

In order to run this code, follow the steps provided in the notebook.

## References
<a id="1">[1]</a> 
Mehran Maghoumi (2020). 
Deep Recurrent Networks for Gesture Recognition and Synthesis 
University of Central Florida Orlando, Florida.
