Repository for the MVA course Bayesian Machine Learning on the topic of "Deep Gaussian Process and Applications"
=====

Repository based on the original repository https://github.com/SheffieldML/PyDeepGP, repository for the code of the paper "Deep Gaussian Processes" from Damianou and al. (2013)

This repository includes the following notebooks for our experiments:

- GP_Samples.ipynb : Notebook exploring the space of functions encapsulated in different Gaussian and Deep Gaussian Processes. Also includes regressions.
- GP-LVM_Unsupervised.ipynb : Application of the Gaussian Process Latent Variable Model (GP-LVM, Titsias and al. 2010) for the Mnist Dataset.
- DeepGP_Supervised.ipynb : Application of deep GPs for regression of non-stationary functions.
- DeepGP_Fashion.ipynb : Application of deep GPs for unsupervised learning on the cifar fashion dataset.
- Bayesian_Optimization.ipynb : Notebook illustrating the use of deep gaussian processes for bayesian optimization.
- bo.py : Python module handling the bayesian optimization mechanisms.

# Unsupervised Learning
![](/examples/ResultsDGPUnsup/super_plot.png)

# Supervised Learning
![](/examples/ResultsDGPSup/SupervisedDeepGp.png)
