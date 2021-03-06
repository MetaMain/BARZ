# BARZ: Barrier Zones for Adversarial Example Defense
Code corresponding to the Barrier Zone (BARZ) defense paper: https://ieeexplore.ieee.org/document/9663375

In this repository we give the BARZ code and models for the CIFAR-10 dataset. We provide a PyTorch version of the code and the trained models in both PyTorch and TensorFlow. 

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation Section (see below).</li>
  <li>Download the models from the Google Drive link listed in the PyTorch BARZ Models Section.</li>
  <li>Set the "modelsDir" variable on line 10 in "BarrierZoneTrainer.py" to the directory where you saved the models e.g. modelDir = "C://Users//Downloads//BARZ-8 PyTorch Models//" </li>
  <li>Open the BarrierZoneTrainer.sln file in the Python IDE of your choice. Choose one of the attack or training lines and uncomment it. Run the main.</li>
</ol>

# Software Installation 

We use the following software packages: 
<ul>
  <li>pytorch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>numpy==1.19.2</li>
</ul>

# PyTorch BARZ Models

The trained PyTorch version of the BARZ models are available for download [here](https://drive.google.com/file/d/1LHxYEjuNPzrvm62hL_o84VMPXA_6stXQ/view?usp=sharing).

# Tensorflow BARZ Models

The original BARZ models for CIFAR-10 in TensorFlow are available for download [here](https://drive.google.com/file/d/18N686ZqgX2oopOrvjcPoeSOLq2o9FTIJ/view?usp=sharing).
While we no longer support TensorFlow implementation, the models can be loaded in TensorFlow 1.X using the following command:
```
model=tensorflow.keras.models.load_model(modelFileDir,  custom_objects={"tensorflow": tensorflow}) 
```
where "modelFileDir" is the file path of the saved model, e.g., *"C://Users//BARZ-8 TensorFlow//BUZZ32_ResNet6v2_model.h5"*

# Code Results vs Paper Results 

The results reported in the main paper were run in 2018-2020 using TensorFlow 1.12. There are several differences between the PyTorch code provided in this GitHub and the original TensorFlow code. As a consequence, the result that can be re-created with this GitHub code are not identical to the results reported in the paper. We list the main differences below:   

<ol>
  <li>The PyTorch attack code uses a balanced dataset (i.e. an equal number of samples from each class). The original TensorFlow code used the first n correctly identified samples.</li>
   <li>The PyTorch attack code generates synthetic data using the Fast Gradient Sign Method (FGSM). The original TensorFlow code used Jacobian-based dataset augmentation.</li>
   <li>The PyTorch ResNet models are trained with different hyperparameters and different dataset augmentation techniques as compared to the ResNet models in TensorFlow/Keras.</li>
</ol>

***For comparisons and follow up work, we acknowledge the legitimacy of using the PyTorch GitHub code for reporting future BARZ related results.***

# Credit

Our code makes use of some existing codes which we credit here. Specifically: 
<ul>
<li>The original PyTorch ResNet code from which the BARZ code is built on is written by Yerlan Idelbayev and can be found here: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py</li>
</ul>
<ul>
<li>The PyTorch version of the Mixed/Adaptive Black-Box attack is based on the TensorFlow Cleverhan's implementation which can be found here: https://github.com/cleverhans-lab/cleverhans.</li>
</ul>

# Contact 

For questions or concerns please contact: kaleel.mahmood@uconn.edu 
