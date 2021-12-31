# BARZ: Barrier Zones for Adversarial Example Defense
Code corresponding to the Barrier Zone (BARZ) defense paper for the CIFAR-10 dataset. We provide a PyTorch version of the code and the trained models in both PyTorch and TensorFlow. 

# PyTorch Implementation of BARZ



# Tensorflow Implementation of BARZ

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

For comparisons and follow up work, we acknowledge the legitimacy of using the PyTorch GitHub code for reporting future BARZ related results. 

# Credit

Our code makes use of some existing codes which we credit here. Specifically: 
<ul>
<li>The original PyTorch ResNet code from which the BARZ code is built on is written by Yerlan Idelbayev can be found here: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py</li>
</ul>
<ul>
<li>The PyTorch version of the Mixed/Adaptive Black-Box attack is based on the TensorFlow Cleverhan's implementation which can be found here: https://github.com/cleverhans-lab/cleverhans.</li>
</ul>

# Contact 

For questions or concerns please contact: kaleel.mahmood@uconn.edu 
