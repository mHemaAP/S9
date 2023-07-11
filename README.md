# S9 - Neural Network Image Classification - Dilated Convolutions and Depthwise Separable Convolutions - CIFAR10 DataSet
This is the repository which contains the assignment work of Session 9

## Description

This project includes 7 Python files: `abstract_dataset.py`, `utils.py`, 'back_propogation.py', 'cifar10_dataset.py', 'model.py', 'model_training.py', 'scheduler.py' and one notebook `S9.ipynb'. ALl the python (.py) files are placed in the **cifar** folder. These files are part of a machine learning project for image classification using the CIFAR10 dataset. The project is about training a neural network model to recognize images using less than 200K parameters, and with test accuracy equal to or greater than 85% using dilated convolutions and depthwise separable convolutions. The weights corresponding to the best accuracies (>=85%) are placed in the folder **weights** 

Few samples in the dataset are shown below.

![Training Sample Images](Images/Train_Image_Samples.png)


## Files

### 1. model.py

This file 'model.py' defines a neural network architecture using PyTorch. It consists of two classes: convLayer and Net. This network architecture uses **Dilated convolution (1, 2, 4, 8) + Depth wise Separable Convolution + Skip Connections** to achieve the goal of receptive field>44, parameter count <= 200K. The convLayer class represents a convolutional layer with customizable parameters such as input channels, output channels, bias, padding, depthwise separable convolution, skip connections, dilation, and dropout. It initializes the layers required for the convolutional operation and provides a forward method to perform the convolution and apply normalization, activation, and dropout if specified. 

The 'Net' class represents the main neural network architecture. It inherits from the nn.Module class and defines the layers and forward method for the network. It consists of several convBlock and transitionBlock layers, followed by an outblock layer. Each convBlock represents a sequence of convolutional layers defined by the convLayer class. Similarly, each transitionBlock represents a single convolutional layer. The outblock is the final block in the network that performs adaptive average pooling, followed by a convolutional layer, flattening, and log-softmax activation. 

The Net class also includes a summary method that uses the torchinfo.summary function to generate a summary of the network architecture, including input and output sizes, the number of parameters, kernel sizes, and computational statistics.

Overall, this code implements a neural network with multiple convolutional layers and skip connections. The network architecture is modular and allows customization of various parameters to create different configurations of the network.

To run this code, we can create an instance of the Net class, pass input data through the network using the forward method, and utilize the summary method to get a summary of the network architecture.

#### 1.1 Model Architecture [Dilated Convolution + Depthwise Separable Convolution + Skip Connection + Batch Normalization] Summary gave Parameter Count = 190,509 given below (Also Refer the S9.ipnb file for the summary)

The receptive field at the output block of this model is 48


=====================================================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Kernel Shape              Param %
=====================================================================================================================================================================
Net                                      [512, 3, 32, 32]          [512, 10]                 --                        --                             --
├─Sequential: 1-1                        [512, 3, 32, 32]          [512, 23, 32, 32]         --                        --                             --
│    └─convLayer: 2-1                    [512, 3, 32, 32]          [512, 23, 32, 32]         --                        --                             --
│    │    └─Sequential: 3-1              [512, 3, 32, 32]          [512, 23, 32, 32]         621                       --                          0.33%
│    │    └─BatchNorm2d: 3-2             [512, 23, 32, 32]         [512, 23, 32, 32]         46                        --                          0.02%
│    │    └─ReLU: 3-3                    [512, 23, 32, 32]         [512, 23, 32, 32]         --                        --                             --
│    └─convLayer: 2-2                    [512, 23, 32, 32]         [512, 23, 32, 32]         --                        --                             --
│    │    └─Sequential: 3-4              [512, 23, 32, 32]         [512, 23, 32, 32]         782                       --                          0.41%
│    │    └─BatchNorm2d: 3-5             [512, 23, 32, 32]         [512, 23, 32, 32]         46                        --                          0.02%
│    │    └─ReLU: 3-6                    [512, 23, 32, 32]         [512, 23, 32, 32]         --                        --                             --
├─convLayer: 1-2                         [512, 23, 32, 32]         [512, 32, 30, 30]         --                        --                             --
│    └─Sequential: 2-3                   [512, 23, 32, 32]         [512, 32, 30, 30]         --                        --                             --
│    │    └─Conv2d: 3-7                  [512, 23, 32, 32]         [512, 32, 30, 30]         6,624                     [3, 3]                      3.48%
│    └─BatchNorm2d: 2-4                  [512, 32, 30, 30]         [512, 32, 30, 30]         64                        --                          0.03%
│    └─ReLU: 2-5                         [512, 32, 30, 30]         [512, 32, 30, 30]         --                        --                             --
├─Sequential: 1-3                        [512, 32, 30, 30]         [512, 32, 30, 30]         --                        --                             --
│    └─convLayer: 2-6                    [512, 32, 30, 30]         [512, 32, 30, 30]         --                        --                             --
│    │    └─Sequential: 3-8              [512, 32, 30, 30]         [512, 32, 30, 30]         1,376                     --                          0.72%
│    │    └─BatchNorm2d: 3-9             [512, 32, 30, 30]         [512, 32, 30, 30]         64                        --                          0.03%
│    │    └─ReLU: 3-10                   [512, 32, 30, 30]         [512, 32, 30, 30]         --                        --                             --
│    └─convLayer: 2-7                    [512, 32, 30, 30]         [512, 32, 30, 30]         --                        --                             --
│    │    └─Sequential: 3-11             [512, 32, 30, 30]         [512, 32, 30, 30]         1,376                     --                          0.72%
│    │    └─BatchNorm2d: 3-12            [512, 32, 30, 30]         [512, 32, 30, 30]         64                        --                          0.03%
│    │    └─ReLU: 3-13                   [512, 32, 30, 30]         [512, 32, 30, 30]         --                        --                             --
├─convLayer: 1-4                         [512, 32, 30, 30]         [512, 63, 26, 26]         --                        --                             --
│    └─Sequential: 2-8                   [512, 32, 30, 30]         [512, 63, 26, 26]         --                        --                             --
│    │    └─Conv2d: 3-14                 [512, 32, 30, 30]         [512, 63, 26, 26]         18,144                    [3, 3]                      9.52%
│    └─BatchNorm2d: 2-9                  [512, 63, 26, 26]         [512, 63, 26, 26]         126                       --                          0.07%
│    └─ReLU: 2-10                        [512, 63, 26, 26]         [512, 63, 26, 26]         --                        --                             --
├─Sequential: 1-5                        [512, 63, 26, 26]         [512, 63, 26, 26]         --                        --                             --
│    └─convLayer: 2-11                   [512, 63, 26, 26]         [512, 63, 26, 26]         --                        --                             --
│    │    └─Sequential: 3-15             [512, 63, 26, 26]         [512, 63, 26, 26]         4,662                     --                          2.45%
│    │    └─BatchNorm2d: 3-16            [512, 63, 26, 26]         [512, 63, 26, 26]         126                       --                          0.07%
│    │    └─ReLU: 3-17                   [512, 63, 26, 26]         [512, 63, 26, 26]         --                        --                             --
│    └─convLayer: 2-12                   [512, 63, 26, 26]         [512, 63, 26, 26]         --                        --                             --
│    │    └─Sequential: 3-18             [512, 63, 26, 26]         [512, 63, 26, 26]         4,662                     --                          2.45%
│    │    └─BatchNorm2d: 3-19            [512, 63, 26, 26]         [512, 63, 26, 26]         126                       --                          0.07%
│    │    └─ReLU: 3-20                   [512, 63, 26, 26]         [512, 63, 26, 26]         --                        --                             --
├─convLayer: 1-6                         [512, 63, 26, 26]         [512, 93, 18, 18]         --                        --                             --
│    └─Sequential: 2-13                  [512, 63, 26, 26]         [512, 93, 18, 18]         --                        --                             --
│    │    └─Conv2d: 3-21                 [512, 63, 26, 26]         [512, 93, 18, 18]         52,731                    [3, 3]                     27.68%
│    └─BatchNorm2d: 2-14                 [512, 93, 18, 18]         [512, 93, 18, 18]         186                       --                          0.10%
│    └─ReLU: 2-15                        [512, 93, 18, 18]         [512, 93, 18, 18]         --                        --                             --
├─Sequential: 1-7                        [512, 93, 18, 18]         [512, 93, 18, 18]         --                        --                             --
│    └─convLayer: 2-16                   [512, 93, 18, 18]         [512, 93, 18, 18]         --                        --                             --
│    │    └─Sequential: 3-22             [512, 93, 18, 18]         [512, 93, 18, 18]         9,672                     --                          5.08%
│    │    └─BatchNorm2d: 3-23            [512, 93, 18, 18]         [512, 93, 18, 18]         186                       --                          0.10%
│    │    └─ReLU: 3-24                   [512, 93, 18, 18]         [512, 93, 18, 18]         --                        --                             --
│    └─convLayer: 2-17                   [512, 93, 18, 18]         [512, 93, 18, 18]         --                        --                             --
│    │    └─Sequential: 3-25             [512, 93, 18, 18]         [512, 93, 18, 18]         9,672                     --                          5.08%
│    │    └─BatchNorm2d: 3-26            [512, 93, 18, 18]         [512, 93, 18, 18]         186                       --                          0.10%
│    │    └─ReLU: 3-27                   [512, 93, 18, 18]         [512, 93, 18, 18]         --                        --                             --
├─convLayer: 1-8                         [512, 93, 18, 18]         [512, 93, 2, 2]           --                        --                             --
│    └─Sequential: 2-18                  [512, 93, 18, 18]         [512, 93, 2, 2]           --                        --                             --
│    │    └─Conv2d: 3-28                 [512, 93, 18, 18]         [512, 93, 2, 2]           77,841                    [3, 3]                     40.86%
│    └─BatchNorm2d: 2-19                 [512, 93, 2, 2]           [512, 93, 2, 2]           186                       --                          0.10%
│    └─ReLU: 2-20                        [512, 93, 2, 2]           [512, 93, 2, 2]           --                        --                             --
├─Sequential: 1-9                        [512, 93, 2, 2]           [512, 10]                 --                        --                             --
│    └─AdaptiveAvgPool2d: 2-21           [512, 93, 2, 2]           [512, 93, 1, 1]           --                        --                             --
│    └─Conv2d: 2-22                      [512, 93, 1, 1]           [512, 10, 1, 1]           940                       [1, 1]                      0.49%
│    └─Flatten: 2-23                     [512, 10, 1, 1]           [512, 10]                 --                        --                             --
│    └─LogSoftmax: 2-24                  [512, 10]                 [512, 10]                 --                        --                             --
=====================================================================================================================================================================
Total params: 190,509
Trainable params: 190,509
Non-trainable params: 0
Total mult-adds (G): 26.45
=====================================================================================================================================================================
Input size (MB): 6.29
Forward/backward pass size (MB): 4740.16
Params size (MB): 0.76
Estimated Total Size (MB): 4747.22
=====================================================================================================================================================================


### 2. back_propogation.py
This file also has **train** and **test** functions which perform the training and evaluation functionalities respectively and return the train loss, train accuracy, and test loss, test accuracy respectively.

### 3. utils.py

The `utils.py` file contains helper functions that are used throughout the project. These functions provide some common functionalities for acquiring underlying device specifics like whether CUDA/GPU is available, visualization, or any other necessary operations. It includes function to obtain the samples of mis-classifications (function: **get_incorrect_test_predictions**) and function to obtain the device underneath (function: **get_device**).

### 4. abstract_dataset.py

The `abstract_dataset.py` file contains the abstract class **dataSet** which defines the needed data set attributes and methods. This is the base class which can be used for various data set types like MNIST, CIFAR and so on. In the context of this project, this base class is inherited to define the CIFAR10 specific data set operations and transforms. The basic functionality to obtain train and test transforms, to obtain train and test data loaders which finally get loaded into the model, and functionality to visualize data set images **show_dataset_images** as needed for this project are defined as methods in this class.

### 5. cifar10_dataset.py

This file contains the definition of cifar10Set which defines the needed attributes and methods for CIFAR10 data set. This class also inherits the needed methods from albumentationTransforms class to perform the desired transformations on the CIFAR10 data set. The transformations or the image augmentations performed after normalizing the train data set are **ColorJitter**, **ToGray**, **HorizontalFlip** (also known as FlipLR), **ShiftScaleRotate**, **CoarseDropout** (also known as CutOut). The parameter values and the probabilities of performing these augmentations were tuned to give out accuracies >= 85%

### 6. model_training.py

This file contains class **trainModel** which covers the attributes and the functionality needed to execute the training of the model. The method of this class that actually trains the model is **run_training_model**. This method uses the train and test methods of backpropogation.py file for training and validating the model. The other methods of this trainModel class are - **display_model_stats** to show the model statistics, **show_epoch_progress** to show the progress of learning rate, train, test accuracies/losses at the end of each epoch 

### 7. scheduler.py

This file contains methods **get_sgd_optimizer** to return the SGD optimizer with starting learning rate of 0.05, and momentum 0.9, and **get_one_cycle_LR_scheduler** to return the scheduler based on One cycle Policy with increasing learning rate for 0.4 of the cycle time, and defining the cycle time to be 3 phased.

### 8. S9.ipynb

The `S9.ipynb` file is the notebook that has executions of training model based on the network architectures Net so as to execute the models right from scratch till we find the optimal model which gives 85%+ accuracy using less than 200K parameters. 

#### 8.1. Descritpion of basic components in the notebook

This `S9.ipynb`file contain code for training and evaluating a neural network model using the CIFAR10 dataset. These files include the following components:

- Importing necessary libraries and dependencies
- Mounting Google Drive
- Setting up the device (CPU or GPU)
- Defining data transformations for training and testing
- Loading the CIFAR10 dataset
- Setting up data loaders
- Displaying sample data from the training set
- Defining the neural network model (model1) and displaying its summary
- Training the model using SGD optimizer and NLL loss
- Displaying model training and testing statistics
- Displaying the incorrect test image predictions

In addition to the above components, each of these notebooks contain "Target-Results-Analysis" section for each of the models executed.

#### 8.2. Descritpion of CIFAR10 dataset

This data set consists of multiple images pertaining to 10 different classes. The following are the specifics of the 10 classes and the count of images in each of these classes.
['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck'] 

 {'frog': 5000,
 'truck': 5000,
 'deer': 5000,
 'automobile': 5000,
 'bird': 5000,
 'horse': 5000,
 'ship': 5000,
 'cat': 5000,
 'dog': 5000,
 'airplane': 5000}

#### 8.3 Description of the model

- Following are the target-result-analysis of the final best model for each of the BN techniques

#### 8.3.1 Code Block - Final Optimal Model

**Target:**

- Network Architecture - **Dilated Convolution + Depthwise Separable Convolution**
- Layer Structure - C1 C2 C3 C4 Output + No MaxPooling, all convolution blocks have 3 layers and use 3x3 convolutions, GAP + FC + 1x1 convolutions
- **Skip connections** used
- Batch Normalization used
- Image Augmentation using Albumentation Transforms - Color Jitter + To Gray + Horizontal flip + Shift Scale Rotate + Coarse Dropout
- Dropout of 0.05 and one cycle learning rate with max LR as 0.1 used
- L1 regularization degraded the performance and led the model to over-fitting, hence it is not used
- The receptive field we got with this network architecture is 48
- 
#### 8.3.2 Findings on the Normalization Techniques tried

**Results:**

- Parameters: 190.509K
- Best Train Accuracy: 86.19%
- Best Test Accuracy: 87.22% (95th Epoch)
- Test Accuracy: 85.39% (59th Epoch) [First occurrence of test accuracy above 85%]

**Analysis:**
- Model has no over fitting through out the training
- Model displayed little under-fitting behaviour which is desirable for us
- Scheduling OneCycle LR with pct_start = 0.4 gave better train/test accuracies trend in the first 100 epochs
  - Keeping the LR fixed at 0.1 did not show this kind of performance, the best test accuracy achieved with in first 100 epochs is only 86.83%


#### 8.4 Graphs of the Normalization Techniques tried [Best Model plots shown below]

The following model statistics for the normalization techniques tried. These plots pertain to the performance of the final optimal model using architecture Net and the other model specifics are highlighted below

##### 8.4.1 Final Optimal Model Training and Test Statistics - BN

![Model Statistics - BN](Images/Model_Statistics.png)

#### 8.5 Collection of Mis-classified Predictions

The following image collection shows the mis-classified predictions for the normalization techniques tried. These images pertain to the preidctions made by the final optimal model (in S9.ipynb) using architecture Net and the other model specifics are highlighted below

![Misclassified Prediction - BN](Images/Model_Incorrect_Predictions.png)


##### 8.6 Images of RF Calculations and Formulae Used

![RF Calculation Table](Images/RF_calculation_Table.png)

![RF Calculation Formulae](Images/RF_calculation_Formulae.png)


## Usage

To run the project, make sure you have the dependencies installed.
```
pip install -r requirements.txt
```
We can execute the `S9.ipynb` notebook to perform the training and testing. Adjust the hyperparameters such as learning rate, momentum, batch size, and number of epochs to improve the model performance as desired.

Please note that this README serves as a placeholder. As I make further modifications to the project, I would keep this file updated accordingly. 

For more detailed information on the project's implementation and code, please refer to the individual files mentioned above.
