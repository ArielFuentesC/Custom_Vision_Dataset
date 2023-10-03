# Custom Vision Dataset Exercises

## Tasks

### Task 1: Data Splitting and Classification

In Task 1, we focus on data splitting and classification of a dataset. The main objectives include:

- **Data Splitting:**
  - Perform a train-validation-test split of the dataset, consisting of 17,034 images.
  - Stratified splitting is applied.
  - A manual seed is used to ensure reproducibility.

- **Verification of Disjoint Splits:**
  - This step ensures that there is no overlap between the training, validation, and test sets.

- **Data Loaders:**
  - Create data loaders for the training, validation, and test dataset for loading and handling of data during model training and evaluation.

- **Choice of Neural Network:**
  - Choose a neural network architecture for the classification task (ResNet50).
  - Perform fine-tuning of the selected neural network for improved performance on the dataset.

- **Training:**
  - Train the neural network using appropriate loss functions for the training task.
  - Experiment with different hyperparameter settings, such as learning rates, optimizers, and data augmentation parameters.
  - Ensure that the batch size during training is adjusted to prevent excessive GPU memory usage.

- **Evaluation:**
  - Report classification accuracy per class on the validation set, as well as the average precision measure per class.

- **Model Selection:**
  - Experiment with multiple hyperparameter settings and choose the best-performing model based on its performance on the validation set.

- **Testing:**
  - Apply the selected model to make predictions on the test dataset.

### Task 2: Internals of Network Feature Map Statistics

In Task 2, the inner workings of the neural network are investigated by computing statistics for selected feature maps. This task involves:

- **Module Iteration:**
  - We iterate through all modules of the neural network using the named_modules() method in PyTorch.

- **Feature Map Selection:**
  - We choose five specific feature maps as the outputs of modules for further analysis.

- **Statistics Computation:**
  - Using forward hooks, we compute statistics for these selected feature maps. These statistics include the percentage of non-positive values averaged over all channels and spatial elements. This analysis helps us understand the behavior of these feature maps within the network.

### Task 3: Feature Map Analysis

In Task 3, we analyze feature maps to gain insights into their properties. The key steps include:

- **Compute Feature Map Mean:**
  - For each of the five chosen feature maps and 200 images, we calculate the mean of the feature map over all spatial dimensions. This results in a tensor with batch and channel indices.

- **Empirical Covariance Matrix:**
  - Next, we compute the empirical covariance matrix over the channels for each feature map. The covariance matrix is of shape (C, C), where C is the number of channels in the feature map.
  - To calculate this, we average over the data in minibatches, and we compute the covariance matrix using a specific formula.

- **Eigenvalues Analysis:**
  - We obtain the top-k eigenvalues of the empirical covariance matrix, sorting them in decreasing order. Typically, k is set to 1000.

- **Eigenvalue Plotting:**
  - We plot the sorted top-k eigenvalues for analysis.

- **Comparison Across Datasets:**
  - We repeat these computations for three different scenarios:
    - Using the ImageNet-initialized model without any fine-tuning and images from our dataset, upscaled to (224, 224).
    - Using the ImageNet-initialized model and 200 ImageNet validation set images, scaled to (224, 224).
    - Using the ImageNet-initialized model and 200 CIFAR-10 images, scaled to (224, 224).
  - This allows us to compare the computed statistics across different datasets.

This task aims to understand feature map characteristics and how they vary across different datasets and model initializations, providing valuable insights into the network's behavior.
