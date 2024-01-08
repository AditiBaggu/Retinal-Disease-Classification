Retinal Disease Classification

A detailed guide and explanation of the codebase for a Retinal Disease Classification project. The project encompasses various components, including dataset handling, model architecture, training procedures, evaluation metrics, and a Streamlit web application for real-time predictions.

1. Dataset Handling

RetinalDiseaseDataset Class:
The RetinalDiseaseDataset class is a custom dataset handler designed to manage the retinal disease dataset. It initializes with the dataset directory and allows for optional image transformations. The class implements methods for loading images and preparing samples for training and evaluation.

default_loader Function:
The default_loader function is a utility that defines a default image loader using the Python Imaging Library (PIL). It converts images to the RGB format.

2. Data Preparation

Transforms:
The code defines two sets of image transformations:
    train_transform: Applied to training images, including resizing, random horizontal flip, random rotation, color jitter, random resized crop, random grayscale, and normalization.
    val_transform: Applied to validation and test images, including resizing and normalization.

Dataset Paths:
The paths for training (train_data_dir) and validation (val_data_dir) datasets are specified.

Dataset Creation:
Instances of the ImageFolder class are created for both training and validation datasets, utilizing the specified transforms.

Data Loaders
Data loaders are employed to handle the batch loading of training and validation datasets, with a specified batch size.

3. Model Architecture

VisionTransformer Class:
The VisionTransformer class implements a Vision Transformer model using the timm library. It allows for a specified number of output classes and replaces the final fully connected layer with a linear layer for classification.

ResNet101 Class:
The ResNet101 class implements a ResNet101 model, replacing the final fully connected layer with a linear layer for classification.

VotingEnsemble Class:
The VotingEnsemble class combines multiple models into an ensemble using a simple averaging mechanism.

4. Model Training

Device Configuration:
The code determines whether to use GPU (cuda) or CPU based on availability.

Random Seed:
A random seed is set to ensure reproducibility of the experiments.

Optimizer and Scheduler:
The Adam optimizer is configured for multiple models with different learning rates. A learning rate scheduler based on validation loss is employed.

Loss Function:
The cross-entropy loss function is defined for model training.

Training Loop:
The training loop runs for a specified number of epochs, updating model weights, and collecting metrics for each epoch. Training and validation accuracy, as well as loss, are printed for each epoch.

5. Model Evaluation

Test Dataset Handling:
The code sets up a test dataset using the ImageFolder class.

Test Loop:
The models are evaluated on the test dataset, calculating accuracy and storing predictions.

Metric Evaluation:
The code computes and prints the classification report and confusion matrix for model evaluation.

Model Saving:
The trained ensemble model, optimizer state, and training metrics are saved for future use.

6. Model Inference

Image Prediction:
The code loads the saved ensemble model and performs real-time predictions on a single input image. The predicted class and associated class information are printed.

7. Streamlit Web Application

Streamlit App:
A Streamlit web application is implemented, allowing users to upload an image for real-time prediction. The application displays the uploaded image, predicted class, and associated information.

Dependencies:
Ensure that the following libraries are installed to run the code:

pip install torch torchvision timm matplotlib pandas scikit-learn pillow streamlit

Usage:

    1. Set the appropriate paths for the training, validation, and test datasets.

    2. Adjust hyperparameters such as batch size, learning rate, and the number of epochs.

    3. Run the code to train the models, evaluate performance, and save the ensemble model.

    4. Load the saved model for real-time predictions using the Streamlit web application.

Credits:
The code utilizes various open-source libraries, including PyTorch, timm, Streamlit, scikit-learn, and Pillow.

License:
This project is licensed under the MIT License. Feel free to use and modify the code for your own purposes.

For any questions or issues, please contact the project contributors.