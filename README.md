# AReM-DeepLearning-ActivityRecognition
This repository contains the code and resources for the Activity Recognition System based on Multisensor Data Fusion (AReM) project, completed as part of the deep learning course (CS7389H) at the Department of Computer Science, Texas State University.

## Dataset
The dataset used in this project is titled "Activity Recognition system based on Multisensor data fusion (AReM)" and is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem). It comprises sensor data collected from wearable devices for activity recognition tasks. Initially, the project identified data leakage in the dataset. To address this issue, preprocessing techniques were employed along with data augmentation methods such as adding noise to generalize the data.


## Installation

To set up the project, you need to have Python and PyTorch installed. Clone the repository and install the required packages:

```bash
git clone https://github.com/Habibirani/AReM-DeepLearning-ActivityRecognitiont.git
cd AReM-DeepLearning-ActivityRecognition
conda env create -f environment.yml

```

## Models Trained
Three different deep learning models were trained for activity recognition:

- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)
- Transformer

All codes for both the original dataset and the augmented dataset are available in the 'scripts' directory.


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


