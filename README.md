# A (2+1)D-CNN for analysis of multidimensional spectra
Code repo for the manuscript "Machine learning for video classification enables quantifying intermolecular couplings from simulated time-evolved multidimensional spectra"

## Overview
This package implements a **(2+1)-dimensional convolutional neural network (_(2+1)D-CNN_)** for interpreting multidimensional, mixed-domain spectra. Specifically, we demonstrate the use of a (2+1)D-CNN to classify electronic couplings in molecular dimers based on their **two-dimensional electronic spectra (2DES)**.
The scripts included implement the following:
- **Construction and initialization** of (2+1)D-CNN and analogous 3D-CNN approaches for spectral classification
  - Learnable parameters are initialized to **pre-trained values**.
  - However, one may train the provided architecture if they supply the appropriate training data and script for training. 
- **Evaluation** of pre-trained (2+1)D-CNN and 3D-CNN approaches using a **subset of the full spectral database** used in the manuscript.
- **Visualization** of CNN activations using class activation maps (CAMs)

## Dependencies
In order to install the required dependencies, use the following command
```bash
pip install torch numpy matplotlib scikit-learn pillow
```

## Usage
### 1. Directory Configuration
The repository consists of 3 scripts. main.py runs all of the calculations and is the only script that one must interface with directly. CAM.py and CNN_models.py simply include helper functions that implement CAM generation and CNN construction, respectively. The user does not need to make any changes to CAM.py and CNN_models.py. Simply ensure all 3 scripts are in the same directory.

Some **external files are required to evaluate the pre-trained models**. This includes (1) the pre-trained **weights** and (2) the **mini-database** of example spectra. Both are freely available in the associated [Zenodo repository]([url](https://doi.org/10.5281/zenodo.15178111)). If evaluating the pre-trained models on the mini-database, place the 3 .py scripts discussed above in the same directory as **Saved_models** (a folder containing the weights for all 3 models as .pt files) and **Mini_databse** (a folder containing the example spectra in dataset.pkl and more information about the Hamiltonian parameters in labels.csv). Both of these folders are available as .zip files in the Zenodo repository. This setup will allow main.py to function as intended.

### 2. Running the Code
main.py is the only script that needs to be run. Before running, choose which of the 3 models to evaluate by editing the following lines in main.py
```bash
modelsAvailable = ["2+1D-CNNa", "2+1D-CNNb", "3D-CNNa"]
modelToUse = modelsAvailable[0]  # chose the model you want to test!
```
Changing the list index here will instruct the code to evaluate that particular model. The code will proceed to construct the model and load its pre-trained weights. The model will then be evaluated on the example spectra, and the final accuracy and F1 score will be displayed. In the process, the code will generate CAMs for the correctly-classified spectra. These CAMs can be found in the directory "CAMs", which the code will create if not already present.

### 3. Visualizing Results
The performance of the pre-trained model on the mini-dataset will output to the console, and the CAMs directory will populate with PNG files that visualize the CAMs. In these images, the black-and-white spectrum is plotted behind the CAM, which is represented by the Viridis color map.

## Citing This Work
If you use this code in your research, please cite:
[manuscript citation info pending]

## License
This project is licensed under the MIT License. See LICENSE for details.
