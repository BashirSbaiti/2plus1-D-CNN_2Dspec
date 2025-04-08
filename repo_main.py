import os
import pickle as pkl
import numpy as np
import torch
from sklearn.metrics import f1_score
from repo_CNN_models import CNN2plus1Le, CNN3dLe
from repo_CAM import generateCAMoverlay

datasetFile = "Mini_database/dataset.pkl"

modelsAvailable = ["2+1D-CNNa", "2+1D-CNNb", "3D-CNNa"]
modelToUse = modelsAvailable[0]  # chose the model you want to test!

savedModelFile2p1a = "Saved_models/2p1a_model.pt"
savedModelFile2p1b = "Saved_models/2p1b_model.pt"
savedModelFile3d = "Saved_models/3d_model.pt"

# below are construction parameters for the 3 models. do not change!
params2p1a = {
    "convShape": (15, 3, 3),
    "stemConvShape": (15, 10, 10),
    "numClasses": 5,
    "convBlocks": 5,
    "outChannels": (32, 64, 128, 256, 512),
    "strides": (2, 2, 2, 2),
    "t2pads": (0, 0, 0, 0)
}

params2p1b = {
    "convShape": (15, 3, 3),
    "stemConvShape": (15, 10, 10),
    "numClasses": 5,
    "convBlocks": 5,
    "outChannels": (32, 64, 128, 256, 512),
    "strides": (2, 2, 2, 1),
    "t2pads": (0, 0, 0, 0)
}

params3d = {
    "convShape": (15, 3, 3),
    "stemConvShape": (15, 10, 10),
    "numClasses": 5,
    "convBlocks": 5,
    "outChannels": (32, 64, 128, 256, 512),
    "strides": (2, 2, 2, 2),
    "t2pads": (0, 0, 0, 0)
}

with open(datasetFile, 'rb') as f:
    dataset = pkl.load(f)

inShape = (
    1, dataset["system spectra"][0].shape[2], dataset["system spectra"][0].shape[0],
    dataset["system spectra"][0].shape[1]
)

params = None
savedModelFile = None
if modelToUse == "2+1D-CNNa":
    params = params2p1a
    savedModelFile = savedModelFile2p1a
elif modelToUse == "2+1D-CNNb":
    params = params2p1b
    savedModelFile = savedModelFile2p1b
elif modelToUse == "3D-CNNa":
    params = params3d
    savedModelFile = savedModelFile3d

model = None
if "2+1D-CNN" in modelToUse:
    model = CNN2plus1Le(inShape, params["convShape"], params["stemConvShape"], params["numClasses"],
                        params["convBlocks"],
                        params["outChannels"], params["strides"], params["t2pads"], useGAP=True)
elif "3D-CNN" in modelToUse:
    model = CNN3dLe(inShape, params["convShape"], params["stemConvShape"], params["numClasses"], params["convBlocks"],
                    params["outChannels"], params["strides"], params["t2pads"], useGAP=True)

stateDict = torch.load(savedModelFile, map_location=torch.device('cpu'))
model.load_state_dict(stateDict)
model.eval()
print("-----------model loaded-----------")

# define function to extract activations of the final convolution automatically during evaluation
# this way, after each data input, featureMap is updated with the feature map from the final conv block automatically
featureMaps = None


def hookFeature(module, input, output):
    global featureMaps
    featureMaps = output.data.numpy()


model._modules.get("fullNet")._modules.get("Conv Block 5").register_forward_hook(hookFeature)

# testing of fully trained model
allPreds = list()
allLabels = list()
for i, spectrum in enumerate(dataset["system spectra"]):
    if i % (len(dataset['system spectra']) // 5) == 0:
        print(f"Evaluation progress: {(float(i) / len(dataset['system spectra'])) * 100}%")
    input = torch.from_numpy(np.moveaxis(spectrum, 2, 0)).float()  # reshapes (w1, w3, t2) --> (t2, w1, w3)
    input = input.unsqueeze(0).unsqueeze(0)

    output = model(input)

    realJ = int(dataset["system parameters"][i]["J_Coul"])
    if realJ < -650:
        Jcategory = 0
    elif realJ < -250:
        Jcategory = 1
    elif realJ < 250:
        Jcategory = 2
    elif realJ < 650:
        Jcategory = 3
    elif realJ <= 800:
        Jcategory = 4
    else:
        raise Exception("Unexpected J value")

    _, prediction = torch.max(output, 1)  # predicted J category is given by the maximum index of output

    # only reporting CAMs for correct classifications
    if prediction.item() == Jcategory:
        CAMoverlay = generateCAMoverlay(model, featureMaps, input.squeeze())

        saveDir = f"CAMs"
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        CAMoverlay.save(f"{saveDir}/CAM{i + 1}.png")

    allPreds.append(prediction.item())
    allLabels.append(Jcategory)

print("Evaluation progress: 100.0%")
print("-----------evaluation completed-----------")
acc = 100.0 * np.average(np.array(allPreds) == np.array(allLabels))
f1 = f1_score(allLabels, allPreds, average="macro")

print(f"Accuracy is {round(np.mean(acc), 2)}, F1 score is {round(f1, 4)}")
