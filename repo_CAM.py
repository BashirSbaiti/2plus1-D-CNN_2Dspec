import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import functional as F


def spec2images(inputSpec):
    allt2Imgs = []
    for t2Ind in range(inputSpec.shape[-3]):
        specImage = np.array(inputSpec).squeeze()[t2Ind]  # extract one w1xw3 spectrum
        specImage = specImage - np.min(specImage)
        specImage = specImage / np.max(specImage)
        specImage = np.flipud(specImage)
        allt2Imgs.append(specImage)
    return allt2Imgs


def generateCAM(convFeatures, weights, classInds, finalSize, heatcmap):
    """
    :param convFeatures: final activation coming out of the network before GAP
                        shape = (nChannels, 1, x, x)
    :param weights: weights matrix for linear taking gap fiter -> numClasses
    :param classInds: for what output class ind to consider
    :return:
    """
    b, n, d, h, w = convFeatures.shape
    allCams = []
    allCamImgs = []

    for i in classInds:
        # weights[i] is 512 dimensional matrix representing weights for that class
        # multiply it by the [512, 14*14] matrix, representing weighting each
        # activation map by its weight
        cam = weights[i].dot(convFeatures.reshape([n, h * w]))
        cam = cam.reshape(h, w)  # rebuild into matrix
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.flipud(cam)

        camSav = Image.fromarray(cam)
        camSav = camSav.resize(finalSize, Image.LANCZOS)
        camSav = np.array(camSav)

        allCams.append(camSav)

        camImg = heatcmap(cam)
        camImg = (camImg[:, :, :3] * 255).astype(np.uint8)
        camImg = Image.fromarray(camImg)
        allCamImgs.append(camImg.resize(finalSize, Image.LANCZOS))

    return allCamImgs, allCams


def generateCAMoverlay(model, featureMaps, spectra, overlayt2ind = 0, heatcmap = plt.get_cmap("viridis")):
    """
    :param featureMaps: arraylike of feature maps to make CAM with, shape is (nFeatureMaps, t2, w1, w3)
    :spectra: arraylike of spectra for all t2, shape is (t2, w1, w2)
    :overlayt2ind: t2 index to overlay CAM on
    :heatcmap: matplotlib colormap for heatmap
    """
    model.eval()
    params = list(model.parameters())
    linearWeights = np.squeeze(params[-2].data.numpy())  # extract weights matrix of final linear layer

    specImageLst = spec2images(spectra)

    outVec = model(spectra.unsqueeze(0).unsqueeze(0))
    smoutVec = F.softmax(outVec, dim=1).data.squeeze()
    probs, classInds = smoutVec.sort(0, True)
    classInds = classInds.numpy()

    CAMimgs, CAMsavs = generateCAM(featureMaps, linearWeights, [classInds[0]], spectra.shape[1:], heatcmap)

    specimg = specImageLst[overlayt2ind]
    specimg = (specimg * 255).astype(np.uint8)
    specimg = Image.fromarray(specimg, mode='L').convert('RGBA')

    heatmap = CAMimgs[0].convert('RGBA')

    result = Image.blend(specimg, heatmap, alpha=0.5)

    return result



