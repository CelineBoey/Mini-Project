from joblib import load   # load model
import argparse     # command line argument
import os           # working directory
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier

# change working directory command line arguments
os.chdir("D:/UTAR course/UCCC2513 mini project/notebook_files")

parser = argparse.ArgumentParser(description = "first knn classifier for cats vs dogs")
parser.add_argument('--input', help = "path to input image", default = './images/dog.jfif')
args = parser.parse_args()

# load external images
img_ori = cv.imread(cv.samples.findFile(args.input))
img = cv.resize(img_ori, (64, 64), interpolation = cv.INTER_AREA)

# HOG
winSize = (64, 64)    
blockSize = (16, 16)  #*
blockStride = (8, 8)  #*
cellSize = (8, 8)     #*
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True  #*
nlevels = 64
signedGradient = False   #*

hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                      nbins, derivAperture, winSigma, histogramNormType,
                      L2HysThreshold, gammaCorrection, nlevels, 
                      signedGradient)

# perform feature extraction with HOG
img_hog = hog.compute(img)
img_hog = img_hog.reshape((1, -1))

# load knn model and make prediction on the new image
knn_custom = load("knn_self.joblib")
pred = knn_custom.predict(img_hog)[0]

# annotate on the image and show
h = img_ori.shape[0]
cv.putText(img_ori, pred, (20, h-30), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

# cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', img_ori)
cv.waitKey(0)
cv.destroyAllWindows()