# File        :   main.py (Android Watch)
# Version     :   1.1.1
# Description :   Mobile/Desktop app that implements character recognition via touchscreen
# Date:       :   Jan 20, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import os

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.window import Window

from kivy.graphics import Color, Rectangle, Line
from datetime import date, datetime

# Additional modules
import cv2
import numpy as np

# from android.permissions import Permission, request_permissions, check_permission
# from android.storage import app_storage_path, primary_external_storage_path, secondary_external_storage_path

Window.clearcolor = (1, 1, 1, 1)  # RGBA

# My goddamn globals:
canvasName = "imageCanvas.png"
imagePath = ""
imageName = "raw_"
processedImageName = "processed_"

# Main project/app path:
appPath = "D://opencvImages//"  # primary_external_storage_path()

# Relative Dirs:
baseDir = "androidWatch//"  # App base directory
samplesDir = "samples//"  # Directory of saved image samples
modelDir = "model//android//"  # Directory where the SVM model resides
modelFilename = "svmModel.xml"  # Name of the SVM model file

# Image(canvas) and Label variables:
image = Image()
label = Label()

# Strings that hold the name
# of the image to write:
outPath = ""
dateString = ""
classifiedChar = ""

# Images and Cell Sizes:
saveSize = (200, 200)
processSize = (100, 100)

cellHeight = processSize[0]
cellWidth = processSize[1]

# The color of the (gray) canvas:
canvasColor = 192

# SVM variables:
svmLoaded = False
SVM = cv2.ml.SVM_create()

# The class dictionary:
classDictionary = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
                   10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
                   20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}


# Android check permissions overdrive:
# def check_permissions(perms):
#     for perm in perms:
#         if check_permission(perm) != True:
#             return False
#     return True

# Check permisions function:
def appPermisions():
    # perms = [Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]

    # if check_permissions(perms) != True:
    #
    #     print("appPermisions>> We don't have W/R permissions!")
    #     request_permissions(perms)
    #     exit()
    #     return False
    #
    # else:
    #
    #     print("appPermisions>> Got android permissions.")
    #     return True

    return True


# Waits for a keyboard event:
def waitInput(delay=0):
    cv2.waitKey(delay)


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    waitInput()


def readImage(imagePath, colorCode=cv2.IMREAD_GRAYSCALE):
    # Open image:
    print("readImage>> Reading: " + imagePath)
    inputImage = cv2.imread(imagePath, colorCode)
    # showImage("Input Image", inputImage)

    if inputImage is None:
        print("readImage>> Could not load Input image.")
        waitInput()
    else:
        print("readImage>> Input image loaded successfully.")

    return inputImage


def writeImage(imagePath, inputImage):
    # Write image and check if the operation
    # was successful
    success = cv2.imwrite(imagePath, inputImage)
    if success:
        print("writeImage>> Wrote image to: " + imagePath)
    else:
        print("writeImage>> Could not write image to: " + imagePath)
        waitInput()


# Function that fills corners of a square image:
# ( cv::Mat &inputImage, int fillColor = 255, int fillOffsetX = 10, int fillOffsetY = 10, cv::Scalar fillTolerance = 4 )
def fillCorners(binaryImage, fillColor=255, fillOffsetX=10, fillOffsetY=10):
    # Get image dimensions:
    (imageHeight, imageWidth) = binaryImage.shape[:2]
    # Flood-fill corners:
    for j in range(2):
        # Compute y coordinate:
        fillY = int(imageHeight * j + (-2 * j + 1) * fillOffsetY)
        for i in range(2):
            # Compute x coordinate:
            fillX = int(imageWidth * i + (-2 * i + 1) * fillOffsetX)
            # Flood-fill the image:
            cv2.floodFill(binaryImage, mask=None, seedPoint=(fillX, fillY), newVal=(fillColor))
            # print("X: " + str(fillX) + ", Y: " + str(fillY))
            # showImage("Flood-Fill", binaryImage)

    return binaryImage


# Gets the bounding box of a blob via horizontal and
# Vertical projections, crop the blob and returns it:

def getCharacterBlob(binaryImage, verbose):
    # Set number of reductions (dimensions):
    dimensions = 2
    # Store the data of the final bounding boxes here,
    # 4 elements cause the list is [x,y,w,h]
    boundingRect = [None] * 4

    # Reduce the image:
    for i in range(dimensions):
        # Reduce image, first horizontal, then vertical:
        reducedImg = cv2.reduce(binaryImage, i, cv2.REDUCE_MAX)
        # showImage("Reduced Image: " + str(i), reducedImg)

        # Get contours, inspect bounding boxes and
        # get the starting (smallest) X and ending (largest) X

        # Find the contours on the binary image:
        contours, hierarchy = cv2.findContours(reducedImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Create temporal list to store the rectangle data:
        tempRect = []

        # Get the largest contour in the contours list:
        for j, c in enumerate(contours):
            currentRectangle = cv2.boundingRect(c)

            # Get the dimensions of the bounding rect:
            rectX = currentRectangle[0]
            rectY = currentRectangle[1]
            rectWidth = currentRectangle[2]
            rectHeight = currentRectangle[3]
            # print("Dimension: " + str(i) + " x: " + str(rectX) + " y: " + str(rectY) + " w: " + str(
            #    rectWidth) + " h: " + str(rectHeight))

            if i == 0:
                # Horizontal dimension, check Xs:
                tempRect.append(rectX)
                tempRect.append(rectX + rectWidth)
            else:
                # Vertical dimension, check Ys:
                tempRect.append(rectY)
                tempRect.append(rectY + rectHeight)

        # Extract the smallest and largest coordinates:
        # print(tempRect)
        currentMin = min(tempRect)
        currentMax = max(tempRect)
        # print("Dimension: " + str(i) + " Start X: " + str(currentMin) + ", End X: " + str(currentMax))
        # Store into bounding rect list as [x,y,w,h]:
        boundingRect[i] = currentMin
        boundingRect[i + 2] = currentMax - currentMin
        # print(boundingRect)

    print("getCharacterBlob>> Bounding box computed, dimensions as [x,y,w,h] follow: ")
    print(boundingRect)

    # Check out bounding box:
    if verbose:
        binaryImageColor = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255)
        cv2.rectangle(binaryImageColor, (int(boundingRect[0]), int(boundingRect[1])),
                      (int(boundingRect[0] + boundingRect[2]), int(boundingRect[1] + boundingRect[3])), color, 2)
        showImage("BBox", binaryImageColor)

    # Crop the character blob:
    cropX = boundingRect[0]
    cropY = boundingRect[1]
    cropWidth = boundingRect[2]
    cropHeight = boundingRect[3]

    # Crop the image via Numpy Slicing:
    croppedImage = binaryImage[cropY:cropY + cropHeight, cropX:cropX + cropWidth]
    print("getCharacterBlob>> Cropped image using bounding box data. ")

    return croppedImage


def saveSample(*args):
    print("saveSample>> Saving image sample..")

    # Save canvas to image:
    saveImage()

    # call to postProcessSample for opencv processing:
    postProcessSample()
    print("saveSample>> Sample successfully processed and saved.")


# Saves the Canvas as an png:
def saveImage():
    print("saveImage>> Attempting to write image..")

    # Get the out dir path:
    global outPath
    outPath = os.path.join(appPath, samplesDir)

    # check if save location exists:
    dirExists = os.path.isdir(outPath)
    print("saveImages>> Found save directory: " + str(dirExists))

    if not dirExists:
        print("saveImages>> Save directory was not found, creating it...")
        os.mkdir(outPath)
        print("saveImages>> Created: " + outPath)

    # Save the png image:
    # Create name string using date and time:

    # get date:
    currentDate = date.today()
    currentDate = currentDate.strftime("%b-%d-%Y")

    # Get time:
    currentTime = datetime.now()
    currentTime = currentTime.strftime("%H:%M:%S")

    # Drop them nasty ":s":
    global dateString
    dateString = "_" + currentTime[0:2] + "-" + currentTime[3:5] + "-" + currentTime[6:8]
    print("Current Time: " + currentTime + " Date String: " + dateString)
    dateString = currentDate + dateString

    # Finally, save the raw canvas as image:
    outFileName = imageName + dateString + ".png"
    savePath = os.path.join(outPath, outFileName)

    global image
    image.export_to_png(savePath)
    print("saveImage>> Wrote image to: " + savePath)


# Process the image via OpenCV before saving it to disk:
def postProcessSample():
    # Build the input path:
    inputPath = imageName + dateString + ".png"
    inputFilename = os.path.join(outPath, inputPath)

    # Read image as grayscale:
    inputImage = readImage(inputFilename)
    # showImage("inputImage", inputImage)

    # Get height and width:
    (inputImageHeight, inputImageWidth) = inputImage.shape[:2]

    # Fill the corners with canvas image
    inputImage = fillCorners(inputImage, fillOffsetX=5, fillOffsetY=5, fillColor=canvasColor)
    # showImage("inputImage [FF]", inputImage)

    # Threshold the image, canvas color (192) = black, line color (0) = white
    _, binaryImage = cv2.threshold(inputImage, 1, 255, cv2.THRESH_BINARY_INV)
    # showImage("Binary Image", binaryImage)

    # Get character bounding box via projections and crop it:
    print("postProcessSample>> Extracting character blob...")
    characterBlob = getCharacterBlob(binaryImage, True)
    # showImage("characterBlob", characterBlob)

    # Create target canvas with the smallest original dimension:
    largestDimension = min((inputImageHeight, inputImageWidth))
    print("postProcessSample>> Largest Dimension: " + str(largestDimension))
    print("postProcessSample>> Creating canvas of: " + str(largestDimension) + " x " + str(largestDimension))

    characterCanvas = np.zeros((largestDimension, largestDimension), np.uint8)
    # showImage("characterCanvas", characterCanvas)

    # Get canvas centroid (it is a square):
    canvasX = 0.5 * largestDimension
    canvasY = canvasX

    # Get character centroid:
    (blobHeight, blobWidth) = characterBlob.shape[:2]

    # Get paste x and y:
    pasteX = int(canvasX - 0.5 * blobWidth)
    pasteY = int(canvasY - 0.5 * blobHeight)
    print("postProcessSample>> Pasting at X: " + str(pasteX) + " ,Y: " + str(pasteY) + " W: " + str(
        blobWidth) + ", H: " + str(blobHeight))

    # Paste character blob into new canvas:
    characterCanvas[pasteY:pasteY + blobHeight, pasteX:pasteX + blobWidth] = characterBlob
    # Invert image:
    characterCanvas = 255 - characterCanvas
    showImage("Pasted Image", characterCanvas)

    # Resize the image?
    (resizeWidth, resizeHeight) = saveSize
    if resizeWidth != largestDimension or resizeHeight != largestDimension:
        print("postProcessSample>> Resizing binary image to: " + str(resizeWidth) + " x " + str(resizeHeight))
        # Call the resize function:
        characterCanvas = cv2.resize(characterCanvas, saveSize, interpolation=cv2.INTER_NEAREST)
        showImage("croppedImage [resized]", characterCanvas)

    # Save the processed image:
    fileName = processedImageName + dateString + ".png"  # png -> Lossless Compression ; jpeg -> Lossy Compression
    outImagePath = os.path.join(outPath, fileName)

    # Write the output file:
    writeImage(outImagePath, characterCanvas)


# Process the image via OpenCV before passing it to
# the SVM:
def preProcessSample(outPath, dateString):
    # Build the input path:
    inputPath = processedImageName + dateString + ".png"
    inputFilename = os.path.join(outPath, inputPath)

    # Open image as grayscale:
    inputImage = readImage(inputFilename)

    # showImage("inputImage", inputImage)

    # Invert image
    # 0 (Black) - Noise, 255 (White) - Shape to analyze
    inputImage = 255 - inputImage

    # showImage("inputImage [Inverted]", inputImage)

    # Set the resizing parameters:
    (imageHeight, imageWidth) = inputImage.shape[:2]
    aspectRatio = imageHeight / imageWidth
    rescaledWidth = cellWidth
    rescaledHeight = int(rescaledWidth * aspectRatio)
    newSize = (rescaledWidth, rescaledHeight)

    # Resize image:
    inputImage = cv2.resize(inputImage, newSize, interpolation=cv2.INTER_NEAREST)

    # showImage("inputImage [Resized]", inputImage)

    # Morphological Filtering
    # Set filter kernel:
    kernelSize = (3, 3)
    opIterations = 2
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

    # Perform Dilate:
    inputImage = cv2.morphologyEx(inputImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations,
                                  cv2.BORDER_REFLECT101)

    # showImage("Dilate", inputImage)

    return inputImage


# Loads the SVM model:
def loadSvm(filePath):
    # Load XML from path:
    modelPath = os.path.join(filePath, modelFilename)
    print("loadSvm>> Loading SVM model: " + modelPath)

    # Create SVM is ready to predict
    global SVM
    SVM = cv2.ml.SVM_load(modelPath)

    # Check if SVM is ready to classify:
    svmTrained = SVM.isTrained()

    if svmTrained:
        print("loadSVM>> SVM loaded, trained and ready to test")
        return True
    else:
        print("loadSVM>> Warning: SVM is NOT trained!")
        return False


# Implements the SVM and classifies a
# sample:
def classifyImage(*args):
    print("classifyImage>> Classifying image...")

    # Save canvas to image:
    saveImage()

    # Postprocess image:
    postProcessSample()

    # Preprocess image before sending it to the SVM
    processedImage = preProcessSample(outPath, dateString)
    print("classifyImage>> Sample successfully saved and preprocessed.")

    # Reshape the image into a plain vector:
    # Convert data type to float 32
    testImage = processedImage.reshape(-1, cellWidth * cellHeight).astype(np.float32)

    # Classify "image" (vector)
    svmResult = SVM.predict(testImage)[1]
    svmResult = svmResult[0][0]
    print("classifyImage>> svmResult: " + str(svmResult))

    # Get character from dictionary:
    svmLabel = classDictionary[svmResult]

    print("classifyImage>> SVM says: " + svmLabel)

    global label
    global classifiedChar

    # Clear the dots:
    if label.text == "...":
        label.text = ""

    # Concatenate char to char on label:
    classifiedChar = label.text
    label.text = classifiedChar + svmLabel


# Check if pointer coordinates are inside an area:
def checkCoordinates(coorData, touch):
    # Unpack them values:
    (cx, cy, w, h, pointX, pointY) = coorData

    # Safe guard:
    border = 10

    # X limit
    xStart = (cx - 0.5 * w) + border
    xEnd = (cx + 0.5 * w) - border

    # Y Limit:
    yStart = (cy + 0.5 * h) - border
    yEnd = (cy - 0.5 * h) + border

    return (xStart < pointX < xEnd) and (yEnd < pointY < yStart)


# Custom Drawable canvas:
class myImage(Image):

    # Call back for touch down event:
    def on_touch_down(self, touch):
        # Width and height of rect:
        h = self.height
        w = self.height

        # Centroid:
        cx = self.center_x
        cy = self.center_y
        print("CX: " + str(cx) + ", CY: " + str(cy), " H: " + str(h) + ", W: " + str(w))

        # Pointer:
        pointX = touch.x
        pointY = touch.y
        print("Click: " + str(pointX) + ", " + str(pointY))

        # Pack them canvas & pointer values:
        coorData = (cx, cy, w, h, pointX, pointY)
        # Perform coordinates check:
        inCanvas = checkCoordinates(coorData, touch)

        # If inside canvas, draw:
        if inCanvas:
            color = (0, 0, 0)
            with self.canvas:
                Color(*color)
                d = 30.
                # Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=2.5)

    # Call back for touch move event:
    def on_touch_move(self, touch):
        # Width and height of rect:
        h = self.height
        w = self.height

        # Centroid:
        cx = self.center_x
        cy = self.center_y
        print("CX: " + str(cx) + ", CY: " + str(cy), " H: " + str(h) + ", W: " + str(w))

        # Pointer:
        pointX = touch.x
        pointY = touch.y
        print("Click: " + str(pointX) + ", " + str(pointY))

        # Pack them canvas & pointer values:
        coorData = (cx, cy, w, h, pointX, pointY)
        # Perform coordinates check:
        inCanvas = checkCoordinates(coorData, touch)

        # If inside canvas, draw:
        if inCanvas:
            if "line" in touch.ud:
                print("Exists")
                touch.ud['line'].points += [touch.x, touch.y]
                print(touch.ud['line'].points)

                x = cx - 0.5 * w
                y = cy + 0.5 * h

                print("Rect Width: " + str(w) + " Height: " + str(h) + " Offset x: " + str(x) + " Offset y: " + str(y))
                # updateNumpyImage(w, h, touch.x, touch.y, x, y)
            else:
                print("Out of canvas")

    # clears the image and reconstructs the "canvas"
    def clearImage(self, touch):
        # Canvas dimensions:
        h = self.height
        w = self.height

        # Centroid:
        x = self.center_x - 0.5 * w
        y = self.center_y + 0.5 * h

        # Clear the canvas:
        self.canvas.clear()

        # Reconstruct the touch area:
        color = (0.7529, 0.7529, 0.7529)
        with self.canvas:
            Color(*color)
            Rectangle(pos=(x, y), size=(h, -w))


class MyPaintApp(App):

    def build(self):
        screenSize = Window.size

        global appPath
        appPath = os.path.join(appPath, baseDir)

        # Should the SVM be loaded?
        global svmLoaded

        if not svmLoaded:
            # The SVM was not previously loaded:
            print("build>> Loading SVM for the first time.")
            svmFilePath = os.path.join(appPath, modelDir)
            opSuccessful = loadSvm(svmFilePath)

            if opSuccessful:
                svmLoaded = True
            else:
                print("build>> Failed to load SVM file. Check traceback.")

        # Set the layout with extra parameters: # spacing = 10 , padding = 40
        # global layout
        layout = GridLayout(cols=1, padding=100)  # col_force_default=False, col_default_width=900

        # Set the image:
        global image
        # image = Image(source=canvasName, allow_stretch=True)  # allow_stretch=True, keep_ratio=True
        image = myImage(source=canvasName, allow_stretch=True)

        # Create the relative layout:
        r1 = RelativeLayout()

        # Set button parameters:
        btnWidth = 200
        btnHeight = 50

        # Create button1:
        btn1 = Button(text="Reset", size_hint=(None, None), width=btnWidth, height=btnHeight,
                      pos_hint={"center_x": 0.5, "center_y": 0.7})  # Adjust til it properly fits into the screen
        btn1.bind(on_press=image.clearImage)
        # Add to relative layout:
        r1.add_widget(btn1)

        # Create button2:
        btn2 = Button(text="Save", size_hint=(None, None), width=btnWidth, height=btnHeight,
                      pos_hint={"center_x": 0.5, "center_y": 0.3})  # Adjust til it properly fits into the screen
        btn2.bind(on_press=saveSample)
        # Add to relative layout:
        r1.add_widget(btn2)

        # Create button3:
        btn3 = Button(text="Classify", size_hint=(None, None), width=btnWidth, height=btnHeight,
                      pos_hint={"center_x": 0.5, "center_y": -0.1})  # Adjust til it properly fits into the screen
        btn3.bind(on_press=classifyImage)
        # Add to relative layout:
        r1.add_widget(btn3)

        # Create text label:
        global label
        label = TextInput(text="...", font_size='30sp', size_hint=(None, None),
                          width=btnWidth, height=btnHeight, pos_hint={"center_x": 0.5, "center_y": 2.3})
        # Add to relative layout:
        r1.add_widget(label)

        # Add the items to layout:
        layout.add_widget(image, 1)  # Image
        layout.add_widget(r1, 0)  # Relative layout with buttons

        return layout


if __name__ == '__main__':
    MyPaintApp().run()
