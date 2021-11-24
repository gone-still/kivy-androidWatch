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
modelDir = "model//win//"  # Directory where the SVM model resides
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

# SVM variables:
svmLoaded = False
SVM = cv2.ml.SVM_create()

# The class dictionary:
classDictionary = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


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

    # Input image
    # showImage("inputImage", inputImage)

    # Get height and width:
    (inputImageHeight, inputImageWidth) = inputImage.shape[:2]

    # Threshold (Umbralizacion/Binarizacion)
    # Conversion of an grayscale image to a binary image
    # Threshold is at 128 -> pixel>= 128 - > white (outPixel = 255), otherwise, outPixel = 0
    # OTSU (automatic calculation of threshold value)
    _, binaryImage = cv2.threshold(inputImage, 128, 255, cv2.THRESH_BINARY)
    # showImage("binaryImage", binaryImage)

    # Get first row (of the binary image):
    # Numpy Slicing (Extracts one small array of a bigger one)
    binaryRow = binaryImage[0:1, :]
    # showImage("binaryRow", binaryRow)

    # Get Bounding Rectangle:
    # the function receives a binary image:
    rect = cv2.boundingRect(binaryRow)

    # Print the bounding rectangle coordinates
    # (topLeft(x,y), width, height)
    # print(rect)

    # Set the cropping dimensions:
    cropX = rect[0]
    cropY = 0
    cropWidth = rect[2]
    cropHeight = inputImageHeight

    # Crop the image (Numpy Slicing):
    croppedImage = binaryImage[cropY:cropY + cropHeight, cropX:cropX + cropWidth]
    # showImage("croppedImage", croppedImage)

    # Resize the image:
    aspectRatio = cropHeight / cropWidth
    rescaledWidth = saveSize[0]
    rescaledHeight = int(rescaledWidth * aspectRatio)
    newSize = (rescaledWidth, rescaledHeight)

    # Call the resize function:
    croppedImage = cv2.resize(croppedImage, newSize, interpolation=cv2.INTER_NEAREST)
    # showImage("croppedImage [resized]", croppedImage)

    # Save the processed image:
    fileName = processedImageName + dateString + ".png"  # png -> Lossless Compression ; jpeg -> Lossy Compression
    outImagePath = os.path.join(outPath, fileName)

    # Write the output file:
    writeImage(outImagePath, croppedImage)


# Process the image via OpenCV before passing it to
# the SVM:
def preProcessSample(outPath, dateString):
    # Build the input path:
    inputPath = processedImageName + dateString + ".png"
    inputFilename = os.path.join(outPath, inputPath)

    # Open image as grayscale:
    inputImage = readImage(inputFilename)

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

    svmLabel = " "

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
