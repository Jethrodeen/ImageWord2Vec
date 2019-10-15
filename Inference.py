"""
IMAGE WORD2VEC
Muhammad Mahir Latif - mahirlatif@live.com

Inference Module - based on the maskRCNN implementaion of matterport
https://github.com/matterport/Mask_RCNN


The module is used to detect the type of objects in a set of images.
Each object is then stored as a separate image.


"""

#import libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
import time
import os
import math
import csv

# define the configuration. This class is defined in "Config.py" and is subclassed here. 
# refer to config.py for details.
class Inference(Config):
    NAME = "Inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

# define 81 classes that the coco model knows about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# a function which crops and stores all the detected objects as separate images
        ## image: keras image file, which is output of the load_image function 
        ## boxes: bounding coordinates for all detected objects in the image
        ## class_id: class IDs for all detected objects in the image
        ## class_names: names corresponding to class IDs
        ## folder: output folder where extracted objects will be stored.
def extract_objects(image, boxes, class_id, class_names, folder):
    # load the image
    data = image
    # crop each box
    for i in range(len(boxes)):
        # get coordinates
        y1, x1, y2, x2 = boxes[i]
        im1= data.crop((x1,y1,x2,y2))
        # show the plot

        im1.save(folder + '/' + str(class_names[class_id[i]]) + str(i) + '.jpg' )

# define the model with the configurations set.
rcnn = MaskRCNN(mode='inference', model_dir='./', config=Inference())

# load the coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

#folder containing all images we want to infer
imgfolder = 'test'

#folder where we want our extracted images to be written
outputfolder = 'outputtest'

#directory assertions
assert os.path.isdir(imgfolder), "Specified image folder does not exist"
if os.path.isdir(outputfolder) == False:
    print("Specified output folder does not exist. Creating it in root directory...")
    os.mkdir(outputfolder)



# load the list of images
imglist =  os.listdir(imgfolder)
totalimgs = str(len(imglist))


imgs = []
results =[]

# set up object counter. This file will detail how many instances of each object are present in our image collection
if os.path.isfile(outputfolder + "/counter.csv"):
    with open(outputfolder + "/counter.csv", mode='r') as infile:
        reader = csv.reader(infile)
        class_counter = {rows[0]: float(rows[1]) for rows in reader}
else:
    class_counter = dict(zip(class_names, np.zeros(81)))


#start loop.

t0 = time.time()


for idx, image in enumerate(imglist):

    #if output folder for image already exists, go to next image.
    if os.path.isdir(outputfolder + '/' + imglist[idx]):
        print("Predicted image " + str(idx + 1) + " of " + totalimgs)
        continue

    #load image as a numpy array   
    imgfile  = load_img(imgfolder + "/" + image)
    img = img_to_array(imgfile)
    img = np.array(img)
    imgs.append(img)
    img = [img]

    #detect objects. The function returns the class and bounding boxes for all objects detected in an image
    # "r" is a dictionary.
    r = rcnn.detect(img, verbose=0)[0]

    #write each object as a separate file using "extract_objects".
    #only do this for images where there is more than one object detected.
    if len(r['class_ids']) > 1 :
        os.mkdir(outputfolder + '/' + imglist[idx])
        extract_objects(imgfile, r['rois'], r['class_ids'], class_names, outputfolder +'/' + imglist[idx])

        #increment counter for the objects detected
        for id in r['class_ids']:
            class_counter[class_names[id]] += 1


    print("Predicted image " + str(idx+1) + " of " + totalimgs)


    #update counter file
    with open(outputfolder + '/counter.csv', 'w') as f:
        for key in class_counter.keys():
            f.write("%s,%s\n" % (key, class_counter[key]))

t1 = time.time()


print("Model Predicted in ", math.floor((t1-t0)/60)," minutes and ", (t1-t0)%60 , " seconds.")

