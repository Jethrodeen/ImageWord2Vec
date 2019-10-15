"""
IMAGE WORD2VEC
Muhammad Mahir Latif - mahirlatif@live.com

Vectorization Module - based on CNNs encoded in Keras.

The module is used to convert a set of images into corresponding feature vectors.
Each feature vector is of length 2048.

"""
# import libraries
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
import os
from keras.layers import Dense
from keras.models import Model
#from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.manifold import TSNE

#===========================#functions=============================================
#=====================================================================================

# Defines the model and the corresponding size of the image they take as input.
#inputs:    #modelname : The CNN. The choices are VGG16, ResNet50, MobileNet, InceptionV3
            #isTop : boolean defining whether to return base or complete model
            #num_classes : The number of classes in the fully connected layer, if isTop = True.

def create_model(modelname, isComplete = False, num_classes = 2048,):

    if modelname == "VGG16":
        model = vgg16.VGG16(weights='imagenet', pooling = 'avg', include_top = False)
        img_size = (224,224)
    elif modelname == "ResNet50":
        model = resnet50.ResNet50(weights='imagenet',  pooling = 'avg', include_top=False)
        img_size = (224, 224)
    elif modelname == "MobileNet":
        model = mobilenet.MobileNet(weights='imagenet',  pooling = 'avg', include_top = False)
        img_size = (224, 224)
    elif modelname == "InceptionV3":
        model = inception_v3.InceptionV3(weights='imagenet',  pooling = 'avg', include_top = False)
        img_size = (299, 299)
    else:
        print ("No valid Model defined. Options are : VGG16, ResNet50, MobileNet, InceptionV3")
        return

    if isComplete:
        x = model.output
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=x)

    return model, img_size

#Process the image according to the model selected.
        #filename : complete name of image file.
        #img_size : output of the function "create_model"
        #modelname: same as the one used for "create_model"
def process_image(filename, img_size, modelname):

    # load an image in PIL format
    test_image = image.load_img(filename, target_size = img_size)

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    test_image = image.img_to_array(test_image)


    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    test_image = np.expand_dims(test_image, axis = 0)

    if modelname == "VGG16":
        processed_image = vgg16.preprocess_input(test_image.copy())
    elif modelname == "ResNet50":
        processed_image = resnet50.preprocess_input(test_image.copy())
    elif modelname == "MobileNet":
        processed_image = mobilenet.preprocess_input(test_image.copy())
    elif modelname == "InceptionV3":
        processed_image = inception_v3.preprocess_input(test_image.copy())
    else:
        print("No valid Model defined. Options are : VGG16, ResNet50, MobileNet, InceptionV3")
        return

    return processed_image

# Plot a TSNE model based on the first "n" elements of input labels and tokens
def plot_vec(tokens, labels, n):
    tsne_model = TSNE(perplexity=33, n_components=2, init='pca', n_iter=2500,random_state = 23)
    new_values = tsne_model.fit_transform(tokens[0:n])
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# ======================== end functions =======================================================
#===============================================================================================

modelname = "ResNet50"

model, img_size = create_model(modelname, isComplete = False)



inputfolder = 'outputtest'
outputfolder = 'ImgVecstest'

assert os.path.isdir(inputfolder), "Specified input folder does not exist"
if os.path.isdir(outputfolder) == False:
    print("Specified output folder does not exist. Creating it in root directory...")
    os.mkdir(outputfolder)

subfolders = os.listdir(inputfolder)
subfolders = subfolders[0:-1]

allpred = []
allnames = []
for idx, subfolder in enumerate(subfolders):
    files = os.listdir(inputfolder +  "/" + subfolder)

    if os.path.isdir(outputfolder + '/' + subfolder):
        print("Processed subfolder ", str(idx+1), " of ", str(len(subfolders)) )
        continue

    predictions = []
    names = []
    for filename in files:
        processed_image = process_image(inputfolder+ "/" + subfolder+ "/" + filename,img_size, modelname )
        # get the predicted probabilities for each class
        x = model.predict(processed_image)
        x = x[0]
        predictions.append(x.tolist())
        names.append(subfolder + "/" + filename)
        allpred.append(x.tolist())
        allnames.append(filename)

    if os.path.isdir(outputfolder+  '/' + subfolder) == False:
        os.mkdir(outputfolder +  '/' + subfolder)
    np.save(outputfolder +  '/' + subfolder + '/' + "ImgVectors", predictions)
    np.save(outputfolder +  '/' + subfolder + '/' + "Names", names)
    print("Processed subfolder ", str(idx+1), " of ", str(len(subfolders)))


plot_vec(allpred,allnames,500)
#plot_vec(np.load("output/000000008211.jpg/ImgVectors.npy"), np.load("output/000000008211.jpg/Names.npy"))