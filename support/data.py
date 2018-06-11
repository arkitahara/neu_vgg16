""" helper library for routine functions """
import glob
import os
import skimage
import numpy as np
from skimage.color import gray2rgb
from keras.applications.vgg16 import preprocess_input
import warnings
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.spatial.distance import cdist


def test():
    """ test that helper library loaded properly """
    print('the test works')

def get_files(directory, extension='*.png'):
    files = glob.glob(os.path.join(directory, extension))
    return files

def image_tensor(image):
    """ Convert grayscale image to keras tensor appropriate for the VGG model """

    # yield an RGB image on the range [0.0,255.0]
    # convert to ubyte (integer on range [0,255])
    image = skimage.img_as_ubyte(image)

    # copy grayscale image onto color channels
    image3d = gray2rgb(image)

    # convert to floating point
    image3d = image3d.astype(np.float32)

    # add the sample dimension to the array
    x = np.expand_dims(image3d, axis=0)

    # keras.applications.vgg16.preprocess_input
    # subtract mean values over the ImageNet dataset from each channel
    return preprocess_input(x)

def image_montage(X, images, bordercolors, mapsize=8192, thumbsize=256, bordersize=4, verbose=False):
    """ make image maps in an embedding space """

    halfthumbsize = int((thumbsize + 2*bordersize)/2)
    map_shape = np.array([mapsize,mapsize,3])
    imagemap = np.ones(map_shape)

    # rescale max distance from origin to 1
    scale = np.max(np.abs(X[:,0:2]))

    # choose some random images to draw
    # sel = np.random.choice(range(keys.size), replace=False, size=2000)

    for ids, image in enumerate(images):

        # get image position
        pos = X[ids][:2]

        # load image
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            im = skimage.io.imread(image, as_grey=True)

        # crop arbitrarily to square aspect ratio
        mindim = min(im.shape)
        cropped = im[:mindim,:mindim]

        # make thumbnail
        # thumbnail = skimage.transform.resize(cropped, (thumbsize,thumbsize), order=1, mode='constant')
        thumbnail = resize(cropped, (thumbsize,thumbsize), order=1, mode='constant')

        # convert to float+color -- float enables use of marker value (-1)
        thumbnail = skimage.img_as_float(thumbnail)
        thumbnail = skimage.color.gray2rgb(thumbnail)

        # add a colored border
        bordercolor = bordercolors[ids]
        thumbnail = np.lib.pad(thumbnail, ((bordersize,bordersize), (bordersize,bordersize), (0,0)),
                               'constant', constant_values=(-1,-1))
        thumbnail[thumbnail[:,:,0] == -1] = bordercolor

        # map position to image coordinates with buffer region
        # x,y = np.round(pos/scale * ((mapsize-(thumbsize+3+bordersize))/2) + (mapsize/2)).astype(int)
        x,y = np.round(pos/scale * ((mapsize-(thumbsize+10+bordersize))/2) + (mapsize/2)).astype(int)
        x = mapsize-x # image convention -- match scatter plot
        # place thumbnail into image map
        if verbose:
            print(thumbnail.shape)
            print(halfthumbsize*2)
            print('({},{})'.format(x,y))
        imagemap[x-(halfthumbsize):x+(halfthumbsize),y-(halfthumbsize):y+(halfthumbsize),:] = thumbnail

    return imagemap

def kmeans(X, n_clusters=6, n_iter=100):
    # start n_clusters random data points as cluster centers 
    init_idx = np.random.choice(X.shape[0], size=n_clusters) 
    centers = X[init_idx]
    for t in range(n_iter):
        # compute cluster membership
        # cdist computes the distance between each pair of points
        # between the two input arrays
        labels = np.argmin(cdist(X, centers), axis=1)
    
        # update each cluster center to centroid of cluster members
        for label in range(n_clusters):
            indicator = (labels == label)
            centers[label] = np.sum(X[indicator], axis=0) / np.sum(indicator)
    return labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
