__author__ = 'othmani'


### Read the data in a numpy 3D array ##########################################
import numpy as np
import skimage.io
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.vq import *
from sklearn.cluster import KMeans as KMeans_clustering

################################################################################
# Superpixel segmentation and selection of the cluster with the highest intensity
################################################################################
def Mean_Intensity_of_cluster(im, im_seg, cl):
    idx= np.nonzero(im_seg==cl)
    image= im[idx[0],idx[1],idx[2]]
    m= np.mean(image)
    return m;

def segmentation(im,nColors,compact):

    #Lumina Extraction
    Lamine = np.array(im)
    #Lamine= Lamine[10:20,256:768,256:768]
    w, d, h = Lamine.shape

    # nombre de classes
    Best_int= 0

    # computing K-Means with K = 3 (3 clusters)
    LamineA_seg_F= slic(Lamine, n_segments=nColors, compactness=compact,multichannel=False)
    LamineA_cl= -1

    # find the id of the cluster with the hightest intensity, it correspond to the Lamine A
    for k in range(0,nColors):
        m=Mean_Intensity_of_cluster(im, LamineA_seg_F, k)
        if m>Best_int:
            LamineA_cl=k
            Best_int= m


    condRes = (LamineA_seg_F!=LamineA_cl)
    im[condRes] = 0
    LamineA_seg_F= im

    return LamineA_seg_F;

################################################################################
# kmeans segmentation and selection of the cluster with the highest intensity
################################################################################

def Segmentation_Kmeans_1(im,nColors ):
    Lam = np.array(im)
    w, d, h = Lam.shape

    # computing K-Means with K =  (n clusters)
    samples = np.reshape(Lam, (w*d*h,1))

    clf = KMeans_clustering(k=nColors)
    labels = clf.fit(samples)
    AAA= np.reshape(clf.labels_,(w,d,h))

    Best_int= 0
    LamineA_cl=-1
    # find the id of the cluster with the LOWEST intensity, it correspond to the BACKGROUND
    for k in range(0,nColors):
        m= Mean_Intensity_of_cluster(Lam, AAA, k)

        if m>Best_int:
            LamineA_cl=k
            Best_int= m

    condRes = (AAA!=LamineA_cl)
    Lam[condRes] = 0

    return Lam;

def Segmentation_Kmeans_2(im, filename, TER):

    #Lumina Extraction
    Lamine = np.array(im)
    #Lamine= Lamine[10:20,256:768,256:768]
    LamineA = Lamine
    w, d, h = LamineA.shape
    Lumina= np.reshape(LamineA,(w*d*h,1))

    # nombre de classes
    nColors=3

    # computing K-Means with K = 3 (3 clusters)
    centroids,_ = kmeans(Lumina,nColors,100,1e-05)

    # assign each sample to a cluster
    idx,_ = vq(Lumina,centroids)

    LamineA_seg = np.reshape(idx,(w,d,h))
    LamineA_seg_F= np.zeros((w, d, h), dtype='uint8')

    # find the id of the cluster with the hightest intensity, it correspond to the Lamine A
    for i in range(0,w-1):
        sl_seg= LamineA_seg[i,:,:]
        Best_int=0
        sl = LamineA[i,:,:]
        sl_org= Lamine[i,:,:]
        for k in range(0,nColors):

            m= Mean_Intensity_of_cluster(sl, sl_seg, k)
            #print ('la classe'+ str(k) +'a une moyenne de '+ str(m))
            if m>Best_int:
                 LamineA_cl=k
                 Best_int= m
        # #the segmented stack image of the Lamine A
        condRes = (sl_seg!=LamineA_cl)
        sl_org[condRes] = 0
        LamineA_seg_F[i,:,:]= sl_org


    return LamineA_seg_F;



