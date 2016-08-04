from __future__ import division
import sys

# Setting libraries path so the imports work fine
sys.path.insert(1,'import/bc_users/biocomp/othmani/python_packages_for_condor/Env01/bin/')
sys.path.insert(2,'import/bc_users/biocomp/othmani/python_packages_for_condor/Env01/lib/python2.7/site-packages/')
sys.path.insert(3,'/usr/lib/python2.7/dist-packages/tvtk/')
sys.path.insert(4,'/usr/bin')
sys.path.insert(5,'/usr/local/lib/python2.7/dist-packages/HTSeq-0.6.1p1-py2.7-linux-x86_64.egg')
sys.path.insert(6,'/usr/lib/python2.7')
sys.path.insert(7,'/usr/lib/python2.7/plat-linux2')
sys.path.insert(8,'/usr/lib/python2.7/lib-tk')
sys.path.insert(9,'/usr/lib/python2.7/lib-old')
sys.path.insert(10,'/usr/lib/python2.7/lib-dynload')
sys.path.insert(11,'/users/biocomp/othmani/.local/lib/python2.7/site-packages')
sys.path.insert(12,'/usr/local/lib/python2.7/dist-packages')
sys.path.insert(13,'/usr/lib/python2.7/dist-packages')
sys.path.insert(14,'/usr/lib/python2.7/dist-packages/PIL')
sys.path.insert(15,'/usr/lib/python2.7/dist-packages/gtk-2.0')
sys.path.insert(16,'/usr/lib/pymodules/python2.7')
sys.path.insert(17,'/usr/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode')
sys.path.insert(18,'/usr/lib/python2.7/dist-packages/IPython/extensions')

from Tkinter import *
import tkMessageBox
#import tkFileDialog
import numpy as np
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import PIL
#import Image
from scipy.cluster.vq import *
from scipy.ndimage import measurements as measur
from scipy.ndimage import label, generate_binary_structure
import skimage.io
from scipy.ndimage import morphology as morph
from scipy import misc
import matplotlib.cm as cm

import os.path
import pandas as pd
from pylab import *
import matplotlib.gridspec as gridspec
#import texture_features as tfeat
from scipy import stats
#from scipy.ndimage import morphology as morph
import scipy.ndimage as ndim
import argparse
from scipy.spatial import distance as Distance
from matplotlib.tri import Triangulation
from skimage import segmentation, color
from skimage.future import graph
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing as preprocessing
from scipy.spatial import ConvexHull

import Image_volume_segmentation as seg2
import three_dim_volume_features as features_from_volume

import math


def NbrConnectedComponentFromBinaryVolume(A):
    AA= A.copy()
    ind_A=  np.nonzero(AA!=0)
    AA[ind_A]= 1
    s = generate_binary_structure(3,3)
    label_im, nb_labels = label(AA,structure=s)
    return nb_labels;

def ConnectedComponentFromBinaryVolume(LamineA_seg):
    Lamine= LamineA_seg.copy()
    ind_A=  np.nonzero(Lamine!=0)
    Lamine[ind_A]= 1
    s = generate_binary_structure(3,3)
    label_im, nb_labels = label(Lamine,structure=s)
    label_im= np.array(label_im)
    sizes = ndim.sum(Lamine, label_im, range(nb_labels + 1))
    mask_size= sizes<np.amax(sizes)
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    idx= np.nonzero(label_im!=0)
    label_im[idx]= LamineA_seg[idx]
    return label_im;


def Points_from_Seg(LamineA_seg):
    idx= np.nonzero(LamineA_seg!=0)
    Pts_LamineA_seg= np.transpose(np.array(idx))
    return Pts_LamineA_seg;


def Volume_features(LamineA_seg, ADN_seg,pixel_width, pixel_height, pixel_depth):

    LamineA_seg= features_from_volume.remove_small_noise(LamineA_seg)
    Pts_LamineA_seg= Points_from_Seg(LamineA_seg)
    data_LamineA_seg= preprocessing.scale(np.array(Pts_LamineA_seg,dtype=float64),with_mean=True, with_std=False)
    # volume ellipse fitted using Pca to LamineA
    volume_Ellipse_LamineA= features_from_volume.fit_ellipsoid_pca(data_LamineA_seg, pixel_width, pixel_height, pixel_depth, False)
    # volume convex hull
    hull_LamineA = features_from_volume.convexHullDraw(data_LamineA_seg,False)

       ################### ADN  ########################################

    ADN_seg= features_from_volume.remove_small_noise(ADN_seg)
    Pts_ADN_seg= Points_from_Seg(ADN_seg)
    data_Pts_ADN_seg= preprocessing.scale(np.array(Pts_ADN_seg,dtype=float64),with_mean=True, with_std=False)
    # volume ellipse fitted using Pca to LamineA
    volume_Ellipse_DNA= features_from_volume.fit_ellipsoid_pca(data_Pts_ADN_seg, pixel_width, pixel_height, pixel_depth, False)
    # volume convex hull
    hull_DNA = ConvexHull(data_Pts_ADN_seg)

     ######################## features ######################################

    #number of points
    nbr_Pts_LamineA_seg= Pts_LamineA_seg.shape[0]

    nbr_Pts_ADN_seg= Pts_ADN_seg.shape[0]

    volume_dna= nbr_Pts_ADN_seg*pixel_width*pixel_height*pixel_depth
    volume_LamineA= nbr_Pts_LamineA_seg*pixel_width*pixel_height*pixel_depth

    volume_hull_DNA= hull_DNA.volume*pixel_width*pixel_height*pixel_depth
    area_hull_DNA= hull_DNA.area*pixel_width*pixel_height

    #the volume of the convex hull of the DNA

    volume_hull_LamineA= hull_LamineA.volume*pixel_width*pixel_height*pixel_depth
    area_hull_LamineA= hull_LamineA.area*pixel_width*pixel_height

    # convexity DNA
    convexity_DNA= volume_hull_DNA/volume_Ellipse_DNA
    # convexity LamineA
    convexity_LamineA= volume_hull_LamineA/volume_Ellipse_LamineA

    feat_str= []
    feat_str.append(nbr_Pts_ADN_seg)
    feat_str.append(nbr_Pts_LamineA_seg)
    feat_str.append(nbr_Pts_ADN_seg/nbr_Pts_LamineA_seg)

    feat_str.append(volume_dna)
    feat_str.append(volume_LamineA)
    feat_str.append(volume_dna/volume_LamineA)

    feat_str.append(volume_Ellipse_DNA)
    feat_str.append(volume_Ellipse_LamineA)

    feat_str.append(volume_hull_DNA)
    feat_str.append(area_hull_DNA)
    feat_str.append(volume_hull_LamineA)
    feat_str.append(area_hull_LamineA)

    feat_str.append(convexity_DNA)
    feat_str.append(convexity_LamineA)

    return feat_str;



def features_extraction_Main(file,pixel_width, pixel_height, pixel_depth):

    stack = skimage.io.imread(file, plugin='tifffile')
    LamineA= np.array(stack[:,1,:,:])
    ADN= np.array(stack[:,0,:,:])

    #the best parameter of compactness for the segmentation of the LamineA= 0.005
    LamineA_seg= seg2.segmentation(LamineA, 3, 0.005)
    Intensities_feat_LamineA_seg= features_from_volume.Intensities_Features(LamineA_seg)
    PcaFeatures_LamineA_seg= features_from_volume.Shape_and_PCA_features(LamineA_seg)
    featuresCC_LamineA_seg= features_from_volume.analysis_connected_components(LamineA_seg)

    #the best parameter of compactness for the segmentation of the DNA= 0.01
    ADN_seg= seg2.segmentation(ADN, 3, 0.01)
    Intensities_feat_ADN_seg= features_from_volume.Intensities_Features(ADN_seg)
    CenterOfMassFeatures_ADN_seg= features_from_volume.FeaturesDistancesFromcenterOfMass(ADN_seg)
    PcaFeatures_ADN_seg= features_from_volume.Shape_and_PCA_features(ADN_seg)
    featuresCC_ADN_seg= features_from_volume.analysis_connected_components(ADN_seg)

    Vol_features= Volume_features(LamineA_seg, ADN_seg,pixel_width, pixel_height, pixel_depth)
    featuresOverlapReg= features_from_volume.FeaturesOverlapRegion(LamineA_seg,ADN_seg, LamineA, ADN)

    other_features = []
    # nombre de composantes connexes
    other_features.append(NbrConnectedComponentFromBinaryVolume(LamineA_seg))
    other_features.append(NbrConnectedComponentFromBinaryVolume(ADN_seg))

    diststoLamineA= features_from_volume.FeaturesDistancesFromCenterOfMassOfObject1(LamineA_seg,ADN_seg)
    diststoDNA= features_from_volume.FeaturesDistancesFromCenterOfMassOfObject1(ADN_seg,LamineA_seg)

    distsConcatLamineDNA= features_from_volume.FeaturesDistancesFromCenterOfMassOfConcatObject_1_and_2(ADN_seg,LamineA_seg)



    #outfile.write("{}\t".format(filename))

    feat_str= filename


    for item in Intensities_feat_LamineA_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in PcaFeatures_LamineA_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in featuresCC_LamineA_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)


    for item in Intensities_feat_ADN_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in CenterOfMassFeatures_ADN_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in PcaFeatures_ADN_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in featuresCC_ADN_seg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in Vol_features:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in featuresOverlapReg:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in other_features:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in diststoLamineA:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in diststoDNA:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    for item in distsConcatLamineDNA:
        #outfile.write("{}\t".format(item))
        feat_str= feat_str + " " +str(item)

    #outfile.write (" \n")

    return feat_str;



global stack, LamineA, ADN, filename, LamineA_seg, ADN_seg, files


# Getting command-line options back
parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='file', required=True)
# parser.add_argument('-m', dest='MeRIP_samples', nargs='+', required=True)
# parser.add_argument('-f', dest='Fisher_index') # Fisher values reference
# parser.add_argument('-f', dest='Fisher_index') # Fisher values reference
#to create or tu update
#parser.add_argument('-o', dest='output', required=True)
options = parser.parse_args()

pixel_width= 0.0800884
pixel_height= 0.0800884
pixel_depth= 0.0800884


filename, file_extension = os.path.splitext(options.file)
if file_extension=='.tif':
    feat_str= features_extraction_Main(options.file,pixel_width, pixel_height, pixel_depth)
    print(feat_str)


