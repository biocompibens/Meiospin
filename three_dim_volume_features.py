
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
import os
import PIL
#import Image
from scipy.cluster.vq import *
from scipy.ndimage import measurements as measur
from scipy.ndimage import label, generate_binary_structure
import skimage.io
from scipy.ndimage import morphology as morph
from scipy import misc
import matplotlib.cm as cm
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
from shapely.geometry import shape,mapping,MultiLineString
from shapely.ops import polygonize, unary_union
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation
from skimage import segmentation, color
from skimage.future import graph
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing as preprocessing
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans as KMeans_clustering
import math


def Points_from_Seg(volume_image):
    idx= np.nonzero(volume_image!=0)
    Pts_volume_image= np.transpose(np.array(idx))
    return Pts_volume_image;

def IntensityRegion(Pts_overlap,overlap):
    intensities= overlap[Pts_overlap[:,0], Pts_overlap[:,1],Pts_overlap[:,2]]
    return intensities;

def Intensities_Features (volume_image):
    feat=[]
    Pts_volume_image= Points_from_Seg(volume_image)

    intensities= IntensityRegion(Pts_volume_image,volume_image)
    #mean intensity
    feat.append(np.mean(intensities))

    #median intensity
    feat.append(np.median(intensities))

    #variance intensity
    feat.append(np.var(intensities))

    return feat;

def dist (A,B):
    return math.sqrt( math.pow(A[0]-B[0],2) + math.pow(A[1]-B[1],2) + math.pow(A[2]-B[2],2))

def dist_reel_dim (A,B,pixel_width, pixel_height, pixel_depth):
    p0x= A[0] * pixel_width
    p0y= A[1] * pixel_height
    p0z= A[2] * pixel_depth
    p1x= B[0] * pixel_width
    p1y= B[1] * pixel_height
    p1z= B[2] * pixel_depth
    return math.sqrt( math.pow(p0x-p1x,2) + math.pow(p0y-p1y,2) + math.pow(p0z-p1z,2))

def FeaturesDistancesFromcenterOfMass(volume_image):
    feat= []
    volume_image_copy= volume_image.copy()
    ind_A= [volume_image_copy!=0]
    volume_image_copy[ind_A]= 1
    Cofmass=measur.center_of_mass(volume_image_copy)

    #calculate the distance from centre of the mass and all points in the object (LamineA or ADN)
    Pts_volume_image= Points_from_Seg(volume_image)
    SumDist=0.0
    MaxDist=0.0
    dists= []
    for i in range(1,Pts_volume_image.shape[0]):
        D= dist(Cofmass,Pts_volume_image[i,:])
        if D> MaxDist:
            MaxDist= D
        SumDist= SumDist+D
        dists.append(D)

    feat.append(MaxDist)
    feat.append(SumDist)
    feat.append(np.mean(np.array(dists)))
    feat.append(np.median(np.array(dists)))
    feat.append(np.var(np.array(dists)))

    return feat;


def FeaturesDistancesFromcenterOfMass_reel_dim(volume_image, pixel_width, pixel_height, pixel_depth):
    feat= []
    volume_image_copy= volume_image.copy()
    ind_A= [volume_image_copy!=0]
    volume_image_copy[ind_A]= 1
    Cofmass=measur.center_of_mass(volume_image_copy)

    #calculate the distance from centre of the mass and all points in the object (LamineA or ADN)
    Pts_volume_image= Points_from_Seg(volume_image)
    SumDist=0.0
    MaxDist=0.0
    dists= []
    for i in range(1,Pts_volume_image.shape[0]):
        D= dist_reel_dim(Cofmass,Pts_volume_image[i,:],pixel_width, pixel_height, pixel_depth)
        if D> MaxDist:
            MaxDist= D
        SumDist= SumDist+D
        dists.append(D)

    feat.append(MaxDist)
    feat.append(SumDist)
    feat.append(np.mean(np.array(dists)))
    feat.append(np.median(np.array(dists)))
    feat.append(np.var(np.array(dists)))

    return feat;


def Bounding_Box(LamineA_seg):
  Pts_LamineA_seg= np.transpose(Points_from_Seg(LamineA_seg))
  xmin= np.amin(Pts_LamineA_seg[0])
  xmax= np.amax(Pts_LamineA_seg[0])
  ymin= np.amin(Pts_LamineA_seg[1])
  ymax= np.amax(Pts_LamineA_seg[1])
  zmin= np.amin(Pts_LamineA_seg[2])
  zmax= np.amax(Pts_LamineA_seg[2])
  return {'xmin':xmin, 'xmax':xmax ,'ymin':ymin,'ymax':ymax,'zmin':zmin,'zmax':zmax }


def FeaturesFromBoundingBox(LamineA_seg):
    feat=[]
    ddF= Bounding_Box(LamineA_seg)
    feat.append( ddF['xmax']- ddF['xmin'])
    feat.append( ddF['ymax']- ddF['ymin'])
    feat.append( ddF['zmax']- ddF['zmin'])
    #the volume of the bounding box
    vol = (ddF['xmax']- ddF['xmin'])*(ddF['ymax']- ddF['ymin'])*(ddF['zmax']- ddF['zmin'])
    feat.append(vol)
    #the ratio between the volume of the bounding box and the volume of the object (LamineA or ADN)
    Pts_LamineA_seg= Points_from_Seg(LamineA_seg)
    feat.append(vol/Pts_LamineA_seg.shape[0])
    return feat;

########################################################################################################################
################################ condensation feature using alpha shape algo ###########################################
########################################################################################################################


def add_edge(edges, v1, v2):
    edges.add((min(v1, v2), max(v1, v2)) )
    return ;

# calcul des alpha_shapes
def add_edge(points,edges, edge_points,i, j):
     #Ajoute une ligne entre le i eme point et le j eme points si elle n'est pas dans la liste
     if (i, j) in edges or (j, i) in edges:
        # deja dans la liste
        return
     edges.add( (i, j) )
     edge_points.append(points[ [i, j] ])


def PointsOfSegmentedObject_2(data):
    idx= np.nonzero(data!=0)
    idx2= np.transpose(np.array(idx))
    # fileN= filename + TER
    # pd.DataFrame(idx2).to_csv('fileN',index=False,sep=' ')
    return idx2;

def calc_alpha2(points,tri,alpha):
     edges = set()
     edge_points = []
     # loop over triangles:
     for ia, ib, ic in tri.vertices:
         # extraction des points de Delaunay
         pa = points[ia]
         pb = points[ib]
         pc = points[ic]
         # Longueurs des cotes du triangle
         a = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
         b = np.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
         c = np.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
         # Semiperimetre du triangle
         s = (a + b + c)/2.0
         # Surface du triangle par la formule de Heron
         area = np.sqrt(s*(s-a)*(s-b)*(s-c))
         # rayon de filtrage
         circum_r = a*b*c/(4.0*area)
         if circum_r < alpha:
            add_edge(points, edges,edge_points, ia, ib)
            add_edge(points, edges,edge_points,ib, ic)
            add_edge(points, edges,edge_points,ic, ia)
     return edge_points


def Condensation_feature (obj):
    w, d, h = obj.shape
    sum=0;
        # alpha-shape in each layer of the stack
    for i in range(0,w-1):
        Im= obj[i,:,:]
        points= np.array([])
        points= PointsOfSegmentedObject_2(Im)
        if points.shape[0]>100:
            # #plot(points[:,0], points[:,1], 'ro')
            # # Delaunay
            tri = Delaunay(np.array(points))

            # plt.figure()
            # plt.gca().set_aspect('equal')
            # plt.triplot(slice[:,0], slice[:,1], tri.vertices, 'go-')

            for alpha in np.arange(1,400,5):

                #The MultiLineString constructor takes a sequence of line-like sequences or objects
                m = MultiLineString(calc_alpha2(points,tri,alpha))
                triangles = list(polygonize(m))
                if triangles:
                    #a representation of the union of triangles
                    a = unary_union(triangles)

                    # bords= np.array(a.exterior.coords)
                    # fig2= plt.figure()
                    # ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
                    # ax2.plot(bords[:,0],bords[:,1],bords[:,2],color='#6699cc', alpha=0.7,linewidth=3, solid_capstyle='round', zorder=2)
                    # selection de la premiere geometrie de type Polygon rencontree
                    if a.geom_type == "Polygon":
                            # print alpha
                            #  #plot the polygon
                            # fig= plt.figure()
                            # ax = fig.add_subplot(111)
                            # plot(points[:,0], points[:,1], 'ro')
                            # plt.hold(True)
                            x,y= a.exterior.xy
                            sum+= points.shape[0]/a.area
                            # ax.plot(x, y, color='#6699cc', alpha=0.7,linewidth=3, solid_capstyle='round', zorder=2)
                            # ax.set_title('the alpha_shape with alpha='+ str(alpha) + ' on the slice='+ str(i))
                            # plt.hold(False)
                            break;
                    else:
                            continue

    #plt.show()
    return sum;

########################################################################################################################
########################################################################################################################
def remove_small_noise(obj):
    #obj= segmentation_KMEANS_Intensity(obj2)
    w, d, h = obj.shape
    AA= obj.copy()
    idx= np.nonzero(AA!=0)
    AA[idx]=1

    s = generate_binary_structure(3,3)
    AA= ndim.binary_opening(AA, structure=s).astype(np.int)
    idx= np.nonzero(AA==0)
    obj[idx]=0
    return obj;

def convexHullDraw(data,showConvexHull):
    hull = ConvexHull(data)
    if showConvexHull:
        #plot convex hull
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2],c='r',marker='^')
        ax.plot_trisurf(data[:,0], data[:,1], data[:,2],triangles=hull.simplices, linewidth=0.2,shade=False, color=(0,0,0,0), edgecolor='Gray')
        plt.show()
    return hull;


def volume_ellipse_pca(eig_val_cov, pixel_width, pixel_height, pixel_depth):
    volume_ellipse= (4/3)*math.pi*math.sqrt(eig_val_cov[0])*2*math.sqrt(eig_val_cov[1])*2*math.sqrt(eig_val_cov[2])*2*pixel_width*pixel_height*pixel_depth
    return volume_ellipse;

def fit_ellipsoid_pca(data, pixel_width, pixel_height, pixel_depth, showEllip):
    eig_val_cov,eig_vec_co= np.linalg.eigh(np.dot(data.T, data)/data.shape[0])
    rx= 2*math.sqrt(eig_val_cov[0])
    ry= 2*math.sqrt(eig_val_cov[1])
    rz= 2*math.sqrt(eig_val_cov[2])

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    if showEllip:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=(0,0,0,0), edgecolor='Gray')
        ax.scatter(data[:,0], data[:,1], data[:,2],c='r',marker='^')
        plt.show()

    volume_ellipse= (4/3)*math.pi*math.sqrt(eig_val_cov[0])*2*math.sqrt(eig_val_cov[1])*2*math.sqrt(eig_val_cov[2])*2*pixel_width*pixel_height*pixel_depth
    return volume_ellipse;

def Shape_Factor(Longest, Intermediate, Shortest):
    return Shortest/ math.sqrt(Intermediate * Longest);

def Aspect_Ratio(Longest, Intermediate, Shortest):
    return Longest/Intermediate;

def Sphericity(Longest, Intermediate, Shortest):
    return pow((Intermediate*Shortest)/pow(Longest,2),  1.0/3.0);

def Flatness(Longest, Intermediate, Shortest):
    return Shortest/Intermediate;


def Shape_and_PCA_features(X):
    feat= []
    # The points cloud of the object
    points= Points_from_Seg(X)
    # PCA
    data= preprocessing.scale(np.array(points,dtype=float64),with_mean=True, with_std=False)
    eig_val_cov,eig_vec_co= np.linalg.eigh(np.dot(data.T, data)/data.shape[0])

    #descending sort of the eigenvalues
    eig_val_cov= np.array(eig_val_cov)
    eig_val_cov= -np.sort(-eig_val_cov)

    # les features
    feat.append(eig_val_cov[0])
    feat.append(eig_val_cov[1])
    feat.append(eig_val_cov[2])
    feat.append(eig_val_cov[0]* eig_val_cov[1]* eig_val_cov[2])
    feat.append(np.mean(eig_val_cov))
    feat.append(Shape_Factor(eig_val_cov[0],eig_val_cov[1],eig_val_cov[2]))
    feat.append(Aspect_Ratio(eig_val_cov[0],eig_val_cov[1],eig_val_cov[2]))
    feat.append(Sphericity(eig_val_cov[0],eig_val_cov[1],eig_val_cov[2]))
    feat.append(Flatness(eig_val_cov[0],eig_val_cov[1],eig_val_cov[2]))

    return feat;

def analysis_connected_components(LamineA_seg):
    feat=[]
    Lamine= LamineA_seg.copy()
    ind_A=  np.nonzero(Lamine!=0)
    Lamine[ind_A]= 1
    s = generate_binary_structure(3,3)
    label_im, nb_labels = label(Lamine,structure=s)
    label_im= np.array(label_im)

    #the size of regions
    sizes = ndim.sum(Lamine, label_im, range(1,nb_labels + 1))

    #the number of regions
    feat.append(sizes.shape[0])
    #the biggest region
    feat.append(np.amax(sizes))

    feat.append(np.var(sizes))
    feat.append(np.mean(sizes))
    feat.append(np.median(sizes))

    small_Reg= ind_A[0].__len__()/nb_labels

    # delete small regions
    idx= sizes<small_Reg
    sizes[idx]=0
    sizes_2= np.trim_zeros(sizes)

    #the number of considerable regions
    feat.append(sizes_2.shape[0])
    feat.append(np.var(sizes_2))
    feat.append(np.mean(sizes_2))
    feat.append(np.median(sizes_2))

    #the mean intensities of the regions
    mean_vals = ndim.sum(LamineA_seg, label_im, range(1,nb_labels + 1))
    mean_vals[idx]=0
    mean_vals_2= np.trim_zeros(mean_vals)

    feat.append(np.amax(mean_vals_2))

    feat.append(np.var(mean_vals_2))
    feat.append(np.mean(mean_vals_2))
    feat.append(np.median(mean_vals_2))

    #the variance of intensities of the regions
    var_vals = ndim.variance(LamineA_seg, label_im, range(1,nb_labels + 1))
    var_vals[idx]=0
    var_vals_2= np.trim_zeros(var_vals)
    feat.append(np.amax(var_vals_2))

    feat.append(np.var(var_vals_2))
    feat.append(np.mean(var_vals_2))
    feat.append(np.median(var_vals_2))

    return feat;

########################################################################################################################
######################################## Features from the region of overlap ###########################################
########################################################################################################################

def FeaturesOverlapRegion(CC_LamineA_seg,CC_ADN_seg, LamineA, ADN):
    Features =[]
    #The volume of the overlaped region
    overlap= OverlapRegionStack(CC_LamineA_seg,CC_ADN_seg,LamineA)
    Pts_overlap= Points_from_Seg(overlap)
    intensities= IntensityRegion(Pts_overlap,overlap)
    Features.append(Pts_overlap.shape[0])

    #Percentage of the overlaped region comparing to the LamineA
    Pts_LamineA_seg= Points_from_Seg(CC_LamineA_seg)
    Features.append(Pts_LamineA_seg.shape[0]/Pts_overlap.shape[0])

    #Percentage of the overlaped region comparing to the ADN
    Pts_ADN_seg= Points_from_Seg(CC_ADN_seg)
    Features.append(Pts_ADN_seg.shape[0]/Pts_overlap.shape[0])

    #intensities features of the overlaped region
    #mean intensity
    Features.append(np.mean(intensities))

    #median intensity
    Features.append(np.median(intensities))

    #variance intensity
    Features.append(np.var(intensities))

    overlap2= OverlapRegionStack(CC_LamineA_seg,CC_ADN_seg,ADN)
    Pts_overlap2= Points_from_Seg(overlap2)
    intensities2= IntensityRegion(Pts_overlap2,overlap2)

    #mean intensity
    Features.append(np.mean(intensities2))

    #median intensity
    Features.append(np.median(intensities2))

    #variance intensity
    Features.append(np.var(intensities2))

    return Features;


def OverlapRegionStack(LamineA_seg,ADN_seg,LamineA):
    #Lumina Extraction
    Lamine = np.array(LamineA).copy()
    LamineA_seg_c= LamineA_seg.copy()
    ADN_seg_C= ADN_seg.copy()

    ind_A= [LamineA_seg_c!=0]
    LamineA_seg_c[ind_A]= 1
    ind_B= [ADN_seg_C!=0]
    ADN_seg_C[ind_B]= 1
    overlap= np.array(LamineA_seg_c!=0) & np.array(ADN_seg_C!=0)
    idx= [overlap==0]
    Lamine[idx]=0
    return Lamine;

########################################################################################################################
########################################### relational features ########################################################
########################################################################################################################
def CenterOfMassFromNdarray(A):
    A_lab= A.copy()
    ind_A= [A_lab!=0]
    A_lab[ind_A]= 1
    Cofmass=measur.center_of_mass(A_lab)
    return Cofmass;

def DistancesFromPointsToListOfPoints(POINT, A):
    dists= []
    Pts_A= Points_from_Seg(A)
    dists= []
    for i in range(1,Pts_A.shape[0]):
        D= dist(POINT,Pts_A[i,:])
        dists.append(D)

    return dists;

def FeaturesDistancesFromCenterOfMassOfObject1(Object1, Object2):
    Cofmass= CenterOfMassFromNdarray(Object1)
    feat= []
    #calculate the distance from centre of the mass and all points in the object1
    dists1= DistancesFromPointsToListOfPoints(Cofmass, Object1)

    #Calculate the distance from the center of mass and all the points of Object2
    dists2= DistancesFromPointsToListOfPoints(Cofmass, Object2)

    feat.append(np.abs(np.mean(np.array(dists1)) - np.mean(np.array(dists2))))
    feat.append(np.var(np.array(dists1))/np.var(np.array(dists2)))

    mn= np.minimum(np.amin(dists1),np.amin(dists2))
    mx= np.maximum(np.amax(dists1),np.amax(dists2))
    #the euclidean distance between the normalized histograms of the two sets of distances
    hist1,bin_edges1= np.histogram(dists1, bins=10, range=(mn,mx))
    hist2,bin_edges2= np.histogram(dists2, bins=10, range=(mn,mx))
    feat.append(Distance.euclidean(hist1,hist2))
    #the Manhattan distance between the normalized histograms
    feat.append(Distance.cityblock(hist1,hist2))

    #the T-test of the two distributions, return The calculated t-statistic and pvalue
    # If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores.
    # If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
    tstat,pvalue= stats.ttest_ind(dists1,dists2)
    feat.append(pvalue)
    return feat;

def FeaturesDistancesFromCenterOfMassOfConcatObject_1_and_2(Object1, Object2):
    feat=[]

    #Center of mass of new object1
    Cofmass1= CenterOfMassFromNdarray(Object1)

    #Center of mass of new object1
    Cofmass2= CenterOfMassFromNdarray(Object2)

    Cofmass= [(Cofmass1[0]+Cofmass2[0])/2,(Cofmass1[1]+Cofmass2[1])/2, (Cofmass1[2]+Cofmass2[2])/2]

    #calculate the distance from centre of the mass and all points in the object1
    dists1= DistancesFromPointsToListOfPoints(Cofmass, Object1)

    #Calculate the distance from the center of mass and all the points of Object2
    dists2= DistancesFromPointsToListOfPoints(Cofmass, Object2)

    feat.append(np.abs(np.mean(np.array(dists1)) - np.mean(np.array(dists2))))
    feat.append(np.var(np.array(dists1))/np.var(np.array(dists2)))

    feat.append(dist(Cofmass1,Cofmass2))

    mn= np.minimum(np.amin(dists1),np.amin(dists2))
    mx= np.maximum(np.amax(dists1),np.amax(dists2))
    #the euclidean distance between the normalized histograms of the two sets of distances
    hist1,bin_edges1= np.histogram(dists1, bins=10, range=(mn,mx))
    hist2,bin_edges2= np.histogram(dists2, bins=10, range=(mn,mx))
    feat.append(Distance.euclidean(hist1,hist2))
    #the Manhattan distance between the normalized histograms
    feat.append(Distance.cityblock(hist1,hist2))

    #the T-test of the two distributions, return The calculated t-statistic and pvalue
    # If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores.
    # If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
    tstat,pvalue= stats.ttest_ind(dists1,dists2)
    feat.append(pvalue)
    return feat;

# def Volume_features (data):
#     #from mayavi import mlab
#     from skimage.measure import marching_cubes, mesh_surface_area
#     verts, faces = marching_cubes(data, 0.0, (0.0800884, 0.0800884, 0.0791103))
#
#     scalars = np.random.random(verts.shape)
#
#     # The TVTK dataset.
#     mesh = tvtk.PolyData(points=verts, polys=faces)
#     mesh.point_data.scalars = scalars
#     mesh.point_data.scalars.name = 'scalars'
#     mass = tvtk.MassProperties(input=mesh)
#
#     area= mesh_surface_area(verts, faces)
#     # mlab.triangular_mesh([vert[0] for vert in verts],
#     #                      [vert[1] for vert in verts],
#     #                      [vert[2] for vert in verts],
#     #                      faces)
#     # mlab.show()
#     Condensation1= float(mass._get_volume())/verts.shape[0]
#     Condensation2 = float(mass._get_surface_area())/verts.shape[0]
#     from scipy.spatial import ConvexHull
#     hull = ConvexHull(verts)
#     area_convex_hull= hull.area
#     volume_convex_hull= hull.volume
#     shape = float(mass._get_surface_area())**2/float(mass._get_volume())
#     concavity= area_convex_hull - float(mass._get_surface_area())
#     convexity= volume_convex_hull/float(mass._get_volume())
#     PROP= list()
#     PROP.append(float(mass._get_volume()))
#     PROP.append(float(mass._get_surface_area()))
#     PROP.append(float(mass._get_normalized_shape_index()))
#     PROP.append(Condensation1)
#     PROP.append(Condensation2)
#     PROP.append(area_convex_hull)
#     PROP.append(volume_convex_hull)
#     PROP.append(shape)
#     PROP.append(concavity)
#     PROP.append(convexity)
#     return PROP;
