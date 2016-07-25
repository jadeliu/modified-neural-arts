import cv2
import numpy as np
import scipy as sp
import os
import itertools
from get_match import *

# function to get most similar style patch 
def get_similar_patch(sty, con, thres=0.8):
    sty_grey = cv2.cvtColor(sty, cv2.COLOR_BGR2GRAY)
    con_grey = cv2.cvtColor(con, cv2.COLOR_BGR2GRAY)
    (kps1, kps2, matches) = match(sty_grey, con_grey)
    if not matches or len(matches)<21:
        print "no or very few matches found between style and content, will not crop"
        return sty
    # get the patch that include threshold percentage of dots
    target = thres*len(matches)
    points = [kps1[matches[i].trainIdx].pt[0] for i in range(len(matches))]
    min_area = 0
    t_x1 = 0
    t_y1 = 0
    t_x2 = 0
    t_y2 = 0
    for subset in itertools.combinations(points, target):
        x1, x2, y1, y2 = polygon_gen(subset)
        area = (x2-x1)*(y2-y1)
        if area<min_area:
            t_x1 = x1
            t_x2 = x2
            t_y1 = y1
            t_y2 = y2
    return sty[x1:x2, y1:y2]

def polygon_gen(points):
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[sp.spatial.ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    x1 = min(rval[:,0])
    x2 = max(rval[:,0])
    y1 = min(rval[:,1])
    y2 = max(rval[:,1])
    return x1, x2, y1, y2

def gen_style_image(sty, con):
    patch = get_similar_patch(sty, con)
    return patch
     
def cal_color_diff(img1, img2, thres=10000):
    a = get_img_hist(img1)
    b = get_img_hist(img2)
    dist = sp.spatial.euclidean(a,b)
    return dist>thres
     
def get_img_hist(img):
    b = img[:,:,0]*0.07
    g = img[:,:,1]*0.72
    r = img[:,:,2]*0.21
    b = cv2.calcHist([b],[0],None,[256],[0,256)
    g = cv2.calcHist([g],[0],None,[256],[0,256])
    r = cv2.calcHist([r],[0],None,[256],[0,256])
    hist = np.hstack((b,g,r))
    return hist

def equalize_img(img):
    channels = cv2.split(imgYCC)
    channels[0] = cv2.equalizeHist(channels[0])
    return cv2.merge(channels)

def main():
    sty = cv2.imread('images/garden.jpg')
    con = cv2.imread('images/source/garden.jpg')
    sty = gen_style_image(sty, con)
    keep_color = 0
    if cal_color_diff(sty, con):
        keep_color = 1.0
    
    sty = cv2.cvtColor(sty, cv2.COLOR_BGR2YCrCb)
    con = cv2.cvtColor(con, cv2.COLOR_BGR2YCrCb)
    sty = equalize_img(sty)
    con = equalize_img(con)

    sty = cv2.cvtColor(sty, cv2.COLOR_YCrCb2BGR)
    con = cv2.cvtColor(con, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite("sty.jpg", sty)
    cv2.imwrite("con.jpg", con)

    cmd = ("th neural_style.lua -style_image sty.jpg -content_image con.jpg -original_colors %d"%keep_color)
    os.system(cmd)
    os.system("rm sty.jpg")
    os.system("rm con.jpg")

if __name__=='__main__':
    main()    
