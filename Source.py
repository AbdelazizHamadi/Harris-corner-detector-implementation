import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os
import random

def non_max_suppresion(harris_image, win_size = 51, stride = 31):
    
    #to move by int (3/2) = 1 
    offset = 31 // 2
    # k value 
    #k = 0.04
    # create empty image
    output_img = np.zeros((harris_image.shape[0], harris_image.shape[1]), dtype = "float32")

    ### suppression non maximal 
    for y in tqdm(range(offset, harris_image.shape[0] - offset, 31)):
        for x in range(offset, harris_image.shape[1] - offset, 31):
            
            temp = np.zeros((31, 31))
            current_window = harris_image[y-offset:y+offset+1, x-offset: x+offset+1]
            #current_window[current_window < np.max(current_window)] = 0
            
            coords = np.array(np.where(current_window == np.max(current_window)))
            maxi = np.max(current_window.flatten())
            temp[coords[0, 0], coords[1, 0]] = maxi
          
            output_img[y-offset:y+offset+1, x-offset: x+offset+1] = temp
    
    # get image points 
    points = np.array(np.where(output_img > 0)).T
    
    
    return output_img, points
    

def harris_response(image, win_size = 3):
    
    #derivative 
    a = np.array([[-1,0,1]]).T
    #axe
    b = np.array([[1,1,1]]).T
    abT = a*b.T
    baT = b*a.T

 
    harris_image_x = cv.filter2D(image, cv.CV_32F, abT)
    harris_image_y = cv.filter2D(image, cv.CV_32F, baT)
    
    

    # calculating Ix**2, Iy **2 and IxIy
    Ix = harris_image_x * harris_image_x
    Iy = harris_image_y * harris_image_y
    IxIy = harris_image_x * harris_image_y

    # smoothing the results with gaussian filter with (3,3) window and sigma = 2 (already in OpenCV library)
    Ixg = np.array(cv.GaussianBlur(Ix, (win_size, win_size), 2))
    Iyg = np.array(cv.GaussianBlur(Iy, (win_size, win_size), 2))
    IxIyg = np.array(cv.GaussianBlur(IxIy, (win_size, win_size), 2))
    
    #harris respsonse calculation 
    
    # list to store the x and y of every r calculated
    corner_list =[]

    #to move by int (3/2) = 1 
    offset = win_size // 2
    # k value 
    k = 0.04
    # create empty image
    output_img = np.zeros((image.shape[0], image.shape[1]), dtype = "float32")
    
    
    for y in tqdm(range(offset, image.shape[0] - offset)):
        for x in range(offset, image.shape[1] - offset):
            #set window coordinates 
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            #get the corresponding area 
            windowIx = Ixg[start_y : end_y, start_x : end_x]
            windowIxIy = IxIyg[start_y : end_y, start_x : end_x]
            windowIy = Iyg[start_y : end_y, start_x : end_x]
            
            # calculate sum of M foreach pixel  
            Sxx = windowIx.sum()
            Sxy = windowIxIy.sum() 
            Syy = windowIy.sum()  
          
            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy 
            
            #Calculate r for Harris Corner equation
            r = det - (k*(trace**2))
            #r = np.linalg.det(c) - (k * (np.trace(c)) **2)
            
            output_img[y, x] = r
        
            
    
    
    # filter out best points 
    thresh = 0.01 * max(output_img.flatten())
    points = []

    
    for y in tqdm(range(offset, output_img.shape[0] - offset)):
        for x in range(offset, output_img.shape[1] - offset):
            
            if output_img[y, x] > thresh * 0.000000000000001:
                
                points.append([x, y])
    
    points = np.array(np.where(output_img != 0))

    
    return output_img, points


target = cv.imread('geo.png', 0)

test = cv.imread('geo_rotated.png', 0)

# 
harris_target, harris_points = harris_response(target)
harris_test, harris_points2 = harris_response(test)

#%% Show results 

plt.imshow(harris_target, cmap='gray')
plt.scatter(harris_points[1, :], harris_points[0, :])
plt.title('shapes points before n_m suppression ')

plt.show()


plt.imshow(harris_test, cmap='gray')
plt.scatter(harris_points2[1, :], harris_points2[0, :])
plt.title('shapes points before n_m suppression ')

plt.show()

#%% Show results with non-max supression

nm_harris_target, points = non_max_suppresion(harris_target)
harris_test, points2 = non_max_suppresion(harris_test)


plt.imshow(target, cmap='gray')
plt.scatter(points[:, 1], points[:, 0])
plt.title('shapes points after n_m suppression ')

plt.show()

plt.imshow(test, cmap='gray')
plt.scatter(points2[:, 1], points2[:, 0])
plt.title('shapes points after n_m suppression ')

plt.show()


