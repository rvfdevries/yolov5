# -*- coding: utf-8 -*-
"""

This module is a toolbox for projecting an image to  
real-world coordinates. It also contains functions to export the image as a 
geotiff, and to detect camera roll and pitch from the horizon in images.

"""


import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit
from numba import cuda
import time
import scipy.linalg
import rasterio
import cv2
import utm


def cam_gopro6(config):

    wsens           = config["Wsens"]
    hsens           = config["Hsens"]
    xpx             = config["xpx"]  
    ypx             = config["ypx"] 
    feq             = config["feq"]  
    cu              = config["cu"]  
    cv              = config["cv"]  

    fscale          = np.sqrt( np.square(wsens) + np.square(hsens)) / 0.035
    freal           = feq * fscale  
    wpix            = wsens/xpx
    hpix            = hsens/ypx
    fxpx            = freal / wpix
    fypx            = freal / hpix
    
    # This is the internal camera calibration matrix:
    K               = np.array([[fxpx, 0, cu], [0, fypx, cv], [0, 0, 1]])

    return K


def peak_votes(accumulator, thetas, rhos):
    """ Finds the max number of votes in the hough accumulator """
    idx = np.argmax(accumulator)
    rho = rhos[int(idx / accumulator.shape[1])]
    theta = thetas[idx % accumulator.shape[1]]

    return idx, theta, rho


def write_geotiff(filename, img, x,y, xx,yy, uvec, vvec, width, height, utmzone):
    
    xmin                = np.min(x)
    xmax                = np.max(x)
    ymin                = np.min(y)
    ymax                = np.max(y)
    
    # print(xmin)
    # print(ymin)
    
    aspect              = (xmax - xmin)/(ymax - ymin)
    
    print('Image height before:' + str(height))
    
    height              = int(np.ceil(width / aspect))
    
    print('IMage height before:' + str(height))
    
    x1                  = (x - np.min(x)) * width / (np.max(x) - np.min(x))
    y1                  = -(y - np.max(y)) * height / (np.max(y) - np.min(y))
    
    
    _interval           = int(x1.shape[0] / 100)
    
#    print(_interval)
    
#    mat                 = cv2.findHomography(np.float32(np.hstack((uvec[::_interval], vvec[::_interval]))), np.transpose(np.float32(np.vstack((x1[::_interval], y1[::_interval])))))[0]

    mat                 = cv2.findHomography(np.float32(np.hstack((uvec[::100000], vvec[::100000]))), np.transpose(np.float32(np.vstack((x1[::100000], y1[::100000])))))[0]

#    mat                 = cv2.findHomography(np.float32(np.hstack((uvec, vvec))), np.transpose(np.float32(np.vstack((x1, y1)))))[0]


#    warped              = cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]))
    warped              = cv2.warpPerspective(img, mat, (width, height))
    warped2             = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # fixme: get the real coordinates
    
    
#    cv2.imwrite('testttt.jpg', warped2)
    
    transform           = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                            height = warped2.shape[0], width = warped2.shape[1],
                            count=3, dtype=str(warped2.dtype),
#                            crs='+proj=utm +zone=10 +ellps=GRS80 +datum=WGS84 +units=m +no_defs',
                            crs='+proj=utm +zone=' +str(utmzone[2])+ '+ellps=GRS80 +datum=WGS84 +units=m +no_defs',
                            transform=transform)
    
    _out                = warped2.transpose(2, 0, 1)
    
    new_dataset.write(_out)
    
    new_dataset.close()
    
    return warped2

@jit(nopython = True)
def Solve(A, y):
    
#    x          = scipy.linalg.lstsq(A, y)[0]
#    x          = np.linalg.lstsq(A, y)[0]
    x          = np.linalg.inv(A).dot(y)        # This is 1 second faster
    
    return x

@jit
def fillgrid(nrow, ncol, xi, yi, x, y, z, binsize, grid, bins, wherebin, retloc, retbin):
    
        # fill in the grid.
    for row in range(nrow):
        
        for col in range(ncol):
            
#            print(row)
#            print(col)
            
            
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

#            print(xc)
#            print(yc)

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            
#            print(posx)
#            print(posy)
            
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]
            
#            print(np.sum(ibin))
#            print(ind)

            # fill the bin.
            binn = z[ibin]
            
#            print(binn)
#            if retloc: wherebin[row][col] = ind
#            if retbin: bins[row, col] = binn.size
            if binn.size != 0:
#                print('proceeding')
                binval         = np.median(binn)
#                print(row, col, binval)
                grid[row, col] = binval
#                if row < 6 and col < 6:
#                    print(grid[:5,:5])
#            else:
#                grid[row, col] = np.nan   # fill empty bins with nans.
#                
    return grid, bins, wherebin

#@jit(nopython = True)
def griddata(x, y, z, binsize=0.01, retbin=True, retloc=True):
    # Don't use this!
    
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).
    
    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.
   
    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xi      = np.arange(xmin, xmax+binsize, binsize)
    yi      = np.arange(ymin, ymax+binsize, binsize)
    
    xi, yi = np.meshgrid(xi,yi)

    # make the grid.
    grid           = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin: bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    grid, bins, wherebin = fillgrid(nrow, ncol, xi, yi, x, y, z, binsize, grid, bins, wherebin, retloc, retbin)

#    # fill in the grid.
#    for row in range(nrow):
#        
#        for col in range(ncol):
#            
##            print(row)
##            print(col)
#            
#            
#            xc = xi[row, col]    # x coordinate.
#            yc = yi[row, col]    # y coordinate.
#
##            print(xc)
##            print(yc)
#
#            # find the position that xc and yc correspond to.
#            posx = np.abs(x - xc)
#            posy = np.abs(y - yc)
#            
##            print(posx)
##            print(posy)
#            
#            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
#            ind  = np.where(ibin == True)[0]
#            
##            print(np.sum(ibin))
##            print(ind)
#
#            # fill the bin.
#            binn = z[ibin]
#            
##            print(binn)
#            if retloc: wherebin[row][col] = ind
#            if retbin: bins[row, col] = binn.size
#            if binn.size != 0:
##                print('proceeding')
#                binval         = np.median(binn)
##                print(row, col, binval)
#                grid[row, col] = binval
##                if row < 6 and col < 6:
##                    print(grid[:5,:5])
##            else:
##                grid[row, col] = np.nan   # fill empty bins with nans.
                
    return grid, bins, wherebin
#    # return the grid
#    if retbin:
#        if retloc:
#            return grid, bins, wherebin
#        else:
#            return grid, bins
#    else:
#        if retloc:
#            return grid, wherebin
#        else:
#            return grid

def Image2World(g, uu,vv,H,inc,roll,azi,K,xcenter,ycenter):

    # Image2World Project points in image pixel coordinates to world
    # coordinates.
    #   [x,y] = Image2World(x,y,H,inc,roll,azi,K) performs the projection
    #   (sometimes called perspective tranformation or orthography) from points
    #   at image coordinates (u,v) to world coordinates (x,y,-H) for a camera
    #   positioned at (0,0,0) and oriented with incidence, inc, roll, roll, and
    #   azimuth, azi, (in radians). inc, roll, azi, and H are scalars.  K is
    #   the 3x3 upper-triangular camera intrinsic matrix. u and v can be
    #   scalars, vectors, or arrays, and must be of the same size. Outputs x
    #   and y are the same size as u and v.
    #   
    #   Written by Michael Schwendeman, June 2014
    #
    #   Citation: Schwendeman, M., J. Thomson, 2014: "A Horizon-tracking Method
    #   for Shipboard Video Stabilization and Rectification."  In Review, J.
    #   Atmos. Ocean. Tech.
    #
    #
    #   Modified by Robin de Vries the Ocean Cleanup for Python 3.5+

#    u = np.array([0])
#    v = np.array([0])
    
#    start1              = time.time()

    R_roll      = np.array([[np.cos(roll), -np.sin(roll), 0],[ np.sin(roll), np.cos(roll), 0],[ 0, 0, 1]])
    
    R_pitch     = np.array([[1, 0, 0],[ 0, -np.cos(inc), -np.sin(inc)],[ 0, np.sin(inc), -np.cos(inc)]])
    
#    R_azi       = np.array([[np.cos(azi), 0, -np.sin(azi)],[0, 1, 0],[ np.sin(azi), 0, np.cos(azi)]])
    
#    R           = R_azi.dot(R_roll.dot(R_pitch))
    R           = R_roll.dot(R_pitch)
    
    Imat            = np.zeros((4,4))
    Imat[0:3,0:3]   = K
    Imat[3,3]       = 1
    
    Rmat            = np.zeros((4,4))
    Rmat[0:3,0:3]   = R
    Rmat[3,3]       = 1
    
    P           = Imat.dot(Rmat)
    

                

    ## DETECT WHETHER INPUT IS SCALAR, VECTOR OR MATRIX
    uvec        = np.reshape(uu, [uu.shape[0] * uu.shape[1], 1], order = 'F')
    vvec        = np.reshape(vv, [vv.shape[0] * vv.shape[1], 1], order = 'F')

      
    
    n_obs       = np.shape(uvec)[0]    
    obs         = np.zeros((n_obs, 4))
    
    ## VECTOR OR SCALAR NOTATION
    obs[:,0]    = np.transpose(uvec)
    obs[:,1]    = np.transpose(vvec)
    obs[:,2]    = 1
    obs[:,3]    = 1
    
    obs         = np.transpose(obs)
    
#    start1      = time.time()    
    

#    pw          = np.linalg.lstsq(P, obs, rcond = None)[0]
#    pw          = scipy.linalg.lstsq(P, obs)[0]
#    pw          = np.linalg.inv(P).dot(obs)
    pw          = Solve(P, obs)
#    pw           = np.linalg.solve(P, obs)

#    end1        = time.time()
#    
#    print(str(end1 - start1))
    
    

    x           = -pw[0,:] / pw[2,:]*H + xcenter
    y           = -pw[1,:] / pw[2,:]*H + ycenter
    
    deci        = 20
    
    
#    xplot       = x[0:12000000:deci]
#    yplot       = y[0:12000000:deci]
    
    gvec        = np.reshape(g, [g.shape[0] * uu.shape[1], 1], order = 'F')
    gplot       = gvec[0:12000000:deci]
    gplot       = np.reshape(gplot, (np.shape(gplot)[0]))
    
#    plt.figure()
#    plt.scatter(xplot, yplot, 1, gplot, cmap=plt.cm.Greys)
#    plt.show()
#    #
    
    xx          = np.reshape(x,np.shape(uu), order = 'F')
    yy          = np.reshape(y,np.shape(vv), order = 'F')
    

#    print('IMAGEPROJECTION TEST')
#    print('IMAGEPROJECTION TEST')

    return x,y, xx, yy, uvec, vvec


@jit(nopython = True)
def Horizon2Angles(theta, rho, K):
    
    ## Adapted by Robin de Vries, The Ocean Cleanup
    ## (Based on Michael Schewndeman, June 2014, toolbox)

    theta       = np.array([theta])
    rho         = np.array([rho])
    
    fx          = K[0,0] 
    fy          = K[1,1] 
    cx          = K[0,2] 
    cy          = K[1,2]

    roll        = np.arctan(-fx / (fy * np.tan(theta)))
    
    inc         = np.arctan((fx*np.sin(roll) * np.cos(theta) - fy*np.cos(roll) * np.sin(theta)) / (rho - cx*np.cos(theta) - cy * np.sin(theta)))
    inc[inc<=0] = np.pi +inc[inc<=0]
    
    return inc, roll

@jit(nopython = True)
def hough_local_maxima(accumulator, thetas, rhos):
    
    thetalook   = 3
    rholook     = 10
    
    maxvals     = []
    alltemps    = []
    
    for rho in range(rholook, accumulator.shape[0], rholook*2):
#    for rho in np.linspace()
        
        for theta in range(thetalook, accumulator.shape[1], thetalook*2):
            
            ymin    = rho - rholook
            ymax    = rho + rholook
            xmin    = theta - thetalook
            xmax    = theta + thetalook
            
            # in this case you will look beyond the array limits
            if xmin < 0 or ymin < 0 or xmax > accumulator.shape[1] or ymax > accumulator.shape[0]:
                
                continue
            
            temp    = accumulator[ymin:ymax,xmin:xmax]
            tempmax = np.max(temp)

#            print(temp)

#            cv2.imshow('', temp)
#            time.sleep(5)
            

            maxvals.append(tempmax)
            alltemps.append(temp)
    
#    maxes, _       = np.unique(np.array(maxvals), return_inverse = True)
    maxes     = np.array(maxvals)
#    maxes       = maxvals
    
    return maxes, alltemps

@jit(nopython = True)    
def hough_robust(accumulator):
    scores  = []
    temp    = accumulator.flatten()
    
    # First, check if multiple values for maximum
    
    srtd    = temp.argsort()[::-1][:2]
    
    cntr    = 0  
    
    for ix in srtd:
        
#        thetas[cntr]    = 
                
        scores.append(temp[ix])
        
        cntr            = cntr+1
        
    if scores[0] - scores[1] < 100:
        print(scores[0] - scores[1])
        return False
    else:
        return True
        
#    return rhos, thetas, scores

def theta2gradient(theta):
    return np.cos(theta) / np.sin(theta)

def rho2intercept(theta, rho):
    return rho / np.sin(theta)

@jit(nopython = True)
def hough_line(img, angle_step=1, lines_are_white=False, value_threshold=150):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(70, 110, angle_step))
#    thetas = np.deg2rad(np.arange(-90, 90, angle_step))
#    thetas = np.deg2rad(np.arange(0, 180, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint16)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # FIXME paralellize...
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos



def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

