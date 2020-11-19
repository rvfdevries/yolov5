# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:27:07 2020

@author: RobindeVriesTheOcean
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


import os
# import cv2
import numpy as np
# # import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import sys
from boxcrop import boxcrop_yolo
import pandas as pd
import gopro_get_GPS
from PIL import Image
# from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from datetime import timedelta
from ImageProjection import Image2World, peak_votes, Horizon2Angles, hough_local_maxima, hough_robust, theta2gradient, rho2intercept, hough_line, write_geotiff, cam_gopro6
import utm
import time
from shapely.geometry import Polygon, LineString

# from joblib import Parallel, delayed



def detect(config, save_img=False):
    
    category_index  = {'name':'object'}
    
    K               = cam_gopro6(config)
    
    ##############
    ######### Start of QC init
    ##############
    
    # Read the initialization parameters from the configuration dict
    Hough_QC        = config["Hough_QC"]
    houghQCwritepath \
                    = config["Hough_QC_dir"]
    
    avg_QC          = config["Avg_QC"]
    avgQCwritepath  \
                    = config["Avg_QC_dir"]
                    
    geotiff         = config["Geotiff"]
    geotiff_dir     = config["Geotiff_dir"]
    
    ##########################
    # End of QC init ######### 
    ##########################
    
    ##############
    ######### Start of camera init
    ##############
    
    H               = config["camheight"]    
    azi             = config["azi"]
               
    scale           = config["hough_scale"]   
    
    
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


    ##########################
    # End of camera init #####
    ##########################
   
    # Some standard Yolov5 code here below:   
    
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # Add this serial counter, to increment by one for each image with more than zero detections
    serial      = 0
    write_flag  = False
    
    rows_list   = []
    object_list = []
    object_id   = 0
    
    counter     = 0
    boxcount    = 0
    
    for path, img, im0s, vid_cap in dataset:
    # input_dir   = r'C:\Users\RobindeVriesTheOcean\Dropbox (The Ocean Cleanup)\Ocean Research\Robin\GoPro NN\data\NPM3\Originals_Portside'    
            
    # for root, dirs, files in os.walk(input_dir):
        # for filename in files:
            # if filename.endswith((".jpg", ".JPG")):
                
                
                # IMAGE_NAME          = filename
            
                # Path to image
                # PATH_TO_IMAGE       = os.path.join(input_dir,root,IMAGE_NAME)
                
                # image                = cv2.imread(path)

                p, s, im0           = Path(path), '', im0s
                
                filename            = p.name

                # RESET TEMPORARY VARS
                incacc              = 0
                rollacc             = 0
                hortotal            = 0
                    
                hasseenfiles        = False
        
                iloc                = 0 

                
                
                coords              = gopro_get_GPS.gopro_get_GPS(path)   
                
                counter+=1
                
                ####### CUSTOM PART FOR HORIZON DETECTION 
                
                image               = cv2.imread(path)
                image2              = cv2.imread(path)
                # image_expanded      = np.expand_dims(image, axis=0)

                ##########
                ##########
                ##########
                ##########
                
                
                
                
                hsv                 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    
                # Read and split image channels for 'background' image
                b,g,r               = cv2.split(image)      
                
                gray                = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # FIXME: make more compact
                # Here is a tentative piece of code
                dim                 = (round(image.shape[1] * scale), round(image.shape[0] * scale))

                small               = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
                                
                small               = small[0:round(0.4*dim[0]), :]
                
                edged               = cv2.adaptiveThreshold(small,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                                            cv2.THRESH_BINARY,11,2)

                if config["FixedCamOri"] == False:
                    # START OF PITCH AND ROLL ESTIMATION
                    # FIXME: PUT ALL OF THIS IN A FUNCTION?
                    accumulator, thetas, rhos   = hough_line(edged)
    
                    maxes, alltemps             = hough_local_maxima(accumulator, thetas, rhos)
     
                    srtd                        = maxes.argsort()[::-1][:10]
    
                    
                    isize               = edged.shape[1]
                    
                    maxcoords           = []
                    
                    ## Make this a function 'test_line_robustness'
                    for value in maxes[srtd[0:2]]:
                    
                        ixtemp              = np.argwhere(value == accumulator)
                        
                        theta               = thetas[ixtemp[0][1]]
                        rho                 = rhos[ixtemp[0][0]]
                        
                        maxcoords.append(ixtemp)
                        
                        aa                  = np.cos(theta)
                        bb                  = np.sin(theta)
                        x0                  = aa*rho
                        y0                  = bb*rho
                        x1                  = int(x0 + isize*(-bb))
                        y1                  = int(y0 + isize*(aa))
                        x2                  = int(x0 - isize*(-bb))
                        y2                  = int(y0 - isize*(aa))
                        
    #                    cv2.line(small,(x1,y1),(x2,y2),(0,0,255),2)
                        
                    for value in maxes[srtd[0:1]]:
                    
                        ixtemp              = np.argwhere(value == accumulator)
                        
                        theta               = thetas[ixtemp[0][1]]
                        rho                 = rhos[ixtemp[0][0]]
                        
    #                    maxcoords.append(ixtemp)
                        
                        aa                  = np.cos(theta)
                        bb                  = np.sin(theta)
                        x0                  = aa*rho
                        y0                  = bb*rho
                        x1                  = int(x0 + isize*(-bb))
                        y1                  = int(y0 + isize*(aa))
                        x2                  = int(x0 - isize*(-bb))
                        y2                  = int(y0 - isize*(aa))
                        
                        cv2.line(small,(x1,y1),(x2,y2),(0,0,255),2)                    
    #                cv2.imwrite(['QC/'+ filename[:-4]+'_hough.jpg'], small)      
                        
                    # Check the virtual matrix between the two maximums found in the accumulator array...
                    dist            = np.sqrt( np.square(maxcoords[1][0][0] - maxcoords[0][0][0]) + np.square(maxcoords[1][0][1] - maxcoords[0][0][1]))
                       
                        
                    if (np.sum(maxes == maxes[srtd[0]]) > 1 or (maxes[srtd[0]] - maxes[srtd[1]] < 75)) and dist > 15:
                        print([filename + ': Ambiguous horizon... skipping to next image'])
                        
                        if Hough_QC:
                            cv2.putText(small, 'Bad Horizon', (100,100), font, 3, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.imwrite(os.path.join(houghQCwritepath, os.path.basename(root) + "-" + filename[:-4]+'_hough_bad.jpg'), small)
                        
                        
    #                    cv2.imwrite('test.jpg', small)
                        # end = time.time() 
                        # print("Elapsed in this iteration = %s" % (end - start))
    #                    total -= 1
                        continue 
                    
                    if Hough_QC:
                        cv2.imwrite(os.path.join(houghQCwritepath, os.path.basename(root) + "-" +  filename[:-4]+'_hough.jpg'), small)
                
                ## Make this a function 'get_horizon_theta_rho' 
                    ixhor               = np.argwhere(maxes[srtd[0]] == accumulator)
                    theta               = thetas[ixhor[0][1]]
                    rho                 = rhos[ixhor[0][0]]
                    
                    # The radius was estimated based on the downsampled image
                    # We can simply convert it to the full scale by dividing by the scale number
                
                    rho                 = rho / scale
                    inc, roll           = Horizon2Angles(theta, rho, K)
                    
                    incacc, rollacc     = incacc + inc, rollacc + roll
    
                    ## END OF PITCH AND ROLL ESTIMATION
    
                    ## Here we calculate image geometry:
                    u                   = np.array([range(1, xpx + 1)])    # Note that this is about real image coordinates, we choose to only have non-zero coordinates....
                    
                    v_hori              = (rho-u*np.cos(theta))/np.sin(theta)
                    
                    hortrunc            = int(np.ceil(np.max(v_hori))) + 100
                    
                    v                   = np.array([range(hortrunc, ypx + 1)])                    
                    
                else:
                    
                    inc, roll           = np.deg2rad(config["CamPitch"]), np.deg2rad(config["CamRoll"])
                                
                    
                    ## Here we calculate image geometry:
                    u                   = np.array([range(1, xpx + 1)])    # Note that this is about real image coordinates, we choose to only have non-zero coordinates....
                    
                    hortrunc            = 1
                    
                    v                   = np.array([range(1, ypx + 1)])    
                
                    incacc, rollacc     = incacc + inc, rollacc + roll
                
                #################################
                #################################
                # End of pitch & roll estimation
                #################################                                           
                #################################
            
                #################################
                #################################
                # Start of image projection step 
                #################################                                           
                #################################
                
                uu, vv              = np.meshgrid(u,v)
                   
                utmzone             = utm.from_latlon(coords[0], coords[1])
                
                x,y, xx,yy, uvec, vvec  = Image2World(gray, uu,vv,H,inc,roll,azi,K,utmzone[0], utmzone[1])
           
                print(np.mean(xx)) 
                print(np.mean(yy)) 
           
                # Quick-n-dirty area calculation            
                ydiff               = np.abs(np.diff(yy, 1, 0))
                xdiff               = np.abs(np.diff(xx, 1, 1))

                # Add one pseudo-row and -column to make it work with original image dimensions
                xdiff               = np.append(xdiff, np.reshape(xdiff[:,xpx - 2], (ypx-hortrunc+1, 1)), axis = 1)
                ydiff               = np.append(ydiff, np.reshape(ydiff[ypx-hortrunc - 2,:], (1, xpx)), axis = 0)
                
                areas               = xdiff * ydiff
                
                fullarea            = np.sum(areas)
                
                # hsv_clipped         = hsv[hortrunc::, : , :]
                # areas_clipped       = areas                
                       
                
                #################################
                #################################
                # End of image projection step 
                #################################                                           
                #################################                
                                    
                    
                # print(coords)
                
                # Padded resize
                # img = letterbox(img0, new_shape=self.img_size)[0]           
                
                
                
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                    
                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
            
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()
            
                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)
                        
                    
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = Path(path), '', im0s
            
                    # print(p.name)
            
                    save_path = str(save_dir) + '/' + str(serial) + str(p.name)
                    txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    
                    if len(det):
                        write_flag  = True
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
                        # # Get the bounding box as numpy array from the GPU memory
                        # print(det[:, :4].cpu().numpy())
            
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string
            
                        # Write results and perform distance metrics
                        
                        # ORIGINAL CODE ----
                        # # for *xyxy, conf, cls in reversed(det):
                        # #     if save_txt:  # Write to file
                        # #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # #         line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        # #         with open(txt_path + '.txt', 'a') as f:
                        # #             f.write(('%g ' * len(line) + '\n') % line)
            
                        # #     if save_img or view_img:  # Add bbox to image
                        # #         label = '%s %.2f' % (names[int(cls)], conf)
                        # #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # ------        
                                
                        boxcount = 0
                        
                        for *xyxy, conf, cls in reversed(det):
                            
                            object_id+=1
                            
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line) + '\n') % line)
            
                            if save_img or view_img:  # Add bbox to image
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)                        

                                boxcount            += 1
                                
                                # prefix              = filename[:-4] + '_' + str(boxcount)
                                prefix              = str(object_id)
                                                                
                                # print(np.array(xyxy.cpu().numpy()))
                                
                                box                 = torch.stack(xyxy, dim=0).cpu().numpy()
                                box                 = box.astype(int)
                                
                                # Write snipped
                                boxcrop_yolo(im0, box, cls.cpu().numpy(), 0, conf.cpu().numpy(), 0.1, config['qc_dir'], prefix)
                                
                                # DISTANCE METRICS PART
                                txmin                = box[0]
                                tymin                = box[1]
                                txmax                = box[2]  
                                tymax                = box[3]
                                
                                # Construct list of coordinates in good order
                                bbu                 = np.array([txmin, txmax, txmax, txmin])
                                bbv                 = np.array([tymax, tymax, tymin, tymin])
                                                                  
                                                    
                                # OPTION 1: use the image2world function to project the box points
                                boxu, boxv          = np.meshgrid(np.array([txmin, txmax]),np.array([tymin, tymax]))
                                dummy               = boxu
                                      
                                bx,by, bxx,byy, buvec, bvvec  = Image2World(dummy,boxu,boxv,H,inc,roll,azi,K,utmzone[0], utmzone[1])
                                
                                # Box center local coordinates
                                xmean               = np.mean(bxx.flatten())
                                ymean               = np.mean(byy.flatten())           
                                
                                # Box center perpendicular distance from vessel
                                bperp               = np.sqrt(np.square((xmean - utmzone[0])) + np.square((ymean - utmzone[1])))
                                
                                # Object area
                                positions           = np.transpose(np.vstack([bxx.ravel(), byy.ravel()]))
                                
                                # Make the point order right
                                positions[[-2, -1]] = positions[[-1, -2]]
            
                                polygon = Polygon(positions)                                                            
                                
                                # Area
                                boxarea             = polygon.area
                                
                                # Major and minor axes of the bounding box
                                # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
                                mbr_points          = list(zip(*polygon.minimum_rotated_rectangle.exterior.coords.xy))
                                
                                # calculate the length of each side of the minimum bounding rectangle
                                mbr_lengths         = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]
                                
                                # get major/minor axis measurements
                                minor_axis          = min(mbr_lengths)
                                major_axis          = max(mbr_lengths)
                                
                                
                                ## Create one data record per box
                                dict2               = {}
                                
                                dict2.update({
                                              'OID'             : object_id,                           
                                              'file'            : filename,
                                              'path'            : path,                                               
                                              'latitude'        : coords[0], 
                                              'longitude'       : coords[1], 
                                              'timestamp'       : coords[2],   
                                              'scanArea'        : fullarea,
                                              'pitch'           : np.real(np.rad2deg(inc[0])),
                                              'roll'            : np.real(np.rad2deg(roll[0])),                                  
                                              'x'               : xmean, 
                                              'y'               : ymean, 
                                              'area'            : polygon.area,
                                              'amaj'            : major_axis,
                                              'amin'            : minor_axis,
                                              'dist'            : bperp,
                                              'conf'            : conf.cpu().numpy(),
                                              'class'           : cls.cpu().numpy(),
                                              'UTMNO'           : utmzone[2],
                                              'UTMLE'           : utmzone[3]
                                              })
                                              
            
                                
                                object_list.append(dict2)
                                
                    hist,bins = np.histogram(gray.flatten(),256,[0,256])
                                    
                    dict1               = {}
                    dict1.update({'ID':counter, 
                                  'Filename'        : filename, 
                                  'path'            : path, 
                                  'Latitude'        : coords[0], 
                                  'Longitude'       : coords[1], 
                                  'Timestamp'       : coords[2], 
                                  'nDetect'         : boxcount,
                                  'ScanArea'        : fullarea,
                                  'Pitch'           : np.real(np.rad2deg(inc[0])),
                                  'Roll'            : np.real(np.rad2deg(roll[0])),
                                  'Height'          : H,       
                                  'Histogram-top'   : np.sum(hist[253:256]),
                                  'Histogram-low'   : np.sum(hist[0:50]),
                                  'R_avg'           : np.average(r),
                                  'G_avg'           : np.average(g),
                                  'B_avg'           : np.average(b),
                                  'H_avg'           : np.average(hsv[:,:,0]),
                                  'S_avg'           : np.average(hsv[:,:,1]),
                                  'V_avg'           : np.average(hsv[:,:,2]),
                                  'R_min'           : np.min(r),
                                  'G_min'           : np.min(g),
                                  'B_min'           : np.min(b),
                                  'H_min'           : np.min(hsv[:,:,0]),
                                  'S_min'           : np.min(hsv[:,:,1]),
                                  'V_min'           : np.min(hsv[:,:,2]),
                                  'R_max'           : np.max(r),
                                  'G_max'           : np.max(g),
                                  'B_max'           : np.max(b),
                                  'H_max'           : np.max(hsv[:,:,0]),
                                  'S_max'           : np.max(hsv[:,:,1]),
                                  'V_max'           : np.max(hsv[:,:,2])}                                 
                                 )
                    
                    
                    
                    
                    rows_list.append(dict1)                                
                                              
                                
                                
                                
            
                    # Print time (inference + NMS)
                    print('%sDone. (%.3fs)' % (s, t2 - t1))
            
                    # Stream results
                    if view_img:
                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration
            
                # Save results (image with detections)
                # if save_img:
                #     # if dataset.mode == 'images' and pred[0] is not None:
                #     if dataset.mode == 'images' and write_flag:
                #         cv2.imwrite(save_path, im0)
                #         print('Writing image...')
                #         serial+=1
                #         write_flag  = False
                
    dfo         = pd.DataFrame(object_list)          
    df          = pd.DataFrame(rows_list)                
    return df, dfo                


                            
                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    
    config                      = {}
#    config['njobs'] 		= 5
    config['qc_dir']            = r'C:/Users/RobindeVriesTheOcean/Dropbox (The Ocean Cleanup)/Ocean Research/Robin/GoPro NN/data/NPM3/'
    # config['process_dir']        = r'C:\Users\RobindeVriesTheOcean\Dropbox (The Ocean Cleanup)\Ocean Research\Robin\ML Training_Sets\Aerial_expedition_labeling_candidates'
    config['process_dir']      = r'C:\Users\RobindeVriesTheOcean\Dropbox (The Ocean Cleanup)\Ocean Research\Robin\GoPro NN\data\NPM3\Originals_Portside'

    
    config["Hough_QC"]      = False
    config["Hough_QC_dir"]  = r'C:\Users\RobindeVriesTheOcean\Dropbox (The Ocean Cleanup)\Ocean Research\Robin\Python Repo\QC'
    config["Avg_QC"]        = False
    config["Avg_QC_dir"]    = r'C:\Users\RobindeVriesTheOcean\Dropbox (The Ocean Cleanup)\Ocean Research\Robin\Python Repo\avg'
    config["Geotiff"]       = False
    config["Geotiff_dir"]   = r'G:\NPM3_Geotiff'
    # config["Crop_dir"]      = r'C:\Users\RobindeVriesTheOcean\Dropbox (The Ocean Cleanup)\Ocean Research\Robin\GoPro NN\data\gtqc'
    
    
    # Camera external parameters
    config["camheight"]     = 18.3      # Maersk Transporter Bridge deck Camera height [meters]
#    config["camheight"]     = 18.3      # Maersk Transporter Bridge deck Camera height [meters]
#    config["camheight"]     = 12.6      # Maersk Transporter C deck Camera height [meters]

    config["azi"]           = 0         # Camera azimuth [Heading degrees clockwise from North]
    config["FixedCamOri"]   = False     # For imagery without horizon: specify fixed pitch
    config["CamPitch"]      = 0         # Camera Pitch; for when the above is set to true [degrees from nadir]
    config["CamRoll"]       = 0         # Camera Roll; or when the above is set to true [degrees clockwise positive]
    
    # Camera internal parameters
    # This is for GOPRO HERO 6 BLACK, LINEAR FOV:
    config["Wsens"]         = 0.00617   # Sensor width  [meters]
    config["Hsens"]         = 0.00463   # Sensor height [meters]
    config["xpx"]           = 4000      # n image columns 
    config["ypx"]           = 3000      # n image rows
    
    # config["xpx"]           = 800       # n image columns 
    # config["ypx"]           = 1080      # n image rows
    
    config["feq"]           = 0.024     # 35mm-equivalent focal length [meters]
    config["cu"]            = 2000      # image center column
    config["cv"]            = 1500      # image center row 
    
    config["score_tresh"]    = 0.5      # Detection confidence treshold
    
    # Some final processing parameters
    config["hough_scale"]   = 0.2       # Image scale down factor for horizon detection (for speedup. Advise to not go lower than 0.1)
    config["njobs"]         = 6         # Number of parallel processing jobs    
    
    

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                df,dfo = detect(config)
                strip_optimizer(opt.weights)
        else:
            df,dfo = detect(config)


    data = df.to_csv('tester.csv', index = False)

    objects = dfo.to_csv('objects_portside.csv', index = False)
