
"""

RVries

This function takes object detection bounding boxes and writes the cropped image parts as separate JPG files.
Ready to paste in your tensorflow script or simply write "from boxcrop import boxcrop"

Inputs:
    frame:              [uint8]      the actual image as numpy array (so far only tested on RGB data) 
    boxes:              [float32]    array with bounding boxes in fractional positions 
    classes:            [float32]    array with classes
    category_index:     [dict]       class number vs textual title
    scores:             [float32]    array with scores
    cutoff_score:       [float]      score below which the cropping will be halted
    targetdir           [str]        destination direcory to write crop images
    prefix:             [str]        file prefix, such as frame[#], or imageid[#]


======================================================
DATE                    CHANGE              WHO
------------------------------------------------------
2019/06/17              First version       RVries

======================================================



"""
import cv2
import numpy as np

def boxcrop(frame, boxes, classes, category_index, scores, cutoff_score, targetdir, prefix):
    
    if cutoff_score is None:        # Defaults to cutoff_score of 80% if no input is given
        cutoff_score        = 0.8
    
    if prefix is None:              # Defaults to simply numbering the exported images by order they are contained
        fileprefix          = ''    
        
    if targetdir is None:           # Defaults to writing the files in the main directory
        targetdir           = ''     
    
        
    ## Export cropped images based on bounding boxes
    xdm                 = frame.shape[1]
    ydm                 = frame.shape[0]
    
    
    # Overlapping boxes detection
    
    
    # Check if there are multiple boxes or just one
    if boxes.ndim == 1:
        
        temp                = boxes
        score               = scores
        
        # cat                 = category_index[int(classes)-1]['name']
 
        cat                 = 'object'
    
        if np.max(score) < cutoff_score:                                                # Break loop if score drops below cutoff
            return
                        
        
        # Compute the box boundaries in image coordinates
        ymin                = int(round(ydm * temp[0]))
        xmin                = int(round(xdm * temp[1]))
        
        ymax                = int(round(ydm * temp[2]))           
        xmax                = int(round(xdm * temp[3]))
        
        # 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
        
        
        # Crop the image by numpy slicing
        crop                = frame[ymin-40:ymax+40, xmin-40:xmax+40, :]
        
        try:
            # Write the cropped area as separate jpg
            cv2.imwrite(targetdir + prefix  + '.jpg', crop)    
        except:
            # Crop the image by numpy slicing
            crop                = frame[ymin:ymax, xmin:xmax, :]
            cv2.imwrite(targetdir + prefix  + '.jpg', crop)    
        
    else:   # Here the above code is just replicated and modified slightly for the case of multiple bounding boxes
        
        # Box outputs
        for i in range(boxes.shape[0]):                                             # Loop through all boxes
            
            temp                = boxes[i]
            score               = scores[i]
            
            
            # print(score)
            # print(temp)
            
            # cat                 = category_index[int(classes[i])-1]['name']
            
            cat                 = 'object'
     
            if np.max(score) < cutoff_score:                                                # Break loop if score drops below cutoff
                break
                             
            
            # Compute the box boundaries in image coordinates
            ymin                = int(round(ydm * temp[0]))
            xmin                = int(round(xdm * temp[1]))
            
            ymax                = int(round(ydm * temp[2]))           
            xmax                = int(round(xdm * temp[3]))
            
            # Crop the image by numpy slicing
            crop                = frame[ymin-40:ymax+40, xmin-40:xmax+40, :]
            
            try:
                # Write the cropped area as separate jpg
                cv2.imwrite(targetdir + prefix  + '.jpg', crop)    
            except:
                # Crop the image by numpy slicing
                crop                = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(targetdir + prefix  + '.jpg', crop)   
        
        return 
    
    

def boxcrop_yolo(frame, boxes, classes, category_index, scores, cutoff_score, targetdir, prefix):
    
    if cutoff_score is None:        # Defaults to cutoff_score of 80% if no input is given
        cutoff_score        = 0.8
    
    if prefix is None:              # Defaults to simply numbering the exported images by order they are contained
        fileprefix          = ''    
        
    if targetdir is None:           # Defaults to writing the files in the main directory
        targetdir           = ''     
    
        
    ## Export cropped images based on bounding boxes
    # xdm                 = frame.shape[1]
    # ydm                 = frame.shape[0]
    
    
    # Overlapping boxes detection
    
    
    # Check if there are multiple boxes or just one
    if boxes.ndim == 1:
        
        temp                = boxes
        score               = scores
        
        # cat                 = category_index[int(classes)-1]['name']
 
        cat                 = 'object'
    
        if np.max(score) < cutoff_score:                                                # Break loop if score drops below cutoff
            return
                        
        
        # Compute the box boundaries in image coordinates
        # ymin                = int(round(ydm * temp[0]))
        # xmin                = int(round(xdm * temp[1]))
        
        # ymax                = int(round(ydm * temp[2]))           
        # xmax                = int(round(xdm * temp[3]))
        
        xmin                = boxes[0]
        ymin                = boxes[1]
        
        xmax                = boxes[2]  
        ymax                = boxes[3]
        
        # print(xmin)
        
        
        # 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
        
        
        # Crop the image by numpy slicing
        crop                = frame[ymin-40:ymax+40, xmin-40:xmax+40, :]
        
        try:
            # Write the cropped area as separate jpg
            cv2.imwrite(targetdir + prefix  + '.jpg', crop)    
        except:
            # Crop the image by numpy slicing
            crop                = frame[ymin:ymax, xmin:xmax, :]
            cv2.imwrite(targetdir + prefix  + '.jpg', crop)    
        
    
        
        return     