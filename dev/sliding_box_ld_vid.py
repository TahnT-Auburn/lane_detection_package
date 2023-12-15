import os
import glob

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

# ----- Get Output Layers for YOLO-----
def getOutputLayers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
    return output_layers

# ----- Draw YOLO Predictions ------
def drawPredictions(frame, class_id, confidence, x, y, x_plus_w, y_plus_h):
    
    #Classes path
    classes = '/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/yolo_toolbox/classes/coco.names'
    with open(classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    #Draw bounding box and class name
    label = str(classes[class_id])
    color = (0,255,65)
    if(label == 'car' or label == 'truck' or label == 'bus'): #Take only vehicle detections
        cv.rectangle(frame, (x,y), (x_plus_w, y_plus_h), color, 1)
    #cv.putText(frame, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
# ------ Run YOLO -----
def runYolo(frame):
    
    #Classes path
    classes = '/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/yolo_toolbox/classes/coco.names'
    with open(classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        
    #Image specs
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    #Read in Darknet
    weights = '/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/yolo_toolbox/weights/yolov3.weights'
    config = '/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/yolo_toolbox/cfg/yolov3.cfg'
    net = cv.dnn.readNet(weights,config)
    
    #Create a 4D blob
    blob = cv.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), True, False)
    
    #Sets input to network
    net.setInput(blob)
    
    #Run forward pass to get output of the ouput layers
    outs = net.forward(getOutputLayers(net))
    
    #Initialize
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.6
    nms_threshold = 0.4
    
    width_vect = []
    bottom_vect = []
    for out in outs:
        for detection in out:
        
            #Parse detections
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            obj_label = str(classes[class_id])
            
            #Filter non-vehicle objects
            if obj_label == 'car' or obj_label == 'truck' or obj_label == 'bus':
                
                if confidence > conf_threshold:
                    
                    #Generate bounding boxes
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    
                    #Append arrays
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x,y,w,h])

                    #Append width and bottom vectors for virtual horizon estiamtion
                    bottom = y + h
                    width_vect.append(w)
                    bottom_vect.append(bottom)
                                    
    #Non-maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        #Draw bounding boxes
        drawPredictions(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    '''
    #Display to check (turn off when processing video)
    cv.imshow("YOLO Detection", frame)
    cv.waitKey(0)
    '''
    
    return indices, boxes, width_vect, bottom_vect

#----- Binary Thresholding -----
def binaryThresholding(frame):
    
    #Set channel limits
    b_thresh = (150, 200)
    l2_thresh = (225, 255)

    #HLS conversion
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    h, l, s = cv.split(hls)

    #LAB conversion
    lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
    l2, a, b = cv.split(lab)

    #LUV conversion
    luv = cv.cvtColor(frame, cv.COLOR_BGR2Luv)
    l3, u, v = cv.split(luv)
    
    #Binary from LAB and LUV
    binary_bl = np.zeros_like(l2)
    binary_bl[
        ((b > b_thresh[0]) & (b <= b_thresh[1])) |
        ((l3 > l2_thresh[0]) & (l3 <= l2_thresh[1]))
    ] = 1

    '''
    #Display to check (turn off when processing video)
    binary_bl_plot = binary_bl * 255
    cv.imshow('Binary BL', binary_bl_plot)
    cv.waitKey(0)
    '''
    
    return binary_bl

#----- Mask Image -----
def maskImage(binary_bl, indices, boxes, frame):
    
    #Initialize mask
    frame_size = frame.shape[::-1][1:] #width by height
    mask = np.zeros(frame.shape[:2], np.uint8) 
    
    #Generate general mask (tol > 0 for trapezoid, tol = 0 for triangle)
    tol = 275 #tunable
    height_scale = 5 #tunable - value is divided by frame height
    mask_points = np.array([
        [frame_size[0]/2 - tol, frame_size[1]/height_scale],   # Top-left corner
        [0, frame_size[1]],                         # Bottom-left corner
        [frame_size[0], frame_size[1]],             # Bottom-right corner
        [frame_size[0]/2 + tol, frame_size[1]/height_scale]    # Top-right corner
    ])
    cv.fillPoly(mask, np.int32([mask_points]), color=(255,255,255))
    
    #Generate mask from YOLO detections
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        detect_points = np.array([
            [x, y],             # Top-left corner
            [x, y + h],         # Bottom-left corner
            [x + w, y + h],     # Bottom-right corner
            [x + w, y]          # Top-right corner
        ])
        cv.fillPoly(mask, np.int32([detect_points]), color=(0,0,0))
    
    #Perform masking via bitwise_and
    masked_binary = cv.bitwise_and(binary_bl, binary_bl, mask=mask)

    '''
    #Display to check (turn off when processing video)
    cv.imshow('Mask', mask)
    cv.waitKey(0)
    masked_binary_plot = masked_binary * 255
    cv.imshow('Masked Binary', masked_binary_plot)
    cv.waitKey(0)
    '''
    
    return mask, masked_binary

#----- Inverse Perspective Mapping -----
def invPersTrans(masked_binary, frame):
    
    frame_size = frame.shape[::-1][1:] #width by height
    offset = 570 #tunable
    tol = 75 #tunable
    height_scale = 5
    
    src_points = np.float32([
        (frame_size[0]/2 - tol, frame_size[1]/height_scale),   # Top-left corner
        (0, frame_size[1]),                         # Bottom-left corner
        (frame_size[0], frame_size[1]),             # Bottom-right corner
        (frame_size[0]/2 + tol, frame_size[1]/height_scale)    # Top-right corner
    ])

    dst_points = np.float32([
        [offset, 0], 
        [offset, frame_size[1]],
        [frame_size[0]-offset, frame_size[1]], 
        [frame_size[0]-offset, 0]
    ])

    trans_mat = cv.getPerspectiveTransform(src_points, dst_points)
    inv_trans_mat = cv.getPerspectiveTransform(dst_points, src_points)

    warped_frame = cv.warpPerspective(frame, trans_mat, frame_size, flags=cv.INTER_LINEAR)
    warped_binary = cv.warpPerspective(masked_binary, trans_mat, frame_size, flags=cv.INTER_LINEAR)
    
    '''
    #Display warped raw as a check
    cv.imshow("IPM Frame", warped_frame)
    cv.waitKey(0)
    '''
    
    return warped_binary, inv_trans_mat

#----- Histogram -----
def histogram(warped_binary):
    
    #Generate and plot histogram
    histogram = np.sum(warped_binary[int(warped_binary.shape[0]/2):,:], axis=0)

    #Separate left and right bases/lanes
    midpoint = np.int(histogram.shape[0]/2)
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint

    #Filter low pixel content
    min_pix_count = 10
    if histogram[left_peak] < min_pix_count:
        left_peak = 0
    if histogram[right_peak] < min_pix_count:
        right_peak = 0
        
    '''
    plt.plot(histogram)
    plt.show()
    '''
    
    return left_peak, right_peak

#----- Sliding Window -----
def slidingWindow(warped_binary, left_peak, right_peak):
    
    #Tunable sliding window parameters
    window_num = 12     #number of sliding windows
    margin = 40        #lateral margin 
    min_pix = 50        #minimum pixel count
    
    out_img = np.dstack((warped_binary, warped_binary, warped_binary)) * 255

    #Window height
    window_height = np.int(warped_binary.shape[0]/window_num)
    
    #Get nonzero values in the image
    nonzero = warped_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    #Initialize
    left_lane_inds = []
    right_lane_inds = []
    leftx_current = left_peak
    rightx_current = right_peak
    
    for window in range(window_num):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_binary.shape[0] - (window + 1) * window_height
        win_y_high = warped_binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 1)
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 1)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                           (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_pix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > min_pix:        
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))
    
    #Concatenate array of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]
    
    plot_y = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
    
    #Check if lane vectors are empty
    if not left_x.size > 0 and not left_y.size > 0:
        left_empty = True
    else:
        left_empty = False

    if not right_x.size > 0 and not right_y.size > 0:
        right_empty = True
    else:
        right_empty = False
    
    if left_empty == False:
        #Fit a second order polynomial to each
        left_fit = np.polyfit(left_y, left_x, 2)
        #Generate x and y values for plotting
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255,0,0]
    else:
        #Pass empty vectors
        left_fit_x = []
        
    if right_empty == False:
        right_fit = np.polyfit(right_y, right_x, 2)
        right_fit_x = right_fit[0]*plot_y**2 +right_fit[1]*plot_y + right_fit[2]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0,0,255]
    else:
        #Pass empty vectors
        right_fit_x = []
    
    #Set detection status
    if left_empty == False and right_empty == False:    #Both lanes detected
        detection_status = 0
        approx_lane = 0
    elif left_empty == False and right_empty == True:   #Left only detected
        detection_status = 1
    elif left_empty == True and right_empty == False:   #Right only detected
        detection_status = 2
    elif left_empty == True and right_empty == True:    #No lanes detected
        detection_status = 3
    
        #Construct undetected lane approximation
    approx_pix_off = 120
    if detection_status == 1:   #left only detection
        approx_lane = 2 #indicate the approximated lane (none = 0, left = 1, right = 2)
        for pnt in left_fit_x:
            right_fit_approx = pnt + approx_pix_off
            right_fit_x.append(right_fit_approx)
    if detection_status == 2:   #right only detection
        approx_lane = 1 #indicate the approximated lane (none = 0, left = 1, right = 2)
        for pnt in right_fit_x:
            left_fit_approx = pnt - approx_pix_off
            left_fit_x.append(left_fit_approx)
                
    #Plot
    '''
    f, axarr = plt.subplots(1,1)
    f.set_size_inches(18, 10)
    axarr.imshow(out_img)
    if left_empty == False: 
        axarr.plot(left_fit_x, plot_y, color='yellow')
    if right_empty == False:
        axarr.plot(right_fit_x, plot_y, color='yellow')
    plt.xlim(0,1280)
    plt.ylim(720,0)
    plt.show()
    '''
    
    return detection_status, approx_lane, left_fit_x, right_fit_x, plot_y
    
#----- Project Lane Lines -----
def projectLanes(detection_status, approx_lane, frame, warped_binary, plot_y, left_fit_x, right_fit_x, inv_trans_mat):

    #Process if there are detections
    if detection_status != 3:
        
        #Create canvas image
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_pts = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])    
        pts = np.hstack((left_pts, right_pts))
        left_pts_int = left_pts.astype(np.int32)
        right_pts_int = right_pts.astype(np.int32)
        
        #Draw lane lines and fill lane in warped image
        cv.polylines(color_warp, left_pts_int, False, (0,255,255), 2)
        cv.polylines(color_warp, right_pts_int, False, (0,255,255), 2)
        if approx_lane == 0:    #No approximated lanes
            cv.fillPoly(color_warp, np.int_([pts]), (255,0,0))
        if approx_lane == 1:    #left lane approximated
            cv.fillPoly(color_warp, np.int_([pts]), (0,140,255))
        if approx_lane == 2:    #right lane approximated
            cv.fillPoly(color_warp, np.int_([pts]), (0,140,255))
                
        #Correct warping using inverse transformation matrix
        new_warp = cv.warpPerspective(color_warp, inv_trans_mat, (frame.shape[1], frame.shape[0]))
        lane_frame = cv.addWeighted(frame, 1, new_warp, 0.4, 0)
        
        #Unflipped right points
        right_pts_uf = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))])
        right_pts_uf_int = right_pts_uf.astype(np.int32)
        
        left_bounds = []
        right_bounds = []
        for i in range(len(left_pts_int[0])):
            
            #Lane points in IPM coordinates
            left_pt_ipm = left_pts_int[0,i]
            right_pt_ipm = right_pts_uf_int[0,i]
            
            #Convert lane points in original coordinates
            left_pt_orig_x = round((inv_trans_mat[0,0] * left_pt_ipm[0] + inv_trans_mat[0,1] * left_pt_ipm[1] + inv_trans_mat[0,2]) / (inv_trans_mat[2,0] * left_pt_ipm[0] + inv_trans_mat[2,1] * left_pt_ipm[1] + inv_trans_mat[2,2]))
            left_pt_orig_y = round((inv_trans_mat[1,0] * left_pt_ipm[0] + inv_trans_mat[1,1] * left_pt_ipm[1] + inv_trans_mat[1,2]) / (inv_trans_mat[2,0] * left_pt_ipm[0] + inv_trans_mat[2,1] * left_pt_ipm[1] + inv_trans_mat[2,2]))
            right_pt_orig_x = round((inv_trans_mat[0,0] * right_pt_ipm[0] + inv_trans_mat[0,1] * right_pt_ipm[1] + inv_trans_mat[0,2]) / (inv_trans_mat[2,0] * right_pt_ipm[0] + inv_trans_mat[2,1] * right_pt_ipm[1] + inv_trans_mat[2,2]))
            right_pt_orig_y = round((inv_trans_mat[1,0] * right_pt_ipm[0] + inv_trans_mat[1,1] * right_pt_ipm[1] + inv_trans_mat[1,2]) / (inv_trans_mat[2,0] * right_pt_ipm[0] + inv_trans_mat[2,1] * right_pt_ipm[1] + inv_trans_mat[2,2]))    
            
            left_pt_x_int = left_pt_orig_x.astype(np.int32)
            left_pt_y_int = left_pt_orig_y.astype(np.int32)
            right_pt_x_int = right_pt_orig_x.astype(np.int32)
            right_pt_y_int = right_pt_orig_y.astype(np.int32)
            
            left_pt = (left_pt_x_int, left_pt_y_int)
            right_pt = (right_pt_x_int, right_pt_y_int)

            left_bounds.append(left_pt)
            right_bounds.append(right_pt)
            
        #Draw bounds
        left_bounds_arr = np.array(left_bounds)
        right_bounds_arr = np.array(right_bounds)
        if approx_lane == 0:    #No approximated lanes
            cv.polylines(lane_frame, [left_bounds_arr], False, (0,255,65), 2)
            cv.polylines(lane_frame, [right_bounds_arr], False, (0,255,65), 2)
        if approx_lane == 1:    #Left lane approximated  
            cv.polylines(lane_frame, [left_bounds_arr], False, (0,0,255), 2)
            cv.polylines(lane_frame, [right_bounds_arr], False, (0,255,65), 2)
        if approx_lane == 2:    #Right lane approximated
            cv.polylines(lane_frame, [left_bounds_arr], False, (0,255,65), 2)
            cv.polylines(lane_frame, [right_bounds_arr], False, (0,0,255), 2)
                    
            #Display to check
            #cv.imshow('Lane Detections', lane_frame)
            #cv.waitKey(0)
    else:
        #Pass empty vectors
        left_bounds = []
        right_bounds = []
        
        #return raw frame
        lane_frame = frame.copy()
        #Display to check
       # cv.imshow('Lane Detections', lane_frame)
        #cv.waitKey(0)
        
    return lane_frame, left_bounds, right_bounds

#----- Vanishing Point via Hough Lines -----
def vanishPoint(frame, mask):
    
    #Initialize vanish point
    vert_vanish_point = 0
    
    #Grayscale raw image
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Gaussian smoothing
    kernel_size = 11
    smooth_frame = cv.GaussianBlur(gray_frame, (kernel_size,kernel_size), 0, 0)
    
    #Perform canny edge detection on masked binary
    min_thresh = 60
    max_thresh = 150
    edge_frame = cv.Canny(smooth_frame, min_thresh, max_thresh)
    
    #Mask edge frame
    masked_edge = cv.bitwise_and(edge_frame, edge_frame, mask=mask)
    
    #Perform Hough transform
    rho = 1
    pi = 3.14159265358979323846                     
    theta = pi/180
    threshold = 45
    min_line_length = 25
    max_line_gap = 15
    max_angle = 85
    min_angle = 30
    
    hough_lines = cv.HoughLinesP(masked_edge, rho, theta, threshold, min_line_length, max_line_gap)

    #Process hough lines if they are detected
    line_frame = frame.copy()
    if hough_lines is not None:
        
        #Initialize vectors 
        left_vect = []      #left lane points
        right_vect = []     #right lane points
        x_vect_l = []         #Seperate items for least squares (left)
        y_vect_l =[]
        x_squared_vect_l = []
        xy_vect_l = []
        x_vect_r = []         #Seperate items for least squares (right)
        y_vect_r =[]
        x_squared_vect_r = []
        xy_vect_r = []
        for i in range(0, len(hough_lines)):    #iterate through hough lines output
            
            #Plot points as given (check)
            points = hough_lines[i][0]
            
            #calculate slope
            m = (points[3] - points[1]) / (points[2] - points[0])
            
            #filter out lines with invalid angles
            angle = math.atan(np.abs(m)) * (1/theta)
            if (angle >= min_angle and angle <= max_angle):
                
                #Separate left lane (neg slope - remember y down is +)
                if (m < 0):
                    
                    #Draw points
                    cv.line(line_frame, (points[0], points[1]), (points[2], points[3]), (0,0,255), 3, cv.LINE_AA)
                    cv.circle(line_frame, (points[0], points[1]), 10, (255,255,0),3)
                    cv.circle(line_frame, (points[2], points[3]), 10, (255,255,0),3)
            
                    #append left vector
                    left_vect.append(points)
                    
                    #append items for least squares
                    x1 = points[0]
                    x2 = points[2]
                    y1 = points[1]
                    y2 = points[3]
                    x_squared1 = x1**2
                    x_squared2 = x2**2
                    xy1 = x1*y1
                    xy2 = x2*y2
                    x_vect_l.extend([x1,x2])
                    y_vect_l.extend([y1,y2])
                    x_squared_vect_l.extend([x_squared1,x_squared2])
                    xy_vect_l.extend([xy1,xy2])
                
                #Separate right lane (positive slope)
                if (m > 0):
                    
                    #Draw points
                    cv.line(line_frame, (points[0], points[1]), (points[2], points[3]), (0,0,255), 3, cv.LINE_AA)
                    cv.circle(line_frame, (points[0], points[1]), 10, (128,0,128),3)
                    cv.circle(line_frame, (points[2], points[3]), 10, (128,0,128),3)
                    
                    #append right vector
                    right_vect.append(points)

                    #append items for least squares
                    x1 = points[0]
                    x2 = points[2]
                    y1 = points[1]
                    y2 = points[3]
                    x_squared1 = x1**2
                    x_squared2 = x2**2
                    xy1 = x1*y1
                    xy2 = x2*y2
                    x_vect_r.extend([x1,x2])
                    y_vect_r.extend([y1,y2])
                    x_squared_vect_r.extend([x_squared1,x_squared2])
                    xy_vect_r.extend([xy1,xy2])
                   
        #Check for valid detections for BOTH lanes
        if (np.array(left_vect).size > 0 and np.array(right_vect).size > 0):
            
            #Perform least squares on Hough points
            N_l = len(x_vect_l)
            m_l = (N_l * np.sum(xy_vect_l) - np.sum(x_vect_l) * np.sum(y_vect_l)) / (N_l * np.sum(x_squared_vect_l) - (np.sum(x_vect_l))**2)
            b_l = (np.sum(y_vect_l) - m_l * np.sum(x_vect_l)) / N_l
            N_r = len(x_vect_r)
            m_r = (N_r * np.sum(xy_vect_r) - np.sum(x_vect_r) * np.sum(y_vect_r)) / (N_r * np.sum(x_squared_vect_r) - (np.sum(x_vect_r))**2)
            b_r = (np.sum(y_vect_r) - m_r * np.sum(x_vect_r)) / N_r
            
            #X-point of intersection
            x_intersect = int((b_r - b_l) / (m_l - m_r))
            
            
            line_frame2 = line_frame.copy()
            left_y_vect = []
            right_y_vect = []
            for x in range(0, frame.shape[1]):
                left_y = m_l*x + b_l
                left_y_vect.append(left_y)
                right_y = m_r*x + b_r
                right_y_vect.append(right_y)

                #Generate vanishing point
                if x == x_intersect:
                    vert_vanish_point = int(left_y)

            '''
            #Display
            cv.line(line_frame2, (0,int(left_y_vect[0])), (frame.shape[1],int(left_y_vect[frame.shape[1]-1])),(0,140,255), 3, cv.LINE_AA)
            cv.line(line_frame2, (0,int(right_y_vect[0])), (frame.shape[1],int(right_y_vect[frame.shape[1]-1])),(0,140,255), 3, cv.LINE_AA)
            cv.line(line_frame2, (0,vert_vanish_point), (frame.shape[1], vert_vanish_point), (0,255,255), 3, cv.LINE_AA)
            cv.circle(line_frame2, (x_intersect, vert_vanish_point), 10, (0,0,255), 3)
            cv.imshow('Hough Lines', line_frame)
            cv.waitKey(0)
            cv.imshow('Test', line_frame2)
            cv.waitKey(0)  
            '''
            
        else:
            print('Invalid lane detections')
            vert_vanish_point = 0    
    else:
        print('Invalid hough lines')
        vert_vanish_point = 0
    
    return vert_vanish_point

#----- Virtual Horizon via Vehicle detections ----
def virtHorizon(frame, width_vect, bottom_vect):
    
    #Height of camera (m)
    camera_height = 2.27
    
    #Average vehicle width (m)
    avg_veh_width = 1.82
    
    #Width tolerance (m)
    min_veh_width = 1.4
    max_veh_width = 2.6
    
    #Initial virt horizon calculation
    avg_width = np.sum(width_vect) / len(width_vect)
    avg_bottom = np.sum(bottom_vect) / len(bottom_vect)
    virtual_horizon = avg_bottom - (camera_height * (avg_width / avg_veh_width))

    #Feedback loop
    i = 0
    while i < 5:
        
        #Print virtual horizon
        #print(f'Virtual Horizon: {virtual_horizon}')
        
        #Initialize valid vectors (Reinitialize every feedback loop)
        width_vect_val = []
        bottom_vect_val = []

        #Filter out detections
        for j in range(0, len(bottom_vect)):
            
            lower_lim = ((bottom_vect[j] - virtual_horizon) / camera_height) * min_veh_width
            upper_lim = ((bottom_vect[j] - virtual_horizon) / camera_height) * max_veh_width
            
            if (width_vect[j] >= lower_lim and width_vect[j] <= upper_lim):
                
                #Append valid vectors
                width_vect_val.append(width_vect[j])
                bottom_vect_val.append(bottom_vect[j])
                
        
        #Recalculate virtual horizon
        avg_width = np.sum(width_vect_val) / len(width_vect_val)
        avg_bottom = np.sum(bottom_vect_val) / len(bottom_vect_val)
        virtual_horizon = avg_bottom - (camera_height * (avg_width / avg_veh_width))
        
        #Update bottom and width vectors
        width_vect = width_vect_val
        bottom_vect = bottom_vect_val
        
        i+=1
    
    #Filter invalid virtual horizon values
    if virtual_horizon < 0 or virtual_horizon > frame.shape[0]:
        virtual_horizon = 0
    
    return virtual_horizon
    
#----- Localize Ego Vehicle -----
def localizeEgo(detection_status, lane_frame, left_bounds, right_bounds, vert_vanish_point):
    
    #Camera origin in image plane
    cam_orig = (lane_frame.shape[1]/2, lane_frame.shape[0])
    
    #Camera focal length in pixels
    focal_length = 2.8669527693619584e+03
    
    #Height of camera
    camera_height = 2.27
    
    #Set lanewidth in m (average ~3.7m)
    lane_width_m = 3.7
    
    #Process if detections are available
    if detection_status != 3:
        
        #Generate center of lane
        lane_center_list = []
        for j in range(len(left_bounds)):
            
            left_pt = left_bounds[j]
            right_pt = right_bounds[j]
            
            #lane_width_px = math.dist(left_pt, right_pt)
            lane_width_px = np.abs(left_pt[0] - right_pt[0])
        
            lane_center = (round(lane_width_px / 2 + left_pt[0]), left_pt[1])
            lane_center_list.append(lane_center)
            
            #Generate range to center
            range_est = focal_length * (lane_width_m / lane_width_px)
            
            #Generate lateral offset to lane boundaries
            lat_off_px_c = np.abs(lane_center[0] - cam_orig[0])
            lat_off_px_l = cam_orig[0] - left_pt[0]
            lat_off_px_r = np.abs(cam_orig[0] - right_pt[0])
            
            center_off = lat_off_px_c * range_est / focal_length
            left_off = lat_off_px_l * range_est / focal_length
            right_off = lat_off_px_r * range_est / focal_length
            lane_width_est = left_off + right_off
            
            #Estimate using vanishing point
            range_est_vh = (focal_length * camera_height) / (lane_center[1] - vert_vanish_point)
            center_off_vh = lat_off_px_c * range_est_vh / focal_length
            left_off_vh = lat_off_px_l * range_est_vh / focal_length
            right_off_vh = lat_off_px_r * range_est_vh / focal_length
            lane_width_est_vh = left_off_vh + right_off_vh
            
        #Convert lane center to array
        lane_center_arr = np.array(lane_center_list)
        lane_center_arr = lane_center_arr.astype(np.int32)
        #cv.polylines(lane_frame, [lane_center_arr], False, (0,0,255), 2)
    
        #Show localization info on final image
        center_offset_label = 'Center:' + ' ' + str(round(center_off_vh,3)) + 'm'
        left_offset_label = 'Left:' + ' ' + str(round(left_off_vh,3)) + 'm'
        right_offset_label = 'Right:' + ' ' + str(round(right_off_vh,3)) + 'm'
        cv.putText(lane_frame, center_offset_label, (lane_frame.shape[1] - 250, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)
        cv.putText(lane_frame, left_offset_label, (lane_frame.shape[1] - 250, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)
        cv.putText(lane_frame, right_offset_label, (lane_frame.shape[1] - 250, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)
        
        '''
        #Display localization info from closest lane points
        print('Lat offset:')
        print(f'Center: {lat_off_m_c} m')
        print(f'Left: {lat_off_m_l} m')
        print(f'Right: {lat_off_m_r} m')
        print(f'Lane Width: {lane_width_est}')
        '''
    else:
        #Show localization info on final image
        center_offset_label = 'Center: N/A' 
        left_offset_label = 'Left: N/A'
        right_offset_label = 'Right: N/A'
        cv.putText(lane_frame, center_offset_label, (lane_frame.shape[1] - 250, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)
        cv.putText(lane_frame, left_offset_label, (lane_frame.shape[1] - 250, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)
        cv.putText(lane_frame, right_offset_label, (lane_frame.shape[1] - 250, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1)
               
    return lane_width_est, lane_width_est_vh


# ----- Run -----
def run():
    '''
    #Image test
    #input_path = '/home/tahnt/Documents/opencv_cpp_tutorial/ros2_ws/src/lane_detection_package/data/1280x720/frame1071.jpg'
    input_path = '/home/tahnt/Documents/opencv_cpp_tutorial/ros2_ws/src/lane_detection_package/data/1280x720/frame852.jpg'
    frame = cv.imread(input_path, cv.IMREAD_COLOR)
    
    #Run YOLO
    indices, boxes = runYolo(frame)
    
    #Binary thresholding
    binary_bl = binaryThresholding(frame)
    
    #Mask image
    masked_binary = maskImage(binary_bl, indices, boxes, frame)
    
    #Inverse perspective transform
    warped_binary, inv_trans_mat = invPersTrans(masked_binary, frame)
    
    #Histogram
    left_peak, right_peak = histogram(warped_binary)
    
    #Sliding window
    detection_status, left_fit_x, right_fit_x, plot_y = slidingWindow(warped_binary, left_peak, right_peak)
    
    #Project lane detections to raw frame
    lane_frame, left_bounds, right_bounds = projectLanes(detection_status, frame, warped_binary, plot_y, left_fit_x, right_fit_x, inv_trans_mat)
    
    #Localize ego within lane boundaries
    range_vect, c_off_vect, l_off_vect, r_off_vect = localizeEgo(detection_status, lane_frame, left_bounds, right_bounds)
    
    #Display final output
    
    cv.imshow("Lane Detections", lane_frame)
    cv.waitKey(0)
    '''
    
    #Read video
    vid_path = '/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/lane_detection_package/data/videos/video2.mp4'
    cap = cv.VideoCapture(vid_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    #Initialize video writer
    write_video = False
    output_path = '/home/tahnt/T3_Repos/post_process_packages/ros2_ws/src/lane_detection_package/output/example_video.mp4'
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out_vid = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    #Process
    print_msg = False
    vanish_point_vect = []
    virtual_horizon_vect = []
    lane_width_est_vect = []
    lane_width_est_vh_vect = []
    lane_width_est_vect2 = []
    lane_width_est_vh_vect2 = []
    lane_width_err_vect = []
    lane_width_err_vect2 = []
    while cap.isOpened:
        
        #Read each frame frome video
        ret, frame = cap.read()
        if not ret:
            print('No frame received, shutting down ...')
            break
        frame_cp = frame.copy()
        
        #Run yolo
        indices, boxes, width_vect, bottom_vect = runYolo(frame)
        
        #Binary thresholding
        binary_bl = binaryThresholding(frame)
        
        #Mask image
        mask, masked_binary = maskImage(binary_bl, indices, boxes, frame)
        
        #Inverse perspective transform
        warped_binary, inv_trans_mat = invPersTrans(masked_binary, frame)
        
        #Histogram
        left_peak, right_peak = histogram(warped_binary)
        
        #Sliding window
        detection_status, approx_lane, left_fit_x, right_fit_x, plot_y = slidingWindow(warped_binary, left_peak, right_peak)
        
        #Project lane detections to raw frame
        lane_frame, left_bounds, right_bounds = projectLanes(detection_status, approx_lane, frame, warped_binary, plot_y, left_fit_x, right_fit_x, inv_trans_mat)
        
        #Vanishing Point via Hough lines
        vert_vanish_point = vanishPoint(frame_cp, mask)
        vanish_point_vect.append(vert_vanish_point)
        
        #Virtual Horizon via vehicle detections
        virtual_horizon = virtHorizon(frame, width_vect, bottom_vect)
        virtual_horizon_vect.append(virtual_horizon)
        
        lane_frame2 = lane_frame.copy()
        #Localize ego within lane boundaries (Vanishing point using Hough Lines)
        lane_width_est, lane_width_est_vh = localizeEgo(detection_status, lane_frame, left_bounds, right_bounds, vert_vanish_point)
        lane_width_est_vect.append(lane_width_est)
        lane_width_est_vh_vect.append(lane_width_est_vh)

        #Localize ego within lane boundaries (Virtual Horizon )
        lane_width_est2, lane_width_est_vh2 = localizeEgo(detection_status, lane_frame2, left_bounds, right_bounds, virtual_horizon)
        lane_width_est_vect2.append(lane_width_est2)
        lane_width_est_vh_vect2.append(lane_width_est_vh2)
        
        #Error signals
        lane_width_err = lane_width_est_vh - 3.7
        lane_width_err2 = lane_width_est_vh2 - 3.7
        lane_width_err_vect.append(lane_width_err)
        lane_width_err_vect2.append(lane_width_err2)
        
        #Write video
        if write_video == True:
            if print_msg == False:
                print('Writing video ...')
            print_msg = True
            out_vid.write(lane_frame)
            
        #Display video (optional - for reference)
        cv.imshow('Output', lane_frame)
        if cv.waitKey(1) == 27:
            print('esc key pressed, shutting down ...')
            break
    
    cap.release()
    out_vid.release()
    cv.destroyAllWindows()
    
    #Get averages
    avg_hough_vh = np.sum(vanish_point_vect) / len(vanish_point_vect)
    avg_detect_vh = np.sum(virtual_horizon_vect) / len(virtual_horizon_vect)
    avg_lane_width_hough = np.sum(lane_width_est_vh_vect) / len(lane_width_est_vh_vect)
    avg_lane_width_detect = np.sum(lane_width_est_vh_vect2) / len(lane_width_est_vh_vect2)
    avg_err_hough = np.sum(lane_width_err_vect[14:]) / len(lane_width_err_vect[14:])
    avg_err_detect = np.sum(lane_width_err_vect2[14:]) / len(lane_width_err_vect2[14:])
    
    print('\nHough Line Averages')
    print(f'VH: {avg_hough_vh}')
    print(f'Lane Width: {avg_lane_width_hough}')
    print(f'Error: {avg_err_hough}')
    
    print('\nDetection Averages')
    print(f'VH: : {avg_detect_vh}')
    print(f'Lane Width: {avg_lane_width_detect}')
    print(f'Error: {avg_err_detect}')
    
    #Plot vanishing point 
    x = np.linspace(1, len(vanish_point_vect), len(vanish_point_vect))
    plt.plot(x, 90*np.ones_like(virtual_horizon_vect), label='Reference VH', linestyle='--', color='red')
    plt.plot(x, vanish_point_vect, label = 'Hough lines')
    plt.plot(x, virtual_horizon_vect, label = 'Vehicle Detections', color='orange')
    font = {'family':'serif','color':'black','size':15}
    plt.legend(loc='lower right', fontsize='15')
    plt.title('Virtual Horizon Estimates', fontdict=font)
    plt.xlabel('Frames', fontdict=font)
    plt.ylabel('Image Height [pixels]', fontdict=font)
    plt.ylim([720, 0])
    #plt.show()
    
    #Plot lane width estimates
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(lane_width_est_vect, 'g')
    ax1.plot(lane_width_est_vh_vect)
    ax1.plot(lane_width_est_vh_vect2, color ='orange')
    ax1.legend(['Static Size (Reference)', 'Hough Lines VH', 'Detections VH'], loc='lower right')
    ax1.set_title('Lane Width Estimation', fontdict=font)
    #ax1.set_xlabel('Frames', fontdict=font)
    ax1.set_ylabel('Width [m]', fontdict=font)
    #ax1.set_ylim([0, 5])
    #plt.show()
    
    #Plot lane width error
    ax2.plot(np.zeros_like(lane_width_err_vect), 'g', linestyle='--', linewidth = 1.5, label = 'Reference')
    ax2.plot(lane_width_err_vect, label = 'Hough Lines VH')
    ax2.plot(lane_width_err_vect2, color ='orange', label = 'Detections VH')
    ax2.plot(0.5*np.ones_like(lane_width_err_vect), 'r', linestyle='--', linewidth = 1.5, label = '_nolabel_')
    ax2.plot(-0.5*np.ones_like(lane_width_err_vect), 'r', linestyle='--', linewidth = 1.5, label = '_nolabel_')
    ax2.legend(loc='lower right')
    ax2.set_title('Lane Width Estimation Error', fontdict=font)
    ax2.set_xlabel('Frames', fontdict=font)
    ax2.set_ylabel('Error [m]', fontdict=font)
    ax2.set_ylim([-1, 1])
    plt.show()
    
if __name__ == '__main__':
    print('\nProcessing ...\n')
    run()
    print('\nProcessing completed successfully\n')