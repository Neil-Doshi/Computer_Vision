
import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_extractor(img1,img2): # finding keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1,kp2,des1,des2

def keypoint_matcher(des1,des2): # my logic for finding which keypoints match from both images
    matches = []
    for i in range(des1.shape[0]):
        thresh = 1000
        for j in range(des2.shape[0]):
            some_value = des1[i] - des2[j]
            some_value = np.linalg.norm(some_value)
            if thresh > some_value:
                min = some_value
                thresh = some_value
                jj = j
                ii = i
        matches.append((min,ii,jj))
    matches.sort()
    return matches

def get_imaginary_frames(img):
    frame = np.float32(np.array([ [0,0], [0,img.shape[1]], [img.shape[0], img.shape[1]], [img.shape[0],0] ])).reshape((-1,1,2))
    return frame
    

def Calculate_orientation_rotation(keypoint1,keypoint2,img1,img2): # This function finds how to images will be stitched i.e. orienation and translation
    src_pts = keypoint1
    dst_pts = keypoint2
    h, _  = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    imaginary_frame1 = get_imaginary_frames(img1)
    # imaginary frame 1 it just takes the corner of image 1 and makes a imaginary rectangle
    imaginary_frame2 = get_imaginary_frames(img2)
    # imaginary frame 2 it just takes the corner of image 2 and makes a imaginary rectangle
    orient_imaginary_frame2 = cv2.perspectiveTransform(imaginary_frame2, h)
    # orients image 2 w.r.t to image 1
    combine_frame = np.concatenate((imaginary_frame1, orient_imaginary_frame2), axis=0)
    # merges both the imaginary frame
    start_x, start_y = np.int32(combine_frame.min(axis=0).ravel())
    # finds the starting points of second frame after merging
    tanslation_matrix_frame2 = np.array([[1, 0, -start_x], [0, 1, -start_y], [0, 0, 1]])
    # using the formula given in class to make the image matrix with x and y position 
    desired_frame = tanslation_matrix_frame2.dot(h)
    # translates the second image so that it aligns with first image
    return desired_frame,start_y,start_x

def merge_images(img1,img2): # simpy pasting the image over the oriented image

    
    kp1,kp2,des1,des2 = feature_extractor(img1,img2)
    matches = keypoint_matcher(des1,des2)
    matches = matches[0:350]
    i = 0
    src_pts = []
    dst_pts = []
    for i in range(len(matches)):
        y = matches[i][2]
        x = matches[i][1]
        src_pts.append(kp1[x].pt)
        dst_pts.append(kp2[y].pt)
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    desired_frame,start_y,start_x = Calculate_orientation_rotation(src_pts,dst_pts,img1,img2)
    desired_image = cv2.warpPerspective(img1, desired_frame, (img2.shape[1] + img1.shape[1], img2.shape[0] + img1.shape[0]))
    canvas = np.zeros((desired_image.shape[0],desired_image.shape[1]))
    ogcanvas = np.zeros((desired_image.shape[0],desired_image.shape[1],3))
    imgx = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    canvas[-start_y:img2.shape[0]-start_y, -start_x:img2.shape[1]-start_x] = imgx
    ogcanvas[-start_y:img2.shape[0]-start_y, -start_x:img2.shape[1]-start_x] = img2
    #desired_image = desired_image.astype(int)
    #desired_image = desired_image.astype('uint8')
    #canvas = canvas.astype(int)
    #canvas = canvas.astype('uint8')
    desired_image1 = cv2.cvtColor(desired_image,cv2.COLOR_BGR2GRAY)
    canvas1 = canvas
    desired_image1 = np.where(desired_image1>127, 0, 255)
    canvas1 = np.where(canvas1>40, 0, 255)
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            if canvas1[i][j] == 255  or canvas1[i][j] - desired_image1[i][j] == 0:
                desired_image[i][j] = desired_image[i][j]
            else:
                desired_image[i][j] = ogcanvas[i][j]

    return desired_image

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    desired_image = merge_images(img2,img1)
    cv2.imwrite(savepath, desired_image)
    return

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

