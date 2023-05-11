# # 1. Only add your code inside the function (including newly improted packages). 
# #  You can design a new function and call the new function in the given functions. 
# # 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# # 3. Not following the project guidelines will result in a 10% reduction in grades

from re import L
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

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
                if some_value < 50:
                    min = some_value
                else:
                    min = 2000
                thresh = some_value
                jj = j
                ii = i
        if min != 2000:
            matches.append((min,ii,jj))
    matches.sort()
    matches = matches[0:350]
    return matches

# def keypoint_matcher(des1,des2): # my logic for finding which keypoints match from both images
#     matches = []
#     for i in range(des1.shape[0]):
#         thresh = 1000
#         for j in range(des2.shape[0]):
#             some_value = des1[i] - des2[j]
#             some_value = np.linalg.norm(some_value)
#             if thresh > some_value:
#                 min = some_value
#                 thresh = some_value
#                 jj = j
#                 ii = i
#         matches.append((min,ii,jj))
#     matches.sort()
#     return matches

def order(imges): # finds if the images match with other images and writing 1 if it does else 0
    g = 0
    f = 0
    length = len(imges)
    overlap_arr = np.zeros((length,length))
    for g in range(len(imges)):
        for f in range(len(imges)):
            kp1,kp2,des1,des2 = feature_extractor(imges[g],imges[f])
            matches = keypoint_matcher(des1,des2)

            if len(matches)!= 0:
                x = min(matches)
                print(x[0])
            if len(matches) > 0 and x[0] < 30:
                overlap_arr[g][f] = 1
            elif len(matches) == 0:
                overlap_arr[g][f] = 0
    return overlap_arr

def get_imaginary_frames(img):
    frame = np.float32(np.array([ [0,0], [0,img.shape[1]], [img.shape[0], img.shape[1]], [img.shape[0],0] ])).reshape((-1,1,2))
    return frame
    

def Calculate_orientation_rotation(keypoint1,keypoint2,img1,img2): # This function finds how to images will be stitched i.e. orienation and translation
    src_pts = np.float32([key_point.pt for key_point in keypoint1]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_point.pt for key_point in keypoint2]).reshape(-1, 1, 2)
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
    # if len(matches) != 0 and min(matches)[0] < 30:
    #print("i reached here")
    # i = 0
    keypoint1 = []
    keypoint2 = []
    for i in range(len(matches)):
        y = matches[i][2]
        x = matches[i][1]
        keypoint1.append(kp1[x])
        keypoint2.append(kp2[y])
    desired_frame,start_y,start_x = Calculate_orientation_rotation(keypoint1,keypoint2,img1,img2)
    desired_image = cv2.warpPerspective(img1, desired_frame, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0]))
    desired_image[-start_y:img2.shape[0]-start_y, -start_x:img2.shape[1]-start_x] = img2
    return desired_image


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    print(len(imgs))
    overlap_arr = order(imgs)
    # overlap_arr = []
    initial_image1 = imgs[1]
    initial_image2 = imgs[0]
    print(len(initial_image1),len(initial_image2))
    imgs = imgs[1:]
    print(len(imgs))
    base_image = merge_images(initial_image1,initial_image2)
    print(len(base_image))
    l = 0
    while l < len(imgs):
        print(l)
        base_image = merge_images(base_image,imgs[l])
        print(len(base_image))
        l+=1
    # plt.imshow(base_image)
    # plt.show()
    cv2.imwrite(savepath,base_image)
    return overlap_arr


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', N=3, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)

