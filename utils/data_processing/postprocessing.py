import cv2
import numpy as np

def process_output(display_img, q_img, cos_img, sin_img, width_img, grip_length=30, normalize=True):
    #0. Convert image if normalize:
    if normalize:
        display_img = np.uint8(display_img*255+128)
    #1. Find the max quality pixel
    result = np.where(q_img == np.amax(q_img))
    # zip the 2 arrays to get the exact coordinates
    listOfCordinates = list(zip(result[0], result[1]))
    (x, y) = listOfCordinates[0]
    #2. Calculate the angle
    grasp_cos = cos_img[x, y]
    grasp_sin = sin_img[x, y]
    # Normalize sine&cosine of 2*phi
    sum_sqrt = np.sqrt(grasp_cos**2+grasp_sin**2)
    grasp_cos = grasp_cos/sum_sqrt
    grasp_sin = grasp_sin/sum_sqrt
    # Because of symmetric so we can use this
    grasp_sin_modi = np.sqrt((1-grasp_cos)/2)
    grasp_cos_modi = grasp_sin/(2*grasp_sin_modi)
    grasp_cos = grasp_cos_modi
    grasp_sin = grasp_sin_modi

    #width = width_img[x, y]*150
    width = 60
    #3. Find the grasp BoundingBoxes
    x1 = width/2.0
    y1 = grip_length/2.0
    x2 = width/2.0
    y2 = -grip_length/2.0
    x3 = -width/2.0
    y3 = grip_length/2.0
    x4 = -width/2.0
    y4 = -grip_length/2.0
    # Rotate the angle
    R = np.array([[grasp_cos, grasp_sin], [-grasp_sin, grasp_cos]])
    pt1 = np.matmul(R, np.array([x1, y1]))
    pt2 = np.matmul(R, np.array([x2, y2]))
    pt3 = np.matmul(R, np.array([x3, y3]))
    pt4 = np.matmul(R, np.array([x4, y4]))

    pt1[0] = y + pt1[0]
    pt1[1] = x + pt1[1]
    pt2[0] = y + pt2[0]
    pt2[1] = x + pt2[1]
    pt3[0] = y + pt3[0]
    pt3[1] = x + pt3[1]
    pt4[0] = y + pt4[0]
    pt4[1] = x + pt4[1]

    pts = np.array([pt1, pt3, pt4, pt2], np.int32)
    print("x,y: ", x,y)
    print("Width: ", width)
    print("Cos/sine: {}, {}".format(grasp_cos, grasp_sin))
    print("Point: \n", pts)
    #print(R)
    #4. Draw in the images
    color = (0, 255, 255)
    cv2.polylines(display_img, [pts], True, color )
    return display_img
