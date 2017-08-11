import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
#from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
imageio.plugins.ffmpeg.download()
#import subprocess
#subprocess.call('python prakhar_another_code.py', shell=True)

def show_images(images, cmap=None):
    cols = 2
    rows = (len(images)+1)

    plt.figure(figsize=(7, 9))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

#test_images = [plt.imread(path) for path in glob.glob('test_images3/*.jpg')]
test_images = [plt.imread(path) for path in glob.glob('testimages1/*.jpg')]

show_images(test_images)

def select_rgb_white_yellow(image):
    # white color mask
    lower = np.uint8([0, 0, 100])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked


show_images(list(map(select_rgb_white_yellow, test_images)))

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

show_images(list(map(convert_hsv, test_images)))

"""def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

show_images(list(map(convert_hls, test_images)))"""

# retuns white and yellow mask from hsv image
def select_white_yellow(image):
    converted = convert_hsv(image) # convert test-image to hsv
    # white color mask
    lower = np.uint8([  0, 0,  100])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

white_yellow_images = list(map(select_white_yellow, test_images))

show_images(white_yellow_images)


def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(convert_gray_scale, white_yellow_images))

show_images(gray_images)

def apply_smoothing(image):
    """
    kernel_size must be postivie and odd
    """
    return cv2.bilateralFilter(image, 9,75,75)

blurred_images = list(map(lambda image: apply_smoothing(image), gray_images))

show_images(blurred_images)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

edge_images = list(map(lambda image: detect_edges(image), blurred_images))

show_images(edge_images)

def filter_region(image, vertices):

    """Create the mask using the vertices and apply it to the input image"""

    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):

    """It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black)."""

    imshape=image.shape
    lower_left = [imshape[1]/200,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/200,imshape[0]]
    top_left = [imshape[1]/2-imshape[1],imshape[0]/2+imshape[0]/70]
    top_right = [imshape[1]/2+imshape[1],imshape[0]/2+imshape[0]/70]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    mask= np.zeros_like(image)
    mask_color=255;
    cv2.fillPoly(mask,vertices,mask_color)
    masked_img=cv2.bitwise_and(image,mask)


    return filter_region(image, vertices)



# images showing the region of interest only
roi_images = list(map(select_region, edge_images))

show_images(roi_images)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=2*np.pi/180, threshold=100, minLineLength=40, maxLineGap=10)


list_of_lines = list(map(hough_lines, edge_images))

def draw_lines(image, lines, color=[255, 0, 0], thickness=0, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


line_images = []
for image, lines in zip(test_images, list_of_lines):
    line_images.append(draw_lines(image, lines))

show_images(line_images)

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None

    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    print(slope)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.55    # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=5):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)


lane_images = []
for image, lines in zip(test_images, list_of_lines):
    lane_images.append(draw_lane_lines(image, lane_lines(image, lines)))


show_images(lane_images)

"""white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(_lanes) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False"""

"""from collections import deque

QUEUE_LENGTH=50

class LaneDetector:
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        white_yellow = select_white_yellow(image)
        gray         = convert_gray_scale(white_yellow)
        smooth_gray  = apply_smoothing(gray)
        edges        = detect_edges(smooth_gray)
        regions      = select_region(edges)
        lines        = hough_lines(regions)
        left_line, right_line = lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
            return line

        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return draw_lane_lines(image, (left_line, right_line))"""



white_output = 'test_video1.mp3'
clip1 = VideoFileClip("Test_Images.wmv")
white_clip = clip1.fl_image(draw_lane_lines)
white_clip.write_videofile(white_output, audio=False)

"""
HTML("""
"""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
"""
"""
.format(white_output))
"""
