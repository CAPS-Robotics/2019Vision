#!/usr/bin/python3

"""
Adapted from https://github.com/WPIRoboticsProjects/GRIP-code-generation/blob/master/python/samples/frc_find_red_areas/frc_find_red_areas.py
"""

import cv2
import numpy
from networktables import NetworkTables
from grip import Vision
from multicam import startCameraServer
from cscore import CameraServer

def extra_processing(pipeline):
    """
    Performs extra processing on the pipeline's outputs and publishes data to NetworkTables.
    :param pipeline: the pipeline that just processed an image
    :return: None
    """
    center_x_positions = []
    center_y_positions = []
    widths = []
    heights = []
    areas = []

    # Find the bounding boxes of the contours to get x, y, width, and height
    for contour in pipeline.filter_contours_output:
        x, y, w, h = cv2.boundingRect(contour)
        center_x_positions.append(x + w / 2)  # X and Y are coordinates of the top-left corner of the bounding box
        center_y_positions.append(y + h / 2)
        widths.append(w)
        heights.append(h)
        area = cv2.contourArea(contour)
        areas.append(area)
    
    print(center_x_positions)

    # Publish to the '/vision/red_areas' network table
    table = NetworkTables.getTable('/GRIP/AllDemContours')
    table.putNumberArray('centerX', center_x_positions)
    table.putNumberArray('centerY', center_y_positions)
    table.putNumberArray('width', widths)
    table.putNumberArray('height', heights)
    table.putNumberArray('area', areas)

def main():
    print('Initializing NetworkTables')
    NetworkTables.initialize(server='10.24.10.2')

    print('Creating pipeline')
    pipeline = Vision()
    
    print('Starting MultiCam server')
    cameras = startCameraServer()

    print('Creating video sink')
    cs = CameraServer.getInstance()
    cvSink = cs.getVideo(camera=cameras[0])
    
    img = numpy.zeros(shape=(640, 360, 3), dtype=numpy.uint8)

    print('Running pipeline')
    while True:
        time, img = cvSink.grabFrame(img)
        if time > 0:
            pipeline.process(img)
            extra_processing(pipeline)

    print('Capture closed')

if __name__ == '__main__':
    main()
