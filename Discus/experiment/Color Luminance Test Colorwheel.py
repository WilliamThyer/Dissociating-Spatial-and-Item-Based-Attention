from __future__ import division
from psychopy import visual, event, core, gui, parallel, monitors
from psychopy.visual import ShapeStim, TextStim, Circle, Rect
import numpy as np
import json
import os

print(os.path.realpath(__file__))

def convert_color_value(color, deconvert=False):
    """Converts a list of 3 values from 0 to 255 to -1 to 1.

    Parameters:
    color -- A list of 3 ints between 0 and 255 to be converted.
    """

    if deconvert is True:
        return [round((((n - -1) * 255) / 2) + 0,1) for n in color]
    else:
        return [round((((n - 0) * 2) / 255) + -1,3) for n in color]

def _load_color_wheel(path):

    with open(path) as f:
        color_wheel = json.load(f)
    color_wheel = [convert_color_value(i) for i in color_wheel]

    return np.array(color_wheel)
 
colorwheel_path = 'd:/Discus/experiment/colors.json'
colorwheel = _load_color_wheel(colorwheel_path)[::15]

monitor_name='Experiment Monitor' 
monitor_width=53
monitor_distance=70
monitor_px=[1920, 1080]

experiment_monitor = monitors.Monitor(
            monitor_name, width=monitor_width,
            distance=monitor_distance)
experiment_monitor.setSizePix(monitor_px)

win = visual.Window(monitor = experiment_monitor, fullscr = True, units='deg')

resp = ['space']
i = 0
while resp[0] == 'space':

    rgb = colorwheel[i]
    square = visual.Rect(win, lineColor=None, fillColor=[0,0,0], fillColorSpace='rgb', pos = (0,0) ,width=4, height=4,units='deg')

    square.fillColor = rgb
    print(square.fillColor)
    square.draw()
    win.flip() 
    resp = event.waitKeys(keyList=['space','escape'])
    i += 1
    