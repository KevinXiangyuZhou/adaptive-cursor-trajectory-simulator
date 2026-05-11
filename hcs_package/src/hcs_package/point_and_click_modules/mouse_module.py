"""
Mouse Module

1) Mouse acceleration function
2) Mouse coordinate disturbance

Reference:
1) https://github.com/INRIA/libpointing/blob/master/pointing-echomouse/darwin-16/f4.dat

This code was written by Seungwon Do (dodoseung) and Byungjoo Lee
"""

import numpy as np
from scipy import interpolate
import csv
import os
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
PATH_GAIN = str(_MODULE_DIR / 'materials' / '0.6875.txt')
PATH_ROTATION_MAP = str(_MODULE_DIR / 'materials' / 'rotation_map_rad.csv')


def rot_mat(x, y, rad):
    c, s = np.cos(rad), np.sin(rad)
    r = np.array(((c, -s), (s, c)))
    vec = np.array((x, y))
    mat = r.dot(vec)

    return mat[0], mat[1]

def mm2in(d):
    return d/25.4

def in2m(d):
    return 0.0254*d

def gain_func(vel):
    return f(vel)

def gain_func_can(vel):
    return g(vel)

cpi = 400
hz = 125
counts = []
pixels = []
with open(PATH_GAIN, 'r') as file:
    for s in file:
        line = s.replace("\n", "").split(': ')
        counts.append(int(line[0]))
        pixels.append(float(line[1]))

motor_speed, gain, visual_speed = [], [], []
for c, p in zip(counts, pixels):
    ms = in2m(c / cpi) * hz
    vs = in2m(p / 110) * hz
    g = 0.0 if ms == 0.0 else vs / ms
    motor_speed.append(ms)
    gain.append(g)
    visual_speed.append(vs)

f = interpolate.interp1d(motor_speed, gain, fill_value="extrapolate")
g = interpolate.interp1d(visual_speed, gain, fill_value="extrapolate")


def get_hand_orientation(hand_loc, forearm_length):
    with open(PATH_ROTATION_MAP, newline='') as file:
        reader = csv.reader(file)
        rotation_map_rad = np.array(list(reader), dtype=float)

    # 31x31 grid covering [-1.5, 1.5] in each axis
    x = np.round(np.linspace(-1.5, 1.5, 31), decimals=3)
    y = np.round(np.linspace(-1.5, 1.5, 31), decimals=3)

    f = interpolate.RegularGridInterpolator(
        (x, y), rotation_map_rad, method='linear', bounds_error=False, fill_value=None
    )

    hand_x = hand_loc[0] / forearm_length
    hand_y = hand_loc[1] / forearm_length

    point = np.array([[hand_x, hand_y]])
    hand_orientation = f(point)[0]

    return float(hand_orientation)


def get_cursor_displacement(hand_start_loc, hand_end_loc, user_forearm_length, gain):
    current_hand_orientation = get_hand_orientation(hand_end_loc, user_forearm_length)
    prev_hand_orientation = get_hand_orientation(hand_start_loc, user_forearm_length)

    hand_displacement = np.linalg.norm(np.array(hand_end_loc) - np.array(hand_start_loc))
    net_hand_rotation = current_hand_orientation - prev_hand_orientation
    if net_hand_rotation == 0: net_hand_rotation = np.finfo(float).tiny

    cursor_displacement = (gain * hand_displacement * 2 * np.sin(net_hand_rotation/2)) / net_hand_rotation

    hand_displacement = np.finfo(float).tiny if hand_displacement == 0 else hand_displacement
    hand_dx = (hand_end_loc[0] - hand_start_loc[0]) / hand_displacement
    hand_dy = (hand_end_loc[1] - hand_start_loc[1]) / hand_displacement

    cursor_dx = (hand_dx * np.cos(net_hand_rotation / 2 + prev_hand_orientation) - hand_dy * np.sin(net_hand_rotation / 2 + prev_hand_orientation)) * cursor_displacement
    cursor_dy = (hand_dx * np.sin(net_hand_rotation / 2 + prev_hand_orientation) + hand_dy * np.cos(net_hand_rotation / 2 + prev_hand_orientation)) * cursor_displacement

    return cursor_dx, cursor_dy
