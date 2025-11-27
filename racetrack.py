import numpy as np

import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.axes as axes
from pchip_interpolate import pchip_interpolate

class RaceTrack:

    def __init__(self, filepath : str, raceline_path : str):
        data = np.loadtxt(filepath, comments="#", delimiter=",")
        self.centerline = data[:, 0:2]
        self.centerline = np.vstack((self.centerline[-1], self.centerline, self.centerline[0]))

        raceline_data = np.loadtxt(raceline_path, comments="#", delimiter=",")
        self.raceline = raceline_data[:, 0:2]
        self.raceline = np.vstack((self.raceline[-1], self.raceline, self.raceline[0]))

        centerline_gradient = np.gradient(self.centerline, axis=0)
        # Unfortunate Warning Print: https://github.com/numpy/numpy/issues/26620
        centerline_cross = np.cross(centerline_gradient, np.array([0.0, 0.0, 1.0]))
        centerline_norm = centerline_cross*\
            np.divide(1.0, np.linalg.norm(centerline_cross, axis=1))[:, None]

        centerline_norm = np.delete(centerline_norm, 0, axis=0)
        centerline_norm = np.delete(centerline_norm, -1, axis=0)

        self.centerline = np.delete(self.centerline, 0, axis=0)
        self.centerline = np.delete(self.centerline, -1, axis=0)

        # Compute track left and right boundaries
        self.right_boundary = self.centerline[:, :2] + centerline_norm[:, :2] * np.expand_dims(data[:, 2], axis=1)
        self.left_boundary = self.centerline[:, :2] - centerline_norm[:, :2]*np.expand_dims(data[:, 3], axis=1)

        # Compute initial position and heading (START ON RACELINE)
        self.initial_state = np.array([
            self.raceline[0, 0],
            self.raceline[0, 1],
            0.0, 0.0,
            np.arctan2(
                self.raceline[1, 1] - self.raceline[0, 1], 
                self.raceline[1, 0] - self.raceline[0, 0]
            )
        ])

        # Matplotlib Plots
        self.code = np.empty(self.centerline.shape[0], dtype=np.uint8)
        self.code.fill(path.Path.LINETO)
        self.code[0] = path.Path.MOVETO
        self.code[-1] = path.Path.CLOSEPOLY

        self.mpl_centerline = path.Path(self.centerline, self.code)
        self.mpl_right_track_limit = path.Path(self.right_boundary, self.code)
        self.mpl_left_track_limit = path.Path(self.left_boundary, self.code)

        self.mpl_centerline_patch = patches.PathPatch(self.mpl_centerline, linestyle="-", fill=False, lw=0.3)
        self.mpl_right_track_limit_patch = patches.PathPatch(self.mpl_right_track_limit, linestyle="--", fill=False, lw=0.2)
        self.mpl_left_track_limit_patch = patches.PathPatch(self.mpl_left_track_limit, linestyle="--", fill=False, lw=0.2)

        # interpolate
        self.interpolate_raceline()

    def plot_track(self, axis : axes.Axes):
        axis.add_patch(self.mpl_centerline_patch)
        axis.add_patch(self.mpl_right_track_limit_patch)
        axis.add_patch(self.mpl_left_track_limit_patch)

    def interpolate_raceline(self, spacing=0.5):
        pts = np.array(self.raceline)
        
        # --- 1. arc-length ---
        d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        s = np.concatenate(([0], np.cumsum(d)))

        # --- 2. Custom PCHIP Interpolation ---
        # Call pchip_interpolate(xi, yi, x) where:
        # xi = s (original arc-lengths)
        # yi = pts[:, 0] or pts[:, 1] (original coordinates)
        # x = s_new (new arc-length grid)
        
        # --- 3. uniform sampling ---
        s_new = np.arange(0, s[-1], spacing)
        
        # Interpolate X-coordinates
        x_new = pchip_interpolate(s, pts[:, 0], s_new)
        
        # Interpolate Y-coordinates
        y_new = pchip_interpolate(s, pts[:, 1], s_new)

        self.raceline = np.column_stack((x_new, y_new))
