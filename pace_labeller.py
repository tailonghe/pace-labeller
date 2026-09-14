##############################################################################
#  PACE OCI NO2 plume labeller
#  Author(s): Tai-Long He
#
#  Dependencies: numpy, matplotlib, netCDF4, tkinter
#
#  Point "NC folder" at e.g.  PACE_NO2_Gridded_MiddleEast_2024m0426
#  Output ->  labelled_plumes/PACE_NO2_Gridded_MiddleEast_2024m0426/
#                 labelled_<original file name>.nc  (+ .png)
##############################################################################
import os
import glob
import warnings
from datetime import datetime

import numpy as np
import netCDF4 as nc

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import path as mpath
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import LassoSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import END, filedialog, messagebox

warnings.filterwarnings("ignore")
plt.rc("font", size=7)

# ---------------------------------------------------------------- settings --
OUTDIR = "labelled_plumes"
NC_PATTERNS = ("*.nc", "*.nc4", "*.NC")
NO2_NAME_HINTS = ["no2", "nitrogendioxide", "nitrogen_dioxide"]
LAT_NAME_HINTS = ["latitude", "lat"]
LON_NAME_HINTS = ["longitude", "lon"]
VMAX_TARGET = 30e15     # colour limit, target panel   [molec/cm2]
VLIM_ENH = 15e15        # colour limit, enhancement    [molec/cm2]

# starting values in the control column
DEFAULT_LON = 49.529005
DEFAULT_LAT = 26.858919
DEFAULT_BOX_KM = "120"
DEFAULT_BG_PCTILE = "25"

# ---- matplotlib compatibility ----------------------------------------------
# matplotlib.colormaps is 3.5+, .resampled() is 3.6+, cm.get_cmap gone in 3.9,
# LassoSelector lineprops -> props in 3.8.  Support old and new in one file.
def get_cmap(name, n=None):
    try:
        cmap = matplotlib.colormaps[name]
        return cmap.resampled(n) if n else cmap
    except (AttributeError, TypeError):
        import matplotlib.cm as cm
        return cm.get_cmap(name, n) if n else cm.get_cmap(name)


# NaN (no retrieval) is drawn transparent so the hatched "no data" backdrop
# shows through -- never as a colour that could be mistaken for a value.
CMAP_ENH = ListedColormap(get_cmap("bwr", 256)(np.linspace(0, 1, 256)))
CMAP_ENH.set_bad(color="none")
CMAP_TARGET = ListedColormap(get_cmap("Reds", 256)(np.linspace(0, 1, 256)))
CMAP_TARGET.set_bad(color="none")
CMAP_UNLABELLED = ListedColormap(["k"])             # valid but not in the mask
CMAP_UNLABELLED.set_bad(color="none")

# hatched backdrop for no-data pixels: light on the two data panels, dark on
# the plume panel so isolated missing pixels don't speckle the black background
NODATA_HATCH = dict(facecolor="none", edgecolor="0.75", hatch="//",
                    linewidth=0.0, zorder=0)
NODATA_HATCH_DARK = dict(facecolor="k", edgecolor="0.4", hatch="//",
                         linewidth=0.0, zorder=0)
NODATA_EDGE = "0.55"
NODATA_EDGE_DARK = "0.45"
NODATA_MIN_OUTLINE = 4   # only outline no-data patches of at least this many px


def _lasso(ax, cb, color, button):
    try:
        return LassoSelector(ax, cb, props={"color": color, "linewidth": 0.8},
                             button=button)
    except TypeError:                                   # matplotlib < 3.8
        return LassoSelector(ax, cb, lineprops={"color": color, "linewidth": 0.8},
                             button=button)


# ------------------------------------------------------------------- io ----

def list_nc_files(folder):
    files = []
    for pat in NC_PATTERNS:
        files += glob.glob(os.path.join(folder, pat))
    return sorted(set(files), key=lambda p: os.path.basename(p).lower())


def walk_variables(group, prefix=""):
    """Flatten a (possibly grouped) NetCDF file into {full/path: variable}."""
    out = {}
    for name, var in group.variables.items():
        out[prefix + name] = var
    for gname, sub in group.groups.items():
        out.update(walk_variables(sub, prefix + gname + "/"))
    return out


def pick_variable(allvars, hints, exclude_hints=()):
    """First variable whose name matches a hint (case-insensitive)."""
    for hint in hints:
        for key in allvars:
            low = key.lower()
            if hint in low and not any(e in low for e in exclude_hints):
                return key
    return None


def _window(coord1d, c0, half):
    """Index slice covering |coord - c0| <= half on a 1-D coordinate axis."""
    idx = np.where(np.abs(coord1d - c0) <= half)[0]
    if idx.size == 0:
        raise ValueError("ROI is empty -- check lon/lat and box size.")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def load_pace_nc(fpath, varname=None, lon0=None, lat0=None, box_km=80.0):
    """
    Read one local PACE file and cut a square ROI around (lon0, lat0).

    Returns (no2, lon2d, lat2d, meta) with north-up orientation,
    all arrays shaped [ny, nx].
    """
    ds = nc.Dataset(fpath, "r")
    try:
        allvars = walk_variables(ds)

        vkey = varname.strip() if varname and varname.strip() else None
        if vkey is None:
            vkey = pick_variable(allvars, NO2_NAME_HINTS,
                                 exclude_hints=("precision", "uncertainty",
                                                "flag", "qa", "amf", "slant"))
        if vkey is None or vkey not in allvars:
            raise KeyError(
                "NO2 variable not found. Available:\n  " +
                "\n  ".join(sorted(allvars))
            )

        latkey = pick_variable(allvars, LAT_NAME_HINTS,
                               exclude_hints=("bounds", "_bnds", "corner"))
        lonkey = pick_variable(allvars, LON_NAME_HINTS,
                               exclude_hints=("bounds", "_bnds", "corner"))
        if latkey is None or lonkey is None:
            raise KeyError("lat/lon variables not found. Available:\n  " +
                           "\n  ".join(sorted(allvars)))

        latvar, lonvar, no2var = allvars[latkey], allvars[lonkey], allvars[vkey]
        lat = np.ma.filled(latvar[:].astype(np.float64), np.nan)
        lon = np.ma.filled(lonvar[:].astype(np.float64), np.nan)

        roi = lon0 is not None and lat0 is not None and box_km and box_km > 0
        if roi:
            half_lat = (box_km / 2.0) / 111.0
            half_lon = half_lat / max(np.cos(np.deg2rad(lat0)), 0.05)

        bounds = (float(np.nanmin(lon)), float(np.nanmax(lon)),
                  float(np.nanmin(lat)), float(np.nanmax(lat)))

        def outside():
            return ValueError(
                "ROI outside file coverage.\n"
                "  file covers lon %.3f..%.3f, lat %.3f..%.3f\n"
                "  requested  lon %.3f +/-%.3f, lat %.3f +/-%.3f"
                % (bounds[0], bounds[1], bounds[2], bounds[3],
                   lon0, half_lon, lat0, half_lat))

        if lat.ndim == 1 and lon.ndim == 1:
            # ---- regular grid: slice the ROI window straight off disk --------
            try:
                rsl = _window(lat, lat0, half_lat) if roi else slice(None)
                csl = _window(lon, lon0, half_lon) if roi else slice(None)
            except ValueError:
                raise outside()

            dims = no2var.dimensions
            latdim, londim = latvar.dimensions[0], lonvar.dimensions[0]
            if latdim in dims and londim in dims:
                slicer = tuple(rsl if d == latdim else
                               csl if d == londim else slice(None) for d in dims)
                no2 = np.squeeze(np.ma.filled(
                    no2var[slicer].astype(np.float64), np.nan))
                if dims.index(londim) < dims.index(latdim):
                    no2 = no2.T                       # stored as [lon, lat]
            else:                                     # dim names don't match
                no2 = np.squeeze(np.ma.filled(no2var[:].astype(np.float64), np.nan))
                if no2.shape == (lon.size, lat.size) and no2.shape[0] != no2.shape[1]:
                    no2 = no2.T
                if no2.shape != (lat.size, lon.size):
                    raise ValueError(f"shape mismatch: {vkey}{no2.shape} vs "
                                     f"lat({lat.size}) x lon({lon.size})")
                no2 = no2[rsl, csl]

            lat, lon = lat[rsl], lon[csl]
            if no2.ndim != 2 or no2.shape != (lat.size, lon.size):
                raise ValueError(f"ROI shape mismatch: {vkey}{no2.shape} vs "
                                 f"lat({lat.size}) x lon({lon.size})")
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            # ---- swath / 2-D coordinates: read all, then cut a bounding box --
            no2 = np.squeeze(np.ma.filled(no2var[:].astype(np.float64), np.nan))
            if no2.ndim != 2:
                raise ValueError(f"'{vkey}' is {no2.ndim}-D after squeeze; "
                                 "expected a 2-D field.")
            lat2d, lon2d = np.squeeze(lat), np.squeeze(lon)
            if lat2d.shape != no2.shape:
                if lat2d.T.shape == no2.shape:
                    lat2d, lon2d = lat2d.T, lon2d.T
                else:
                    raise ValueError("2-D lat/lon shape does not match the field")
            if roi:
                inbox = ((np.abs(lat2d - lat0) <= half_lat) &
                         (np.abs(lon2d - lon0) <= half_lon))
                if not inbox.any():
                    raise outside()
                rows = np.where(inbox.any(axis=1))[0]
                cols = np.where(inbox.any(axis=0))[0]
                sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
                no2, lat2d, lon2d = no2[sl], lat2d[sl], lon2d[sl]

        # ---- north up --------------------------------------------------------
        if lat2d.shape[0] > 1 and np.nanmean(lat2d[-1]) > np.nanmean(lat2d[0]):
            no2, lat2d, lon2d = no2[::-1], lat2d[::-1], lon2d[::-1]

        meta = {"path": fpath, "file": os.path.basename(fpath),
                "stem": os.path.splitext(os.path.basename(fpath))[0],
                "variable": vkey, "lat_var": latkey, "lon_var": lonkey,
                "shape": no2.shape, "bounds": bounds}
        return no2, lon2d, lat2d, meta
    finally:
        ds.close()


def write_nc(fname, lon2d, lat2d, no2, background, enhancement, mask, meta):
    with nc.Dataset(fname, "w", format="NETCDF4") as ds:
        ds.createDimension("y", no2.shape[0])
        ds.createDimension("x", no2.shape[1])

        def add(name, data, units, long_name, dtype=np.float32):
            v = ds.createVariable(name, dtype, ("y", "x"), zlib=True, complevel=4)
            v[:] = data
            v.units = units
            v.long_name = long_name
            return v

        add("lat", lat2d, "degrees_north", "latitude")
        add("lon", lon2d, "degrees_east", "longitude")
        add("no2", no2, "molec cm-2", "NO2 column (target scene)")
        add("no2_enhancement", enhancement, "molec cm-2",
            "NO2 column minus scene background")
        add("plume_mask", mask.astype(np.int8), "1",
            "1 = labelled plume, 0 = background", dtype=np.int8)

        ds.background_value = float(background)
        ds.source_file = meta["file"]
        ds.source_variable = meta["variable"]
        ds.created = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ds.history = "created by pace_labeller.py"


# ------------------------------------------------------------------ state --
STATE = {
    "no2": None, "lon": None, "lat": None, "meta": None,
    "bg": np.nan, "enh": None, "mask": None,
    "lon0": None, "lat0": None,
    "folder": None, "files": [],
    "im_mask": None, "lassos": [],
    "roi": None,            # (lon, lat, box, pctile) currently drawn
    "index": None,          # list index currently drawn
    "unsaved": False,       # mask edited since the last save
}

# constrained_layout keeps the three square axes and their colorbars aligned
# without hand-placed cbar axes (works on matplotlib >= 2.2).
# 8.35 x 4.0 in at 130 dpi = 1085 x 520 px, matching the canvas frame below,
# so the three panels fill it with no wasted vertical band.
fig = Figure(figsize=(8.35, 4.0), dpi=130, constrained_layout=True)


def large_holes(nodata, min_px):
    """Keep only no-data patches of >= min_px pixels (4-connected)."""
    if min_px <= 1 or not nodata.any():
        return nodata
    ny, nx = nodata.shape
    seen = np.zeros(nodata.shape, dtype=bool)
    keep = np.zeros(nodata.shape, dtype=bool)
    for i0 in range(ny):
        for j0 in range(nx):
            if not nodata[i0, j0] or seen[i0, j0]:
                continue
            stack, comp = [(i0, j0)], []
            seen[i0, j0] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < ny and 0 <= xx < nx
                            and nodata[yy, xx] and not seen[yy, xx]):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            if len(comp) >= min_px:
                for y, x in comp:
                    keep[y, x] = True
    return keep


def pixel_aspect(lat2d, lon2d):
    """
    Display aspect that makes a grid cell physically square.

    The ROI is cut square in km, but the grid is regular in degrees, so a
    longitude step is cos(lat) shorter on the ground than a latitude step.
    aspect = dlat / (dlon * cos(lat)) undoes that; with equal dlat/dlon it is
    just 1/cos(lat), and the panel comes out square.
    """
    try:
        dlon = np.nanmedian(np.abs(np.diff(lon2d[0, :])))
        dlat = np.nanmedian(np.abs(np.diff(lat2d[:, 0])))
        coslat = max(np.cos(np.deg2rad(np.nanmean(lat2d))), 0.05)
        a = float(dlat / (dlon * coslat))
        return a if np.isfinite(a) and a > 0 else 1.0
    except Exception:
        return 1.0


def build_panels():
    """(Re)draw all three panels from scratch after a new file is loaded."""
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    no2, enh, mask = STATE["no2"], STATE["enh"], STATE["mask"]
    nodata = np.isnan(no2)
    outlined = large_holes(nodata, NODATA_MIN_OUTLINE)
    aspect = pixel_aspect(STATE["lat"], STATE["lon"])

    fig.suptitle("%s  |  %s\nleft-drag = add to mask,  right-drag = erase"
                 "   |   hatched = no retrieval"
                 % (STATE["meta"]["file"], STATE["meta"]["variable"]),
                 fontsize=8)

    def backdrop(ax, dark=False):
        """Hatched panel background + outline around the no-data regions."""
        ax.set_facecolor("k" if dark else "white")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               **(NODATA_HATCH_DARK if dark else NODATA_HATCH)))
        if outlined.any() and not outlined.all():
            ax.contour(outlined.astype(float), levels=[0.5],
                       colors=NODATA_EDGE_DARK if dark else NODATA_EDGE,
                       linewidths=0.6, zorder=3)

    def draw(ax, data, cmap, vmin, vmax, zorder=1):
        return ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                         interpolation="nearest", aspect=aspect, zorder=zorder)

    def finish(ax, im, title, extend):
        ax.set_title(title, fontsize=9)
        ax.xaxis.tick_bottom()
        fig.colorbar(im, ax=ax, orientation="horizontal", extend=extend,
                     shrink=0.92, pad=0.03, aspect=28,
                     label="[ molec / cm$^2$ ]")

    backdrop(ax1)
    finish(ax1, draw(ax1, no2, CMAP_TARGET, 0, VMAX_TARGET),
           "Target NO$_2$", "max")

    backdrop(ax2)
    finish(ax2, draw(ax2, enh, CMAP_ENH, -VLIM_ENH, VLIM_ENH),
           "NO$_2$ Enhancement", "both")

    # panel 3: hatch = no data, black = valid but unlabelled, colour = mask
    backdrop(ax3, dark=True)
    draw(ax3, np.where(nodata, np.nan, 0.0), CMAP_UNLABELLED, 0, 1, zorder=1)
    STATE["im_mask"] = draw(ax3, np.where(mask > 0, enh, np.nan),
                            CMAP_ENH, -VLIM_ENH, VLIM_ENH, zorder=2)
    finish(ax3, STATE["im_mask"], "NO$_2$ Plume Signal", "both")

    # the lasso lives on the middle panel; keep references or they get GC'd
    STATE["lassos"] = [_lasso(ax2, on_add, "blue", 1),
                       _lasso(ax2, on_del, "white", 3)]
    app.canvas.draw_idle()


def refresh_mask_panel():
    """Cheap update after a lasso stroke -- only the right panel changes."""
    if STATE["im_mask"] is None:
        return
    STATE["im_mask"].set_data(np.where(STATE["mask"] > 0, STATE["enh"], np.nan))
    app.canvas.draw_idle()


# ------------------------------------------------------------- lasso ------
def _apply_lasso(verts, value):
    if STATE["mask"] is None:
        return
    ny, nx = STATE["mask"].shape
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    pix = np.vstack((xv.ravel(), yv.ravel())).T
    ind = mpath.Path(verts).contains_points(pix, radius=0.3)
    flat = STATE["mask"].ravel()
    flat[ind] = value
    STATE["mask"] = flat.reshape(ny, nx)
    STATE["unsaved"] = bool(STATE["mask"].sum())
    # LassoSelector already fires this on mouse release, so redraw right here.
    # (Hooking button_release_event separately ran *before* this callback and
    #  made the mask lag one stroke behind.)
    refresh_mask_panel()


def on_add(verts):
    _apply_lasso(verts, 1)


def on_del(verts):
    _apply_lasso(verts, 0)


# --------------------------------------------------------------- the GUI --
PAD = {"padx": 4, "pady": 3}


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PACE OCI NO2 plume labeller (local folder) ver 4.0")
        self.geometry("1400x680")
        self.resizable(0, 0)
        self._suppress_select = False
        self.build()

    # ---- layout ----------------------------------------------------------
    def build(self):
        # canvas frame is sized to the figure's own aspect ratio; the log sits
        # in the strip underneath so no space is left blank.
        self.figure_frame = tk.Frame(self)
        self.figure_frame.place(relx=0.0, rely=0.0, relwidth=0.775, relheight=0.765)
        self.canvas = FigureCanvasTkAgg(fig, self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        log = tk.Frame(self)
        log.place(relx=0.008, rely=0.772, relwidth=0.762, relheight=0.218)
        log_sb = tk.Scrollbar(log, orient="vertical")
        self.print_text = tk.Text(log, height=6, state="disabled",
                                  yscrollcommand=log_sb.set)
        log_sb.config(command=self.print_text.yview)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.print_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        col = tk.Frame(self, padx=10, pady=10)
        col.place(relx=0.782, rely=0.0, relwidth=0.218, relheight=1.0)
        # buttons are packed to the BOTTOM first, so they stay on screen no
        # matter how tall the file list above them grows.
        foot = tk.Frame(col)
        foot.pack(side=tk.BOTTOM, fill=tk.X)
        p = tk.Frame(col)
        p.pack(side=tk.TOP, fill=tk.X)
        r = 0

        tk.Label(p, foreground="red", text="NC folder:").grid(
            row=r, column=0, columnspan=2, sticky="w", **PAD)
        r += 1
        self.folder_entry = tk.Entry(p, width=32)
        self.folder_entry.grid(row=r, column=0, columnspan=2, sticky="w", **PAD)
        self.folder_entry.insert(0, "PACE_NO2_Gridded_MiddleEast_2024m0426")
        r += 1
        tk.Button(p, text="Browse", width=10, command=self.browse_folder).grid(
            row=r, column=0, sticky="w", **PAD)
        tk.Button(p, text="Scan", width=10, command=self.scan_folder).grid(
            row=r, column=1, sticky="w", **PAD)
        r += 1

        tk.Label(p, foreground="red", text="Files  (click to load):").grid(
            row=r, column=0, columnspan=2, sticky="w", **PAD)
        r += 1
        lb = tk.Frame(p)
        lb.grid(row=r, column=0, columnspan=2, sticky="w", **PAD)
        sb = tk.Scrollbar(lb, orient="vertical")
        self.file_list = tk.Listbox(lb, width=28, height=8,
                                    exportselection=False, yscrollcommand=sb.set)
        sb.config(command=self.file_list.yview)
        self.file_list.pack(side=tk.LEFT)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.file_list.bind("<<ListboxSelect>>", self.on_select)
        r += 1

        nav = tk.Frame(p)
        nav.grid(row=r, column=0, columnspan=2, sticky="w", **PAD)
        tk.Button(nav, text="<<", width=5,
                  command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=3)
        tk.Button(nav, text=">>", width=5,
                  command=lambda: self.step(1)).pack(side=tk.LEFT, padx=3)
        r += 1

        def field(label, default):
            nonlocal r
            tk.Label(p, foreground="red", text=label).grid(
                row=r, column=0, sticky="w", **PAD)
            e = tk.Entry(p, width=14)
            e.grid(row=r, column=1, sticky="w", **PAD)
            e.insert(0, default)
            e.bind("<Return>", lambda ev: self.load_current())
            r += 1
            return e

        self.lon_entry = field("Longitude:", "%.6f" % DEFAULT_LON)
        self.lat_entry = field("Latitude:", "%.6f" % DEFAULT_LAT)
        self.box_entry = field("Box (km):", DEFAULT_BOX_KM)
        self.bg_entry = field("Bkg pctile:", DEFAULT_BG_PCTILE)
        self.roi_entries = [self.lon_entry, self.lat_entry,
                            self.box_entry, self.bg_entry]
        for e in self.roi_entries:
            e.bind("<KeyRelease>", self.flag_roi_edit)
        self.roi_hint = tk.Label(p, text="(Enter or Reload re-cuts the ROI)",
                                 foreground="grey")
        self.roi_hint.grid(
            row=r, column=0, columnspan=2, sticky="w", **PAD)
        r += 1

        btns = tk.Frame(foot)
        btns.pack(side=tk.TOP, anchor="w", pady=4)
        tk.Button(btns, text="Reload ROI", width=10,
                  command=self.load_current).pack(side=tk.LEFT, padx=3)
        tk.Button(btns, text="Clear", width=7,
                  command=self.clear_mask).pack(side=tk.LEFT, padx=3)
        tk.Button(btns, text="Save", width=7,
                  command=self.save_output).pack(side=tk.LEFT, padx=3)

        self.action_label = tk.Label(foot, foreground="red", anchor="w")
        self.action_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

    # ---- helpers ---------------------------------------------------------
    def post(self, msg):
        self.print_text.configure(state="normal")
        self.print_text.insert(END, msg + "\n")
        self.print_text.configure(state="disabled")
        self.print_text.see("end")
        print(msg)

    def current_file(self):
        sel = self.file_list.curselection()
        if not sel or not STATE["files"]:
            return None
        return STATE["files"][sel[0]]

    def roi_fields(self):
        """Current text of the four ROI entries, or None if not all numeric."""
        try:
            return tuple(float(e.get()) for e in self.roi_entries)
        except ValueError:
            return None

    def flag_roi_edit(self, _event=None):
        """Highlight the ROI fields while they differ from what is drawn."""
        pending = STATE["roi"] is not None and self.roi_fields() != STATE["roi"]
        for e in self.roi_entries:
            e.configure(background="#fff4cc" if pending else "white")
        self.roi_hint.configure(
            text="ROI changed -- press Enter or Reload" if pending
            else "(Enter or Reload re-cuts the ROI)",
            foreground="#b06000" if pending else "grey")

    def confirm_discard(self):
        """Ask before throwing away a mask that has not been saved."""
        if not STATE["unsaved"]:
            return True
        return messagebox.askokcancel(
            "Unsaved mask",
            "The current mask has not been saved. Discard it?")

    def select_index(self, i):
        """Move the highlight without letting <<ListboxSelect>> double-fire."""
        self._suppress_select = True
        self.file_list.selection_clear(0, END)
        self.file_list.selection_set(i)
        self.file_list.see(i)
        self.update_idletasks()
        self._suppress_select = False

    # ---- folder / file handling -----------------------------------------
    def browse_folder(self):
        d = filedialog.askdirectory(title="Select the folder holding the PACE nc files")
        if d:
            self.folder_entry.delete(0, END)
            self.folder_entry.insert(0, d)
            self.scan_folder()

    def scan_folder(self):
        folder = self.folder_entry.get().strip().strip('"')
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Folder not found:\n%s" % folder)
            return
        files = list_nc_files(folder)
        STATE["folder"], STATE["files"] = folder, files

        self._suppress_select = True
        self.file_list.delete(0, END)
        for f in files:
            self.file_list.insert(END, os.path.basename(f))
        self._suppress_select = False

        self.post("> folder: %s" % os.path.basename(os.path.normpath(folder)))
        self.post("> found %d nc file(s)" % len(files))
        if files:
            self.select_index(0)
            self.load_current()
        else:
            self.post("! nothing matching *.nc / *.nc4 there")
            self.action_label["text"] = "no files"

    def on_select(self, _event):
        if not self._suppress_select:
            self.load_current()

    def step(self, delta):
        if not STATE["files"]:
            return
        sel = self.file_list.curselection()
        self.select_index(((sel[0] if sel else 0) + delta) % len(STATE["files"]))
        self.load_current()

    # ---- actions ---------------------------------------------------------
    def load_current(self):
        if not STATE["files"]:
            self.scan_folder()
            return
        sel = self.file_list.curselection()
        fpath = self.current_file()
        if fpath is None:
            return
        if not self.confirm_discard():
            if STATE["index"] is not None:
                self.select_index(STATE["index"])       # undo the click
            return
        try:
            lon0 = float(self.lon_entry.get())
            lat0 = float(self.lat_entry.get())
            box = float(self.box_entry.get())
            bgp = float(self.bg_entry.get())
        except ValueError:
            messagebox.showerror("Error", "lon / lat / box / percentile must be numbers.")
            return

        try:
            no2, lon2d, lat2d, meta = load_pace_nc(fpath, None, lon0, lat0, box)
        except Exception as exc:
            self.post("! load failed: %s" % exc)
            messagebox.showerror("Load failed", str(exc))
            return

        nvalid = int(np.sum(~np.isnan(no2)))
        # an ROI with no valid retrieval still gets drawn (fully hatched), so
        # the panels never disagree with the file highlighted in the list
        bg = np.nanpercentile(no2, bgp) if nvalid else np.nan
        STATE.update(no2=no2, lon=lon2d, lat=lat2d, meta=meta, bg=bg,
                     enh=np.where(np.isnan(no2), np.nan, no2 - bg),
                     mask=np.zeros(no2.shape, dtype=np.int8),
                     lon0=lon0, lat0=lat0,
                     roi=(lon0, lat0, box, bgp),
                     index=(sel[0] if sel else 0),
                     unsaved=False)
        self.flag_roi_edit()

        self.post("> %s" % meta["file"])
        self.post("   ROI %.4fE %.4fN, %g km  ->  lon %.3f..%.3f, lat %.3f..%.3f"
                  % (lon0, lat0, box,
                     float(np.nanmin(lon2d)), float(np.nanmax(lon2d)),
                     float(np.nanmin(lat2d)), float(np.nanmax(lat2d))))
        ntot = no2.size
        if nvalid:
            self.post("   %d x %d px, %d valid (%.0f%% no retrieval), bkg %.2e (p%.0f)"
                      % (no2.shape[0], no2.shape[1], nvalid,
                         100.0 * (ntot - nvalid) / ntot, bg, bgp))
            self.action_label["text"] = "Loaded."
        else:
            self.post("   %d x %d px, NO valid retrieval in this ROI "
                      "(fully screened for this scene)" % no2.shape)
            self.action_label["text"] = "No data in ROI."
        build_panels()

    def clear_mask(self):
        if STATE["mask"] is not None:
            STATE["mask"][:] = 0
            STATE["unsaved"] = False
            refresh_mask_panel()

    def save_output(self):
        if STATE["no2"] is None:
            messagebox.showwarning("Nothing to save", "Load a file first.")
            return
        if STATE["mask"].sum() == 0:
            if not messagebox.askokcancel("Empty mask", "Mask is empty. Save anyway?"):
                return

        folder_name = os.path.basename(os.path.normpath(STATE["folder"]))
        subdir = os.path.join(OUTDIR, folder_name)
        os.makedirs(subdir, exist_ok=True)
        outname = os.path.join(subdir, "labelled_" + STATE["meta"]["stem"])

        if os.path.exists(outname + ".nc"):
            if not messagebox.askokcancel(
                    "Overwrite?",
                    "labelled_%s.nc already exists. Overwrite?" % STATE["meta"]["stem"]):
                return

        write_nc(outname + ".nc", STATE["lon"], STATE["lat"], STATE["no2"],
                 STATE["bg"], STATE["enh"], STATE["mask"], STATE["meta"])
        fig.savefig(outname + ".png", dpi=200)
        STATE["unsaved"] = False
        self.post("> saved labelled_%s (.nc/.png), %d plume px"
                  % (STATE["meta"]["stem"], int(STATE["mask"].sum())))
        self.action_label["text"] = "Saved."

    def on_closing(self):
        if not self.confirm_discard():
            return
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.quit()
            self.destroy()


app = Application()
app.protocol("WM_DELETE_WINDOW", app.on_closing)
app.mainloop()
