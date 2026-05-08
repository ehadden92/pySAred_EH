
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import h5py, os, sys, pkgutil, platform
import pyqtgraph as pg
from scipy.interpolate import griddata

# --- Matplotlib Qt backend for interactive plot in Monitors/Time tab ---
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.pyplot as _plt

# --- helper: safe x-range for preview lines (handles empty arrays) ---
def _safe_xrange_from(seq, fallback=(0.0, 1.0)):
    try:
        a = np.asarray(seq)
        if a.size:
            lo = np.nanmin(a)
            hi = np.nanmax(a)
            return np.array([lo, hi])
    except Exception:
        pass
    return np.array(list(fallback))


QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# --- unified helpers: avoid ambiguous truth-value for numpy arrays ---
try:
    first_non_none
except NameError:
    def first_non_none(*args):
        for a in args:
            if a is not None:
                return a
        return None
try:
    _first_not_none
except NameError:
    def _first_not_none(*args):
        return first_non_none(*args)

# --- safe 'is empty' for arrays/lists/None ---
def _is_empty(obj):
    try:
        import numpy as _np
        if obj is None:
            return True
        a = _np.asarray(obj)
        return a.size == 0
    except Exception:
        try:
            return len(obj) == 0
        except Exception:
            return False

# --- helper: safe xrange for preview/2D lines when input may be empty ---
def _safe_xrange_from(seq, fallback=(0.0, 1.0)):
    import numpy as _np
    try:
        a = _np.asarray(seq)
        if a.size:
            lo = _np.nanmin(a)
            hi = _np.nanmax(a)
            return _np.array([lo, hi])
    except Exception:
        pass
    return _np.array(list(fallback))



# --- BEGIN: HDF5 schema resolver (SuperADAM + MiniADAM) ---
class H5Resolver:
    """
    Unifies HDF5 layout differences between SuperADAM (NR/PNR) and MiniADAM (NR).
    All public methods return numpy arrays or h5py datasets compatible with current code.
    """

    def __init__(self, scan_group):
        # scan_group is /Sxxxxx
        self.SCAN = scan_group
        self.INST = scan_group.get("instrument") if "instrument" in scan_group else None

    # ---------- small helpers ----------
    def _ds(self, g, key):
        if g is None: return None
        try:
            return g.get(key)
        except Exception:
            return None

    def _as_str(self, x):
        if x is None: return None
        if isinstance(x, h5py.Dataset):
            v = x[()]
        else:
            v = x
        if isinstance(v, (bytes, np.bytes_)): return v.decode("utf-8", "ignore")
        return str(v)

    # ---------- instrument identity ----------
    def instrument_name(self):
        node = self._ds(self.SCAN, "instrument")
        name = None
        if isinstance(node, h5py.Dataset):
            name = self._as_str(node)
        else:
            for k in ("instrument", "name", "title"):
                ds = self._ds(node, k)
                if isinstance(ds, h5py.Dataset):
                    name = self._as_str(ds); break
            if name is None and node is not None:
                for k in ("instrument", "name", "title"):
                    if k in node.attrs:
                        v = node.attrs[k]
                        name = v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                        break
        if not name: return "SuperADAM"
        return "MiniADAM" if "mini" in name.lower() else "SuperADAM"

    # ---------- counters / monitors / time ----------
    def _scalers_candidates(self):
        cands = []
        if self.INST is not None and "scalers" in self.INST:
            cands.append(self.INST["scalers"])
        gisans = self._ds(self.SCAN, "gisans")
        if gisans is not None:
            inst2 = self._ds(gisans, "instrument")
            if inst2 is not None and "scalers" in inst2:
                cands.append(inst2["scalers"])
        return cands

    def monitor_and_time(self, pol=None):
        # MiniADAM convenience 1D first
        if self.INST is not None and "scalers" in self.INST:
            mon1d = self._ds(self.INST["scalers"], "mon")
            sec1d = self._ds(self.INST["scalers"], "sec")
            if isinstance(mon1d, h5py.Dataset) and isinstance(sec1d, h5py.Dataset):
                return np.array(mon1d), np.array(sec1d)

        # matrix + mnemonics (SuperADAM; Mini can also ship a smaller table)
        for scalers in self._scalers_candidates():
            d = self._ds(scalers, "data")
            names = self._ds(scalers, "SPEC_counter_mnemonics")
            if not (isinstance(d, h5py.Dataset) and isinstance(names, h5py.Dataset)): 
                continue
            arr = np.array(d).T
            lbls = [self._as_str(x).strip().lower() for x in names]

            def col(label):
                label = label.lower()
                for i, nm in enumerate(lbls):
                    if label == nm or label in nm:
                        return arr[i]
                return None

            if pol in ("uu", "dd", "du", "ud"):
                # prefer per-channel m1..m4 if present
                mapping = {"uu": "m1", "dd": "m2", "du": "m3", "ud": "m4"}
                mon = col(mapping[pol]) or col("mon0") or col("mon") or col("monitor")
            else:
                mon = col("mon0") or col("mon") or col("monitor")

            sec = col("sec") or col("time") or col("seconds")

            if mon is not None and sec is not None:
                return mon, sec

        return None, None  # caller handles fallback

    # ---------- motors & slits ----------
    def _motors_by_mnemonics(self):
        out = {}
        motors = self._ds(self.INST, "motors") if self.INST is not None else None
        if motors is not None:
            data = self._ds(motors, "data")
            names = self._ds(motors, "SPEC_motor_mnemonics")
            if isinstance(data, h5py.Dataset) and isinstance(names, h5py.Dataset):
                M = np.array(data).T
                for idx, nm in enumerate(names):
                    out[self._as_str(nm).lower()] = M[idx]
        return out
    
    def motor_series(self, *aliases):
        table = self._motors_by_mnemonics()  # {mnemonic(lower): vector}

        # 1) If caller requested 'th', return exact 'th' immediately (no substring match!)
        if any(al.lower() == "th" for al in aliases):
            if "th" in table:
                return table["th"]

        # 2) Otherwise: match aliases by exact or underscore-delimited forms only
        def _match(nm, al):
            # Allow: 'alpha' == 'alpha', 'alpha_*', '*_alpha' (but never substring in the middle)
            return (nm == al) or nm.startswith(al + "_") or nm.endswith("_" + al)

        for al in (a.lower() for a in aliases):
            for nm, vec in table.items():
                if _match(nm, al):
                    return vec

        # 3) Fallback: direct group path lookup (unchanged)
        mgrp = self._ds(self._ds(self.INST, "motors"), aliases[0])
        val = self._ds(mgrp, "value") if mgrp is not None else None
        return np.array(val) if isinstance(val, h5py.Dataset) else None

    def th_list(self):
        return self.motor_series("th", "theta", "samth", "th_sam", "th_sample")

    def tth_list(self):
        v = self.motor_series("tth", "2th", "twotheta", "two_theta", "a2")
        if v is None:
            th = self.th_list()
            if th is not None: return 2*th  # sensible fallback
        return v

    def slit_lists(self):
        if self.instrument_name() == "MiniADAM":
            s1 = self.motor_series("s3hg", "s3h", "ap3hg", "slit3")
            s2 = self.motor_series("s4hg", "s4h", "ap4hg", "slit4")
        else:
            s1 = self.motor_series("s1hg", "s1h", "ap1hg", "slit1")
            s2 = self.motor_series("s2hg", "s2h", "ap2hg", "slit2")
        return s1, s2

    # ---------- polarisation & images ----------
    def pol_list(self):
        dets = self._ds(self.INST, "detectors")
        if dets is not None:
            keys = list(dets.keys())
            if any(k in keys for k in ("psd_uu", "psd_dd", "psd_du", "psd_ud")):
                return [p for p in ["uu", "dd", "du", "ud"] if f"psd_{p}" in keys]
        return ["uu"]  # NR

    def detector_stack(self, pol=None):
        dets = self._ds(self.INST, "detectors")
        # PNR case
        if pol in ("uu", "dd", "du", "ud") and dets is not None and f"psd_{pol}" in dets:
            ds = self._ds(dets[f"psd_{pol}"], "data")
            if isinstance(ds, h5py.Dataset): return ds
        # Super NR may have instrument/detectors/psd
        if dets is not None and "psd" in dets:
            ds = self._ds(dets["psd"], "data")
            if isinstance(ds, h5py.Dataset): return ds
        # GISANS NR
        gisans = self._ds(self.SCAN, "gisans")
        dnode = self._ds(gisans, "data")
        ds = self._ds(dnode, "data")
        if isinstance(ds, h5py.Dataset): return ds           # Super NR
        imgs = self._ds(dnode, "images")
        if isinstance(imgs, h5py.Dataset): return imgs       # Mini NR
        # Last resort: first 3D dataset under detectors
        if dets is not None:
            for k in dets:
                ds = self._ds(dets[k], "data")
                try:
                    if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                        return ds
                except Exception:
                    pass
        return None

    # ---------- ROI ----------
    def roi(self, images_hw=None):
        for scalers in self._scalers_candidates():
            roi_grp = self._ds(scalers, "roi")
            roi_ds = self._ds(roi_grp, "roi") if roi_grp is not None else None
            if isinstance(roi_ds, h5py.Dataset):
                arr = np.array(roi_ds).astype(int)
                if arr.size == 4:
                    return [int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])]
        # synthesize default ROI if file ROI is absent
        if images_hw:
            H, W = images_hw
            y_mid = H // 2
            y_half = max(4, H // 20)
            x_left = max(0, W // 3)
            x_right = min(W, 2 * W // 3)
            return [y_mid - y_half, y_mid + y_half, x_left, x_right]
        return [0, -1, 0, -1]

    # ---------- optional 1D lineouts (PONOS/OSS) ----------
    def lineouts(self, pol=None):
        for gname in ("ponos", "oss"):
            grp = self._ds(self.SCAN, gname)
            data = self._ds(grp, "data") if grp is not None else None
            if data is None: 
                continue
            if gname == "ponos":
                if pol in ("uu", "dd", "du", "ud"):
                    ds = self._ds(data, f"data_{pol}")
                    if isinstance(ds, h5py.Dataset): return np.array(ds)
            else:  # oss
                ds = self._ds(data, "data")
                if isinstance(ds, h5py.Dataset): return np.array(ds)
        return None
# --- END: HDF5 schema resolver ---
# --- BEGIN: Safety patch to avoid numpy truth-value ambiguity and to robustly resolve monitors (MiniADAM/SuperADAM) ---
import numpy as _np, h5py as _h5py

def _safe_list_like(v):
    if v is None:
        return []
    if isinstance(v, _np.ndarray):
        return v.tolist()
    try:
        return list(v)
    except Exception:
        return [v]

def _first_not_none(*args):
    for a in args:
        if a is not None:
            return a
    return None

# Monkey-patch th_list/tth_list to always return Python lists (never numpy arrays or None)
if hasattr(H5Resolver, 'th_list'):
    _orig_th_list = H5Resolver.th_list
    def _th_list_wrapped(self):
        return _safe_list_like(_orig_th_list(self))
    H5Resolver.th_list = _th_list_wrapped

if hasattr(H5Resolver, 'tth_list'):
    _orig_tth_list = H5Resolver.tth_list
    def _tth_list_wrapped(self):
        return _safe_list_like(_orig_tth_list(self))
    H5Resolver.tth_list = _tth_list_wrapped

# Replace monitor_and_time with a safe, schema-aware implementation
def _monitor_and_time_safe(self, pol=None):
    # 1) MiniADAM convenience 1D scalers at /instrument/scalers/{mon,sec}
    if self.INST is not None and "scalers" in self.INST:
        scal = self.INST["scalers"]
        mon1d = scal.get("mon", None)
        sec1d = scal.get("sec", None)
        if isinstance(mon1d, _h5py.Dataset) and mon1d.ndim == 1:
            mon = _np.array(mon1d)
            sec = _np.array(sec1d) if isinstance(sec1d, _h5py.Dataset) and sec1d.ndim == 1 else None
            return mon, sec

    # 2) Matrix + mnemonics under /instrument/scalers or /gisans/instrument/scalers
    def _scaler_candidates():
        cands = []
        if self.INST is not None and "scalers" in self.INST:
            cands.append(self.INST["scalers"])
        gisans = self._ds(self.SCAN, "gisans")
        if gisans is not None:
            inst2 = self._ds(gisans, "instrument")
            if inst2 is not None and "scalers" in inst2:
                cands.append(inst2["scalers"])
        return cands

    for scal in _scaler_candidates():
        data = self._ds(scal, "data")
        names = self._ds(scal, "SPEC_counter_mnemonics")
        if isinstance(data, _h5py.Dataset) and isinstance(names, _h5py.Dataset):
            arr = _np.array(data)
            # orient to (nCounters, Nframes)
            arrT = arr.T if arr.ndim >= 2 else arr.reshape(1, -1)
            labels = [self._as_str(nm).strip().lower() for nm in names]
            # helper to fetch by label (returns 1D array or None)
            def col(label):
                if label is None:
                    return None
                try:
                    idx = labels.index(label)
                    return arrT[idx]
                except ValueError:
                    return None

            mon = None
            # prefer per-channel monitors on SuperADAM PNR
            if pol in ("uu", "dd", "du", "ud"):
                pol_map = {"uu":"m1","dd":"m2","du":"m3","ud":"m4"}
                mon = col(pol_map.get(pol))
                # treat all-zero as missing
                if isinstance(mon, _np.ndarray) and mon.size and not mon.any():
                    mon = None

            # fallbacks: mon0, mon, monitor
            for lbl in ("mon0","mon","monitor"):
                v = col(lbl)
                if v is not None and mon is None:
                    mon = v

            # time column
            sec = None
            for lbl in ("sec","time","seconds"):
                v = col(lbl)
                if v is not None:
                    sec = v; break

            if mon is not None or sec is not None:
                return mon, sec

    return None, None

H5Resolver.monitor_and_time = _monitor_and_time_safe
# --- END: Safety patch ---



class Ui_MainWindow(QtWidgets.QMainWindow):

    def __create_element(self, object, geometry, objectName, text=None, font=None, placeholder=None, visible=None, stylesheet=None, checked=None, title=None, combo=None, enabled=None):

        object.setObjectName(objectName)

        if not geometry == [999, 999, 999, 999]: object.setGeometry(QtCore.QRect(geometry[0], geometry[1], geometry[2], geometry[3]))

        if not text == None: object.setText(text)
        if not title == None: object.setTitle(title)
        if not font == None: object.setFont(font)
        if not placeholder == None: object.setPlaceholderText(placeholder)
        if not visible == None: object.setVisible(visible)
        if not checked == None: object.setChecked(checked)
        if not enabled == None: object.setEnabled(enabled)

        if not stylesheet == None: object.setStyleSheet(stylesheet)

        if not combo == None:
            for i in combo: object.addItem(str(i))

    ##--> define user interface elements
    def setupUi(self, MainWindow):
        # To preserve original behavior set to True in order to skip first data point on export
        self.export_skip_first_point = True

        # Fonts
        font_headline = QtGui.QFont()
        font_headline.setPointSize(11 if platform.system() == 'Windows' else 12)
        font_headline.setBold(True)

        font_button = QtGui.QFont()
        font_button.setPointSize(10 if platform.system() == 'Windows' else 11)
        font_button.setBold(True)

        font_graphs = QtGui.QFont()
        font_graphs.setPixelSize(11 if platform.system() == 'Windows' else 12)
        font_graphs.setBold(False)

        font_ee = QtGui.QFont()
        font_ee.setPointSize(8 if platform.system() == 'Windows' else 10)
        font_ee.setBold(False)

        # Main Window
        MainWindow.setObjectName("MainWindow")
        MainWindow_size = [1180, 721] if platform.system() == 'Windows' else [1180, 701]
        MainWindow.resize(MainWindow_size[0], MainWindow_size[1])
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        # Allow the application window itself to fit on small laptop screens.
        # The full GUI content is kept scrollable via QScrollArea below.
        MainWindow.setMinimumSize(QtCore.QSize(700, 450))
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowNestedDocks|QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        MainWindow.setWindowTitle("pySAred")

        # when we create .exe with pyinstaller, we need to store icon inside it. Then we find it inside unpacked temp directory.
        self.iconpath = ""  # no icon shipped
        MainWindow.setIconSize(QtCore.QSize(30, 30))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # Keep the designed GUI workspace large enough that controls are not
        # crushed or hidden; small screens will access it through scrollbars.
        self.centralwidget.setMinimumSize(QtCore.QSize(1000, 650))

        # Block: .h5 files
        self.label_h5Scans = QtWidgets.QLabel(self.centralwidget)
        self.__create_element(self.label_h5Scans, [15, 5, 200, 20], "label_h5Scans", text=".h5 files", font=font_headline, stylesheet="QLabel { color : blue; }")
        self.groupBox_data = QtWidgets.QGroupBox(self.centralwidget)
        self.__create_element(self.groupBox_data, [10, 11, 279, 667], "groupBox_data", font=font_ee)
        self.label_dataFiles = QtWidgets.QLabel(self.groupBox_data)
        self.__create_element(self.label_dataFiles, [10, 20, 121, 21], "label_dataFiles", text="Data", font=font_headline)
        self.tableWidget_scans = QtWidgets.QTableWidget(self.groupBox_data)
        self.__create_element(self.tableWidget_scans, [10, 45, 260, 342], "tableWidget_scans", font=font_ee)
        self.tableWidget_scans.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget_scans.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget_scans.setAutoScroll(True)
        self.tableWidget_scans.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.tableWidget_scans.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableWidget_scans.setColumnCount(4)
        self.tableWidget_scans.setRowCount(0)
        headers_table_scans = ["Scan", "DB", "Scan_file_full_path"]
        for i in range(0,3):
            self.tableWidget_scans.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
            self.tableWidget_scans.horizontalHeaderItem(i).setText(headers_table_scans[i])
        self.tableWidget_scans.horizontalHeader().setVisible(True)
        self.tableWidget_scans.verticalHeader().setVisible(False)
        self.tableWidget_scans.setColumnWidth(0, 200)
        self.tableWidget_scans.setColumnWidth(1, int(self.tableWidget_scans.width()) - int(self.tableWidget_scans.columnWidth(0)) - 2)
        self.tableWidget_scans.setColumnWidth(2, 0)
        self.pushButton_deleteScans = QtWidgets.QPushButton(self.groupBox_data)
        self.__create_element(self.pushButton_deleteScans, [10, 390, 81, 20], "pushButton_deleteScans", text="Delete scans", font=font_ee)
        self.pushButton_importScans = QtWidgets.QPushButton(self.groupBox_data)
        self.__create_element(self.pushButton_importScans, [189, 390, 81, 20], "pushButton_importScans", text="Import scans", font=font_ee)
        self.label_DB_files = QtWidgets.QLabel(self.groupBox_data)
        self.__create_element(self.label_DB_files, [10, 415, 191, 23], "label_DB_files", text="Direct Beam(s)", font=font_headline)
        self.checkBox_rearrangeDbAfter = QtWidgets.QCheckBox(self.groupBox_data)
        self.__create_element(self.checkBox_rearrangeDbAfter, [10, 435, 210, 20], "checkBox_rearrangeDbAfter", text="DB's were measured after the scans", font=font_ee)
        self.tableWidget_DB = QtWidgets.QTableWidget(self.groupBox_data)
        self.__create_element(self.tableWidget_DB, [10, 455, 260, 183], "tableWidget_DB", font=font_ee)
        self.tableWidget_DB.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget_DB.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget_DB.setAutoScroll(True)
        self.tableWidget_DB.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.tableWidget_DB.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableWidget_DB.setColumnCount(2)
        self.tableWidget_DB.setRowCount(0)
        headers_table_db = ["Scan", "Path"]
        for i in range(0, 2):
            self.tableWidget_DB.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
            self.tableWidget_DB.horizontalHeaderItem(i).setText(headers_table_db[i])
        self.tableWidget_DB.horizontalHeader().setVisible(False)
        self.tableWidget_DB.verticalHeader().setVisible(False)
        self.tableWidget_DB.setColumnWidth(0, self.tableWidget_DB.width())
        self.tableWidget_DB.setColumnWidth(1, 0)
        self.tableWidget_DB.setSortingEnabled(True)
        self.pushButton_deleteDB = QtWidgets.QPushButton(self.groupBox_data)
        self.__create_element(self.pushButton_deleteDB, [10, 640, 81, 20], "pushButton_deleteDB", text="Delete DB", font=font_ee)
        self.pushButton_importDB = QtWidgets.QPushButton(self.groupBox_data)
        self.__create_element(self.pushButton_importDB, [189, 640, 81, 20], "pushButton_importDB", text="Import DB", font=font_ee)

        # Block: Sample
        self.label_sample = QtWidgets.QLabel(self.centralwidget)
        self.__create_element(self.label_sample, [305, 5, 200, 20], "label_sample", text="Sample", font=font_headline, stylesheet="QLabel { color : blue; }")
        self.groupBox_sampleLen = QtWidgets.QGroupBox(self.centralwidget)
        self.__create_element(self.groupBox_sampleLen, [300, 11, 282, 102], "groupBox_sampleLen", font=font_ee)
        self.label_sampleLen = QtWidgets.QLabel(self.groupBox_sampleLen)
        self.__create_element(self.label_sampleLen, [10, 24, 131, 16], "label_sampleLen", text="Sample length (mm)", font=font_ee)
        self.lineEdit_sampleLen = QtWidgets.QLineEdit(self.groupBox_sampleLen)
        self.__create_element(self.lineEdit_sampleLen, [192, 22, 83, 21], "lineEdit_sampleLen", text="50")
        self.label_sampleShape = QtWidgets.QLabel(self.groupBox_sampleLen)
        self.__create_element(self.label_sampleShape, [10, 48, 131, 16], "label_sampleShape", text="Sample shape", font=font_ee)
        self.comboBox_sampleShape = QtWidgets.QComboBox(self.groupBox_sampleLen)
        self.__create_element(self.comboBox_sampleShape, [132, 46, 143, 21], "comboBox_sampleShape", font=font_ee, combo=["Segment / Rectangle / Square", "Disk / Ellipse", "Square rotated by 45° / Diamond"])
        self.label_sampleDy = QtWidgets.QLabel(self.groupBox_sampleLen)
        self.__create_element(self.label_sampleDy, [10, 72, 171, 16], "label_sampleDy", text="Transverse size D_y (mm)", font=font_ee, visible=False)
        self.lineEdit_sampleDy = QtWidgets.QLineEdit(self.groupBox_sampleLen)
        self.__create_element(self.lineEdit_sampleDy, [192, 70, 83, 21], "lineEdit_sampleDy", text="50", visible=False)

        # Block: Reductions and Instrument settings
        self.label_reductions = QtWidgets.QLabel(self.centralwidget)
        self.__create_element(self.label_reductions, [305, 65, 200, 16], "label_reductions", text="Reductions", font=font_headline, stylesheet="QLabel { color : blue; }")
        self.tabWidget_reductions = QtWidgets.QTabWidget(self.centralwidget)
        self.__create_element(self.tabWidget_reductions, [300, 87, 281, 226], "tabWidget_reductions", font=font_ee)
        self.tabWidget_reductions.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget_reductions.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget_reductions.setElideMode(QtCore.Qt.ElideNone)

        # Tab: Reductions
        self.tab_reductions = QtWidgets.QWidget()
        self.tab_reductions.setObjectName("tab_reductions")
        self.checkBox_reductions_divideByMonitorOrTime = QtWidgets.QCheckBox(self.tab_reductions)
        self.__create_element(self.checkBox_reductions_divideByMonitorOrTime, [10, 10, 131, 18], "checkBox_reductions_divideByMonitorOrTime", font=font_ee, text="Divide by")
        self.comboBox_reductions_divideByMonitorOrTime = QtWidgets.QComboBox(self.tab_reductions)
        self.__create_element(self.comboBox_reductions_divideByMonitorOrTime, [80, 9, 70, 20], "comboBox_reductions_divideByMonitorOrTime", font=font_ee, combo=["monitor", "time"])
        self.checkBox_reductions_normalizeByDB = QtWidgets.QCheckBox(self.tab_reductions)
        self.__create_element(self.checkBox_reductions_normalizeByDB, [10, 35, 181, 18], "checkBox_reductions_normalizeByDB", text="Normalize by direct beam", font=font_ee)
        # User will need Attenuator only with DB. Otherwice I hide this option and replace with Scale factor
        self.checkBox_reductions_attenuatorDB = QtWidgets.QCheckBox(self.tab_reductions)
        self.__create_element(self.checkBox_reductions_attenuatorDB, [10, 60, 161, 18], "checkBox_reductions_attenuatorDB", text="Direct beam attenuator", font=font_ee, checked=True, visible=False)
        self.lineEdit_reductions_attenuatorDB = QtWidgets.QLineEdit(self.tab_reductions)
        self.__create_element(self.lineEdit_reductions_attenuatorDB, [30, 85, 221, 20], "lineEdit_reductions_subtractBkg_Skip", text="", font=font_ee, placeholder="Attenuator correction factor [default 10]", visible=False)
        self.checkBox_reductions_scaleFactor = QtWidgets.QCheckBox(self.tab_reductions)
        self.__create_element(self.checkBox_reductions_scaleFactor, [10, 60, 161, 18], "checkBox_reductions_scaleFactor", text="Scale factor", font=font_ee, checked=False)
        self.lineEdit_reductions_scaleFactor = QtWidgets.QLineEdit(self.tab_reductions)
        self.__create_element(self.lineEdit_reductions_scaleFactor, [30, 85, 221, 20], "lineEdit_reductions_scaleFactor", text="",  font=font_ee, placeholder="Divide reflectivity curve by [default 10]")
        self.checkBox_reductions_subtractBkg = QtWidgets.QCheckBox(self.tab_reductions)
        self.__create_element(self.checkBox_reductions_subtractBkg, [10, 115, 231, 18], "checkBox_reductions_subtractBkg", text="Subtract background (using 1 ROI)", font=font_ee)
        self.lineEdit_reductions_subtractBkg_Skip = QtWidgets.QLineEdit(self.tab_reductions)
        self.__create_element(self.lineEdit_reductions_subtractBkg_Skip, [30, 140, 221, 20], "lineEdit_reductions_subtractBkg_Skip", text="", font=font_ee, placeholder="Skip background corr. at Qz < [default 0]")
        self.checkBox_reductions_overilluminationCorr = QtWidgets.QCheckBox(self.tab_reductions)
        self.__create_element(self.checkBox_reductions_overilluminationCorr, [10, 170, 181, 18], "checkBox_reductions_overilluminationCorr", text="Overillumination correction", font=font_ee)
        self.tabWidget_reductions.addTab(self.tab_reductions, "")
        self.tabWidget_reductions.setTabText(0, "Reductions")

        # Tab: Instrument settings
        self.tab_instrumentSettings = QtWidgets.QWidget()
        self.tab_instrumentSettings.setObjectName("tab_instrumentSettings")
        self.label_instrument_wavelength = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_wavelength, [10, 10, 111, 16], "label_instrument_wavelength", text="Wavelength (A)", font=font_ee)
        self.lineEdit_instrument_wavelength = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_wavelength, [225, 10, 41, 18], "lineEdit_instrument_wavelength", font=font_ee, text="5.183")
        self.label_instrument_wavelengthResolution = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_wavelengthResolution, [10, 33, 271, 16], "label_instrument_wavelengthResolution", text="Wavelength resolution (d_lambda/lambda)", font=font_ee)
        self.lineEdit_instrument_wavelengthResolution = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_wavelengthResolution, [225, 33, 41, 18], "lineEdit_instrument_wavelengthResolution", font=font_ee, text="0.004")
        self.label_instrument_distanceS1ToSample = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_distanceS1ToSample, [10, 56, 241, 16], "label_instrument_distanceS1ToSample", font=font_ee, text="Mono_slit to Samplle distance (mm)")
        self.lineEdit_instrument_distanceS1ToSample = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_distanceS1ToSample, [225, 56, 41, 18], "lineEdit_instrument_distanceS1ToSample", font=font_ee, text="2300")
        self.label_instrument_distanceS2ToSample = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_distanceS2ToSample, [10, 79, 241, 16], "label_instrument_distanceS2ToSample", font=font_ee, text="Sample_slit to Sample distance (mm)")
        self.lineEdit_instrument_distanceS2ToSample = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_distanceS2ToSample, [225, 79, 41, 18], "lineEdit_instrument_distanceS2ToSample", font=font_ee, text="290")
        self.label_instrument_distanceSampleToDetector = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_distanceSampleToDetector, [10, 102, 241, 16], "label_instrument_distanceSampleToDetector", font=font_ee, text="Sample to Detector distance (mm)")
        self.lineEdit_instrument_distanceSampleToDetector = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_distanceSampleToDetector, [225, 102, 41, 18], "lineEdit_instrument_distanceSampleToDetector", font=font_ee, text="2500")
        self.label_instrument_sampleCurvature = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_sampleCurvature, [10, 152, 241, 16], "label_instrument_sampleCurvature", font=font_ee, text="Sample curvature (in ROI) (SFM) (rad)")
        self.lineEdit_instrument_sampleCurvature = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_sampleCurvature, [225, 152, 41, 18], "lineEdit_instrument_sampleCurvature", font=font_ee, text="0")
        self.label_instrument_offsetFull = QtWidgets.QLabel(self.tab_instrumentSettings)
        self.__create_element(self.label_instrument_offsetFull, [10, 175, 241, 16], "label_instrument_offsetFull", font=font_ee, text="Sample angle offset (th - deg)")
        self.lineEdit_instrument_offsetFull = QtWidgets.QLineEdit(self.tab_instrumentSettings)
        self.__create_element(self.lineEdit_instrument_offsetFull, [225, 175, 41, 18], "lineEdit_instrument_offsetFull", font=font_ee, text="0")
        self.tabWidget_reductions.addTab(self.tab_instrumentSettings, "")
        self.tabWidget_reductions.setTabText(1, "Instrument / Corrections")

        # Tab: Export options
        self.tab_exportOptions = QtWidgets.QWidget()
        self.tab_exportOptions.setObjectName("tab_exportOptions")
        self.checkBox_export_addResolutionColumn = QtWidgets.QCheckBox(self.tab_exportOptions)
        self.__create_element(self.checkBox_export_addResolutionColumn, [10, 10, 260, 18], "checkBox_export_addResolutionColumn", text="Include ang. resolution column in the output file", font=font_ee, checked=True)
        self.checkBox_export_resolutionLikeSared = QtWidgets.QCheckBox(self.tab_exportOptions)
        self.__create_element(self.checkBox_export_resolutionLikeSared, [10, 35, 250, 18], "checkBox_export_resolutionLikeSared", text="Use original 'Sared' way for ang. resolution calc.", font=font_ee, checked=False)
        self.checkBox_export_removeZeros = QtWidgets.QCheckBox(self.tab_exportOptions)
        self.__create_element(self.checkBox_export_removeZeros, [10, 60, 250, 18], "checkBox_export_removeZeros", text="Remove zeros from reduced files", font=font_ee, checked=False)
        self.label_export_angle = QtWidgets.QLabel(self.tab_exportOptions)
        self.__create_element(self.label_export_angle, [10, 85, 70, 18], "label_export_angle", font=font_ee, text="Export angle:")
        self.comboBox_export_angle = QtWidgets.QComboBox(self.tab_exportOptions)
        self.__create_element(self.comboBox_export_angle, [85, 84, 70, 20], "comboBox_export_angle", font=font_ee, combo=["Qz", "Degrees", "Radians"])
        self.tabWidget_reductions.addTab(self.tab_exportOptions, "")
        self.tabWidget_reductions.setTabText(2, "Export")

        # Block: Save reduced files at
        self.label_saveAt = QtWidgets.QLabel(self.centralwidget)
        self.__create_element(self.label_saveAt, [305, 320, 200, 20], "label_saveAt", font=font_headline, text="Save reduced files at", stylesheet="QLabel { color : blue; }")
        self.groupBox_saveAt = QtWidgets.QGroupBox(self.centralwidget)
        self.__create_element(self.groupBox_saveAt, [299, 325, 282, 48], "groupBox_saveAt", font=font_ee, title="")
        self.lineEdit_saveAt = QtWidgets.QLineEdit(self.groupBox_saveAt)
        self.__create_element(self.lineEdit_saveAt, [10, 22, 225, 22], "lineEdit_saveAt", font=font_ee, text=self.dir_current)
        self.toolButton_saveAt = QtWidgets.QToolButton(self.groupBox_saveAt)
        self.__create_element(self.toolButton_saveAt, [248, 22, 27, 22], "toolButton_saveAt", font=font_ee, text="...")

        # Button: Clear
        self.pushButton_clear = QtWidgets.QPushButton(self.centralwidget)
        self.__create_element(self.pushButton_clear, [300, 380, 88, 30], "pushButton_clear", font=font_button, text="Clear all")

        # Button: Reduce all
        self.pushButton_reduceAll = QtWidgets.QPushButton(self.centralwidget)
        self.__create_element(self.pushButton_reduceAll, [493, 380, 88, 30], "pushButton_reduceAll", font=font_button, text="Reduce all")

        # Block: Recheck following files in SFM
        self.label_recheckFilesInSFM = QtWidgets.QLabel(self.centralwidget)
        self.__create_element(self.label_recheckFilesInSFM, [305, 490, 250, 20], "label_recheckFilesInSFM", font=font_headline, text="Recheck following files in SFM", stylesheet="QLabel { color : blue; }")
        self.groupBox_recheckFilesInSFM = QtWidgets.QGroupBox(self.centralwidget)
        self.__create_element(self.groupBox_recheckFilesInSFM, [299, 500, 282, 178], "groupBox_recheckFilesInSFM", font=font_ee, title="")
        self.listWidget_recheckFilesInSFM = QtWidgets.QListWidget(self.groupBox_recheckFilesInSFM)
        self.__create_element(self.listWidget_recheckFilesInSFM, [10, 27, 262, 143], "listWidget_recheckFilesInSFM")

        # Block: Single File Mode
        self.label_SFM = QtWidgets.QLabel(self.centralwidget)
        self.__create_element(self.label_SFM, [596, 5, 200, 20], "label_SFM", font=font_headline, text="Single File Mode (SFM)", stylesheet="QLabel { color : blue; }")
        self.groupBox_SFM_scan = QtWidgets.QGroupBox(self.centralwidget)
        self.__create_element(self.groupBox_SFM_scan, [591, 11, 472, 47 ], "groupBox_SFM_scan", font=font_ee)
        self.label_SFM_scan = QtWidgets.QLabel(self.groupBox_SFM_scan)
        self.__create_element(self.label_SFM_scan, [10, 24, 47, 16], "label_SFM_scan", font=font_ee, text="Scan")
        self.comboBox_SFM_scan = QtWidgets.QComboBox(self.groupBox_SFM_scan)
        self.__create_element(self.comboBox_SFM_scan, [40, 22, 300, 21], "comboBox_SFM_scan", font=font_ee)
        self.label_SFM_DB = QtWidgets.QLabel(self.groupBox_SFM_scan)
        self.__create_element(self.label_SFM_DB, [360, 24, 20, 16], "label_SFM_DB", font=font_ee, text="DB")
        self.comboBox_SFM_DB = QtWidgets.QComboBox(self.groupBox_SFM_scan)
        self.__create_element(self.comboBox_SFM_DB, [380, 22, 85, 21], "comboBox_SFM_DB", font=font_ee)
        pg.setConfigOption('background', (255, 255, 255))
        pg.setConfigOption('foreground', 'k')

        # Button: Reduce SFM
        self.pushButton_reduceSFM = QtWidgets.QPushButton(self.centralwidget)
        self.__create_element(self.pushButton_reduceSFM, [1070, 28, 100, 31], "pushButton_reduceSFM", font=font_button, text="Reduce SFM")

        # Block: Detector Images and Reflectivity preview
        self.tabWidget_SFM = QtWidgets.QTabWidget(self.centralwidget)
        self.__create_element(self.tabWidget_SFM, [592, 65, 578, 613], "tabWidget_SFM", font=font_ee)

        # Tab: Detector images
        linedit_size_X = 30
        linedit_size_Y = 18
        self.tab_SFM_detectorImage = QtWidgets.QWidget()
        self.tab_SFM_detectorImage.setObjectName("tab_SFM_detectorImage")
        self.graphicsView_SFM_detectorImage_roi = pg.PlotWidget(self.tab_SFM_detectorImage, viewBox=pg.ViewBox())
        self.__create_element(self.graphicsView_SFM_detectorImage_roi, [0, 450, 577, 90], "graphicsView_SFM_detectorImage_roi")
        self.graphicsView_SFM_detectorImage_roi.hideAxis("left")
        self.graphicsView_SFM_detectorImage_roi.getAxis("bottom").setTickFont(font_graphs)
        self.graphicsView_SFM_detectorImage_roi.getAxis("bottom").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_detectorImage_roi.setMouseEnabled(y=False)
        self.graphicsView_SFM_detectorImage = pg.ImageView(self.tab_SFM_detectorImage, view=pg.PlotItem(viewBox=pg.ViewBox()))
        self.graphicsView_SFM_detectorImage.setGeometry(QtCore.QRect(0, 30, 577, 510))
        self.graphicsView_SFM_detectorImage.setObjectName("graphicsView_SFM_detectorImage")
        self.graphicsView_SFM_detectorImage.ui.histogram.hide()
        self.graphicsView_SFM_detectorImage.ui.menuBtn.hide()
        self.graphicsView_SFM_detectorImage.ui.roiBtn.hide()
        self.graphicsView_SFM_detectorImage.view.showAxis("left", False)
        self.graphicsView_SFM_detectorImage.view.showAxis("bottom", False)
        self.graphicsView_SFM_detectorImage.view.getViewBox().setXLink(self.graphicsView_SFM_detectorImage_roi)
        self.label_SFM_detectorImage_polarisation = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_polarisation, [180, 7, 60, 16], "label_SFM_detectorImage_polarisation", font=font_ee, text="Polarisation")
        self.comboBox_SFM_detectorImage_polarisation = QtWidgets.QComboBox(self.tab_SFM_detectorImage)
        self.__create_element(self.comboBox_SFM_detectorImage_polarisation, [240, 5, 40, 20], "comboBox_SFM_detectorImage_polarisation", font=font_ee)
        self.label_SFM_detectorImage_colorScheme = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_colorScheme, [295, 7, 60, 16], "label_SFM_detectorImage_colorScheme", font=font_ee, text="Colors")
        self.comboBox_SFM_detectorImage_colorScheme = QtWidgets.QComboBox(self.tab_SFM_detectorImage)
        self.__create_element(self.comboBox_SFM_detectorImage_colorScheme, [330, 5, 90, 20], "comboBox_SFM_detectorImage_colorScheme", font=font_ee, combo=["Green / Blue", "White / Black"])
        self.label_SFM_detectorImage_cursor = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_cursor, [430, 7, 40, 16], "label_SFM_detectorImage_cursor", font=font_ee, text="Cursor")
        self.horizontalSlider_SFM_detectorImage_index = QtWidgets.QSlider(self.tab_SFM_detectorImage)
        self.__create_element(self.horizontalSlider_SFM_detectorImage_index, [475, 5, 90, 20], "horizontalSlider_SFM_detectorImage_index")
        self.horizontalSlider_SFM_detectorImage_index.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_SFM_detectorImage_index.setMinimum(0)
        self.horizontalSlider_SFM_detectorImage_index.setMaximum(0)
        self.horizontalSlider_SFM_detectorImage_index.setSingleStep(1)
        self.horizontalSlider_SFM_detectorImage_index.setPageStep(1)
        self.horizontalSlider_SFM_detectorImage_index.setTracking(True)
        self.label_SFM_detectorImage_incidentAngle = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_incidentAngle, [570, 7, 100, 16], "label_SFM_detectorImage_incidentAngle", font=font_ee, text="Incident ang. (deg)")
        self.comboBox_SFM_detectorImage_incidentAngle = QtWidgets.QComboBox(self.tab_SFM_detectorImage)
        self.__create_element(self.comboBox_SFM_detectorImage_incidentAngle, [670, 5, 65, 20], "comboBox_SFM_detectorImage_incidentAngle", font=font_ee)
        self.pushButton_SFM_detectorImage_showIntegratedRoi = QtWidgets.QPushButton(self.tab_SFM_detectorImage)
        self.__create_element(self.pushButton_SFM_detectorImage_showIntegratedRoi, [745, 5, 120, 20], "pushButton_SFM_detectorImage_showIntegratedRoi", font=font_ee, text="Integrated ROI")
        self.label_SFM_detectorImage_roi = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roi, [10, 545, 31, 16], "label_SFM_detectorImage_roi", font=font_ee, text="ROI (")
        self.checkBox_SFM_detectorImage_lockRoi = QtWidgets.QCheckBox(self.tab_SFM_detectorImage)
        self.__create_element(self.checkBox_SFM_detectorImage_lockRoi, [38, 545, 50, 16], "checkBox_SFM_detectorImage_lockRoi", text="lock):", font=font_ee)
        self.label_SFM_detectorImage_roiX_left = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roiX_left, [85, 545, 51, 16], "label_SFM_detectorImage_roiX_left", font=font_ee, text="left")
        self.lineEdit_SFM_detectorImage_roiX_left = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_roiX_left, [115, 544, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_roiX_left", font=font_ee)
        self.label_SFM_detectorImage_roiX_right = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roiX_right, [85, 565, 51, 16], "label_SFM_detectorImage_roiX_right", font=font_ee, text="right")
        self.lineEdit_SFM_detectorImage_roiX_right = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_roiX_right, [115, 564, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_roiX_right", font=font_ee)
        self.label_SFM_detectorImage_roiY_bottom = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roiY_bottom, [155, 545, 51, 16], "label_SFM_detectorImage_roiY_bottom", font=font_ee, text="bottom")
        self.lineEdit_SFM_detectorImage_roiY_bottom = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_roiY_bottom, [195, 544, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_roiY_bottom", font=font_ee)
        self.label_SFM_detectorImage_roiY_top = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roiY_top, [155, 565, 51, 16], "label_SFM_detectorImage_roiY_top", font=font_ee, text="top")
        self.lineEdit_SFM_detectorImage_roiY_top = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_roiY_top, [195, 564, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_roiY_top", font=font_ee)
        self.label_SFM_detectorImage_roi_bkg = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roi_bkg, [245, 545, 47, 16], "label_SFM_detectorImage_roi_bkg", font=font_ee, text="BKG:")
        self.label_SFM_detectorImage_roi_bkgX_left = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roi_bkgX_left, [270, 545, 51, 16], "label_SFM_detectorImage_roi_bkgX_left", font=font_ee, text="left")
        self.lineEdit_SFM_detectorImage_roi_bkgX_left = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_roi_bkgX_left, [300, 544, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_roi_bkgX_left", font=font_ee, enabled=False, stylesheet="color:rgb(0,0,0)")
        self.label_SFM_detectorImage_roi_bkgX_right = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_roi_bkgX_right, [270, 565, 51, 16], "label_SFM_detectorImage_roi_bkgX_right", font=font_ee, text="right")
        self.lineEdit_SFM_detectorImage_roi_bkgX_right = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_roi_bkgX_right, [300, 564, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_roi_bkgX_right", font=font_ee)
        self.label_SFM_detectorImage_time = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_time, [350, 545, 71, 16], "label_SFM_detectorImage_time", font=font_ee, text="Time (s):")
        self.lineEdit_SFM_detectorImage_time = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_time, [400, 544, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_time", font=font_ee, enabled=False, stylesheet="color:rgb(0,0,0)")
        self.label_SFM_detectorImage_slits = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_slits, [450, 545, 51, 16], "label_SFM_detectorImage_slits", font=font_ee, text="Slits (mm):")
        self.label_SFM_detectorImage_slits_s1hg = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_slits_s1hg, [505, 545, 41, 16], "label_SFM_detectorImage_slits_s1hg", font=font_ee, text="s1hg")
        self.lineEdit_SFM_detectorImage_slits_s1hg = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_slits_s1hg, [535, 544, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_slits_s1hg", font=font_ee, enabled=False, stylesheet="color:rgb(0,0,0)")
        self.label_SFM_detectorImage_slits_s2hg = QtWidgets.QLabel(self.tab_SFM_detectorImage)
        self.__create_element(self.label_SFM_detectorImage_slits_s2hg, [505, 565, 30, 16], "label_SFM_detectorImage_slits_s2hg", font=font_ee, text="s2hg")
        self.lineEdit_SFM_detectorImage_slits_s2hg = QtWidgets.QLineEdit(self.tab_SFM_detectorImage)
        self.__create_element(self.lineEdit_SFM_detectorImage_slits_s2hg, [535, 564, linedit_size_X, linedit_size_Y], "lineEdit_SFM_detectorImage_slits_s2hg", font=font_ee, enabled=False, stylesheet="color:rgb(0,0,0)")
        self.tabWidget_SFM.addTab(self.tab_SFM_detectorImage, "")
        self.tabWidget_SFM.setTabText(self.tabWidget_SFM.indexOf(self.tab_SFM_detectorImage), "Detector Image")

        # Tab: Reflectivity preview
        self.tab_SFM_reflectivityPreview = QtWidgets.QWidget()
        self.tab_SFM_reflectivityPreview.setObjectName("tabreflectivity_preview")
        self.graphicsView_SFM_reflectivityPreview = pg.PlotWidget(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.graphicsView_SFM_reflectivityPreview, [0, 20, 577, 540], "graphicsView_SFM_reflectivityPreview")
        self.graphicsView_SFM_reflectivityPreview.getAxis("bottom").setTickFont(font_graphs)
        self.graphicsView_SFM_reflectivityPreview.getAxis("bottom").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_reflectivityPreview.getAxis("left").setTickFont(font_graphs)
        self.graphicsView_SFM_reflectivityPreview.getAxis("left").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_reflectivityPreview.showAxis("top")
        self.graphicsView_SFM_reflectivityPreview.getAxis("top").setTicks([])
        self.graphicsView_SFM_reflectivityPreview.showAxis("right")
        self.graphicsView_SFM_reflectivityPreview.getAxis("right").setTicks([])
        self.checkBox_SFM_reflectivityPreview_showOverillumination = QtWidgets.QCheckBox(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.checkBox_SFM_reflectivityPreview_showOverillumination, [10, 6, 140, 18], "checkBox_SFM_reflectivityPreview_showOverillumination", text="Show Overillumination", font=font_ee)
        self.checkBox_SFM_reflectivityPreview_showZeroLevel = QtWidgets.QCheckBox(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.checkBox_SFM_reflectivityPreview_showZeroLevel, [150, 6, 150, 18], "checkBox_SFM_reflectivityPreview_showZeroLevel", text="Show Zero level", font=font_ee)
        self.label_SFM_reflectivityPreview_view_reflectivity = QtWidgets.QLabel(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.label_SFM_reflectivityPreview_view_reflectivity, [320, 7, 100, 16], "label_SFM_reflectivityPreview_view_reflectivity", text="View: Reflectivity", font=font_ee)
        self.comboBox_SFM_reflectivityPreview_view_reflectivity = QtWidgets.QComboBox(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.comboBox_SFM_reflectivityPreview_view_reflectivity, [410, 5, 50, 20], "comboBox_SFM_reflectivityPreview_view_reflectivity", font=font_ee, combo=["Log", "Lin"])
        self.label_SFM_reflectivityPreview_view_angle = QtWidgets.QLabel(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.label_SFM_reflectivityPreview_view_angle, [470, 7, 50, 16], "label_SFM_reflectivityPreview_view_angle", text="vs Angle", font=font_ee)
        self.comboBox_SFM_reflectivityPreview_view_angle = QtWidgets.QComboBox(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.comboBox_SFM_reflectivityPreview_view_angle, [515, 5, 50, 20], "comboBox_SFM_reflectivityPreview_view_angle", font=font_ee, combo=["Qz", "Deg"])
        self.checkBox_SFM_reflectivityPreview_includeErrorbars = QtWidgets.QCheckBox(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.checkBox_SFM_reflectivityPreview_includeErrorbars, [10, 565, 111, 18], "checkBox_SFM_reflectivityPreview_includeErrorbars", text="Include Error Bars", font=font_ee)
        self.label_SFM_reflectivityPreview_skipPoints_left = QtWidgets.QLabel(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.label_SFM_reflectivityPreview_skipPoints_left, [372, 565, 100, 16], "label_SFM_reflectivityPreview_skipPoints_left", text="Points to skip:  left", font=font_ee)
        self.lineEdit_SFM_reflectivityPreview_skipPoints_left = QtWidgets.QLineEdit(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.lineEdit_SFM_reflectivityPreview_skipPoints_left, [470, 565, linedit_size_X, linedit_size_Y], "lineEdit_SFM_reflectivityPreview_skipPoints_left", text="0", font=font_ee)
        self.label_SFM_reflectivityPreview_skipPoints_right = QtWidgets.QLabel(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.label_SFM_reflectivityPreview_skipPoints_right, [510, 565, 80, 16], "label_SFM_reflectivityPreview_skipPoints_right", text="right", font=font_ee)
        self.lineEdit_SFM_reflectivityPreview_skipPoints_right = QtWidgets.QLineEdit(self.tab_SFM_reflectivityPreview)
        self.__create_element(self.lineEdit_SFM_reflectivityPreview_skipPoints_right, [535, 565, linedit_size_X, linedit_size_Y], "lineEdit_SFM_reflectivityPreview_skipPoints_right", text="0", font=font_ee)
        self.tabWidget_SFM.addTab(self.tab_SFM_reflectivityPreview, "")
        self.tabWidget_SFM.setTabText(self.tabWidget_SFM.indexOf(self.tab_SFM_reflectivityPreview), "Reflectivity preview")

        # Tab: Monitors / Time (preview + export)
        self.tab_SFM_monitors = QtWidgets.QWidget()
        self.tab_SFM_monitors.setObjectName("tab_SFM_monitors")

        # Table
        self.tableWidget_SFM_monitors = QtWidgets.QTableWidget(self.tab_SFM_monitors)
        self.__create_element(self.tableWidget_SFM_monitors, [0, 30, 577, 270], "tableWidget_SFM_monitors", font=font_ee)
        self.tableWidget_SFM_monitors.setColumnCount(0)
        self.tableWidget_SFM_monitors.setRowCount(0)
        self.tableWidget_SFM_monitors.verticalHeader().setVisible(False)
        self.tableWidget_SFM_monitors.horizontalHeader().setVisible(True)
        self.tableWidget_SFM_monitors.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget_SFM_monitors.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tableWidget_SFM_monitors.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.tableWidget_SFM_monitors.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableWidget_SFM_monitors.horizontalHeader().setSectionsClickable(True)
        self.tableWidget_SFM_monitors.verticalHeader().setSectionsClickable(True)
        self.tableWidget_SFM_monitors.horizontalHeader().setHighlightSections(True)
        self.tableWidget_SFM_monitors.verticalHeader().setHighlightSections(True)
        self.tableWidget_SFM_monitors.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableWidget_SFM_monitors.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Caption + Export button
        self.label_SFM_monitors = QtWidgets.QLabel(self.tab_SFM_monitors)
        self.__create_element(self.label_SFM_monitors, [5, 6, 260, 18], "label_SFM_monitors", font=font_ee, text="Preview of monitor / time used by 'Divide by'")
        self.pushButton_SFM_monitors_export = QtWidgets.QPushButton(self.tab_SFM_monitors)
        self.__create_element(self.pushButton_SFM_monitors_export, [445, 2, 120, 25], "pushButton_SFM_monitors_export", font=font_button, text="Export as .dat")

        # -- Plot controls (below the table) --
        self.label_SFM_monitors_xaxis = QtWidgets.QLabel(self.tab_SFM_monitors)
        self.__create_element(self.label_SFM_monitors_xaxis, [5, 305, 20, 18], "label_SFM_monitors_xaxis", font=font_ee, text="X:")
        self.comboBox_SFM_monitors_xaxis = QtWidgets.QComboBox(self.tab_SFM_monitors)
        self.__create_element(self.comboBox_SFM_monitors_xaxis, [25, 303, 90, 22], "comboBox_SFM_monitors_xaxis", font=font_ee, combo=["Index", "th(deg)"])
        self.checkBox_SFM_monitors_logY = QtWidgets.QCheckBox(self.tab_SFM_monitors)
        self.__create_element(self.checkBox_SFM_monitors_logY, [125, 305, 70, 18], "checkBox_SFM_monitors_logY", font=font_ee, text="Log Y")
        self.pushButton_SFM_monitors_plot = QtWidgets.QPushButton(self.tab_SFM_monitors)
        self.__create_element(self.pushButton_SFM_monitors_plot, [200, 301, 100, 24], "pushButton_SFM_monitors_plot", font=font_button, text="Plot selection")
        # --- Y selector button (multi-column chooser) ---
        self.pushButton_SFM_monitors_chooseY = QtWidgets.QPushButton(self.tab_SFM_monitors)
        self.__create_element(self.pushButton_SFM_monitors_chooseY, [310, 301, 80, 24], "pushButton_SFM_monitors_chooseY", font=font_button, text="Y…")
        # --- Plot style ---
        self.label_SFM_monitors_style = QtWidgets.QLabel(self.tab_SFM_monitors)
        self.__create_element(self.label_SFM_monitors_style, [400, 305, 40, 18], "label_SFM_monitors_style", font=font_ee, text="Style:")
        self.comboBox_SFM_monitors_style = QtWidgets.QComboBox(self.tab_SFM_monitors)
        self.__create_element(self.comboBox_SFM_monitors_style, [440, 303, 135, 22], "comboBox_SFM_monitors_style", font=font_ee, combo=["line+scatter", "scatter", "line"])

        # --- separator line between controls and plot ---
        self.frame_SFM_monitors_sep = QtWidgets.QFrame(self.tab_SFM_monitors)
        self.frame_SFM_monitors_sep.setGeometry(QtCore.QRect(0, 327, 577, 2))
        self.frame_SFM_monitors_sep.setFrameShape(QtWidgets.QFrame.HLine)
        self.frame_SFM_monitors_sep.setFrameShadow(QtWidgets.QFrame.Sunken)

        # -- Plot widget --
        self.graphicsView_SFM_monitors_plot = pg.PlotWidget(self.tab_SFM_monitors)
        self.__create_element(self.graphicsView_SFM_monitors_plot, [0, 335, 577, 240], "graphicsView_SFM_monitors_plot")
        self.graphicsView_SFM_monitors_plot.getPlotItem().getAxis("bottom").setTickFont(font_graphs)
        self.graphicsView_SFM_monitors_plot.getPlotItem().getAxis("left").setTickFont(font_graphs)
        self.graphicsView_SFM_monitors_plot.getPlotItem().setLabel('bottom', 'Index')
        self.graphicsView_SFM_monitors_plot.getPlotItem().setLabel('left', 'Value')

        # --- Matplotlib canvas + toolbar for interactive plot (replacing pyqtgraph view here) ---
        self.canvas_SFM_monitors = FigureCanvas(_plt.Figure(figsize=(5.7, 2.4), dpi=100))
        self.canvas_SFM_monitors.setParent(self.tab_SFM_monitors)
        # place toolbar and canvas below the X/Y/Plot controls (controls end ~ y=305)
        self.toolbar_SFM_monitors = NavigationToolbar(self.canvas_SFM_monitors, self.tab_SFM_monitors)
        self.toolbar_SFM_monitors.setGeometry(QtCore.QRect(0, 334, 577, 24))
        self.canvas_SFM_monitors.setGeometry(QtCore.QRect(0, 360, 577, 220))

        # Default: hide the old pg plot; use Matplotlib instead
        try:
            self.graphicsView_SFM_monitors_plot.setVisible(False)
        except Exception:
            pass

        self.tabWidget_SFM.addTab(self.tab_SFM_monitors, "")
        self.tabWidget_SFM.setTabText(self.tabWidget_SFM.indexOf(self.tab_SFM_monitors), "Monitors / Time")

        # Tab: 2D Map
        self.tab_2Dmap = QtWidgets.QWidget()
        self.tab_2Dmap.setObjectName("tab_2Dmap")
        # scaling options are different for different views
        # "scale" for "Qx vs Qz"
        self.checkBox_SFM_2Dmap_flip = QtWidgets.QCheckBox(self.tab_2Dmap)
        self.__create_element(self.checkBox_SFM_2Dmap_flip, [10, 6, 160, 18], "checkBox_SFM_2Dmap_flip", text="Flip (when Analyzer used)", font=font_ee, visible=False)
        self.label_SFM_2Dmap_QxzThreshold = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_QxzThreshold, [5, 7, 220, 16], "label_SFM_2Dmap_QxzThreshold", text="Threshold for the view (number of neutrons):", font=font_ee, visible=False)
        self.comboBox_SFM_2Dmap_QxzThreshold = QtWidgets.QComboBox(self.tab_2Dmap)
        self.__create_element(self.comboBox_SFM_2Dmap_QxzThreshold, [230, 5, 40, 20], "comboBox_SFM_2Dmap_QxzThreshold", font=font_ee, visible=False, combo=[1, 2, 5, 10])
        self.label_SFM_2Dmap_view_scale = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_view_scale, [183, 7, 40, 16], "label_SFM_2Dmap_view_scale", text="View", font=font_ee)
        self.comboBox_SFM_2Dmap_view_scale = QtWidgets.QComboBox(self.tab_2Dmap)
        self.__create_element(self.comboBox_SFM_2Dmap_view_scale, [210, 5, 50, 20], "comboBox_SFM_2Dmap_view_scale", font=font_ee, combo=["Log", "Lin"])
        self.label_SFM_2Dmap_polarisation = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_polarisation, [284, 7, 71, 16], "label_SFM_2Dmap_polarisation", text="Polarisation", font=font_ee)
        self.comboBox_SFM_2Dmap_polarisation = QtWidgets.QComboBox(self.tab_2Dmap)
        self.__create_element(self.comboBox_SFM_2Dmap_polarisation, [344, 5, 40, 20], "comboBox_SFM_2Dmap_polarisation", font=font_ee)
        self.label_SFM_2Dmap_axes = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_axes, [405, 7, 71, 16], "label_SFM_2Dmap_axes", text="Axes", font=font_ee)
        self.comboBox_SFM_2Dmap_axes = QtWidgets.QComboBox(self.tab_2Dmap)
        self.__create_element(self.comboBox_SFM_2Dmap_axes, [435, 5, 130, 20], "comboBox_SFM_2Dmap_axes", font=font_ee, combo=["Pixel vs. Point", "Alpha_i vs. Alpha_f", "Qx vs. Qz"])
        self.graphicsView_SFM_2Dmap = pg.ImageView(self.tab_2Dmap, view=pg.PlotItem())
        self.__create_element(self.graphicsView_SFM_2Dmap, [0, 30, 577, 522], "graphicsView_SFM_2Dmap")
        self.graphicsView_SFM_2Dmap.ui.menuBtn.hide()
        self.graphicsView_SFM_2Dmap.ui.roiBtn.hide()
        colmap = pg.ColorMap(np.array([0, 0.3333, 0.6666, 1]), np.array([[0, 0, 0, 255],[185, 0, 0, 255],[255, 220, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte))
        self.graphicsView_SFM_2Dmap.setColorMap(colmap)
        self.graphicsView_SFM_2Dmap.view.showAxis("left")
        self.graphicsView_SFM_2Dmap.view.getAxis("left").setTickFont(font_graphs)
        self.graphicsView_SFM_2Dmap.view.getAxis("left").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_2Dmap.view.showAxis("bottom")
        self.graphicsView_SFM_2Dmap.view.getAxis("bottom").setTickFont(font_graphs)
        self.graphicsView_SFM_2Dmap.view.getAxis("bottom").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_2Dmap.view.showAxis("top")
        self.graphicsView_SFM_2Dmap.view.getAxis("top").setTicks([])
        self.graphicsView_SFM_2Dmap.view.showAxis("right")
        self.graphicsView_SFM_2Dmap.view.getAxis("right").setTicks([])
        self.graphicsView_SFM_2Dmap.getView().getViewBox().invertY(b=False)

        # 2D map for "Qx vs Qz" is a plot, compared to "Pixel vs Points" which is Image.
        # I rescale graphicsView_SFM_2Dmap_Qxz_theta to show/hide it
        self.graphicsView_SFM_2Dmap_Qxz_theta = pg.PlotWidget(self.tab_2Dmap)
        self.__create_element(self.graphicsView_SFM_2Dmap_Qxz_theta, [0, 0, 0, 0], "graphicsView_SFM_2Dmap_Qxz_theta")
        self.graphicsView_SFM_2Dmap_Qxz_theta.getAxis("bottom").setTickFont(font_graphs)
        self.graphicsView_SFM_2Dmap_Qxz_theta.getAxis("bottom").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_2Dmap_Qxz_theta.getAxis("left").setTickFont(font_graphs)
        self.graphicsView_SFM_2Dmap_Qxz_theta.getAxis("left").setStyle(tickTextOffset=10)
        self.graphicsView_SFM_2Dmap_Qxz_theta.showAxis("top")
        self.graphicsView_SFM_2Dmap_Qxz_theta.getAxis("top").setTicks([])
        self.graphicsView_SFM_2Dmap_Qxz_theta.showAxis("right")
        self.graphicsView_SFM_2Dmap_Qxz_theta.getAxis("right").setTicks([])

        # Lower points (top row)
        self.label_SFM_2Dmap_lowerNumberOfPointsBy = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_lowerNumberOfPointsBy, [5, 30, 211, 16], "label_SFM_2Dmap_lowerNumberOfPointsBy",
                            text="Lower the number of points by factor", font=font_ee, visible=True)
        self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy = QtWidgets.QComboBox(self.tab_2Dmap)
        self.__create_element(self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy, [195, 28, 40, 20], "comboBox_SFM_2Dmap_lowerNumberOfPointsBy",
                            font=font_ee, visible=True, combo=[5, 4, 3, 2, 1])

        # Rescale image (top row, to the right)
        self.label_SFM_2Dmap_rescaleImage_x = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_rescaleImage_x, [260, 30, 85, 16], "label_SFM_2Dmap_rescaleImage_x",
                            text="Rescale image: x", font=font_ee)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_x = QtWidgets.QSlider(self.tab_2Dmap)
        self.__create_element(self.horizontalSlider_SFM_2Dmap_rescaleImage_x, [350, 28, 90, 22], "horizontalSlider_SFM_2Dmap_rescaleImage_x")
        self.horizontalSlider_SFM_2Dmap_rescaleImage_x.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_x.setMinimum(1)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_x.setMaximum(15)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_x.setValue(1)

        self.label_SFM_2Dmap_rescaleImage_y = QtWidgets.QLabel(self.tab_2Dmap)
        self.__create_element(self.label_SFM_2Dmap_rescaleImage_y, [445, 30, 20, 16], "label_SFM_2Dmap_rescaleImage_y",
                            text="y", font=font_ee)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_y = QtWidgets.QSlider(self.tab_2Dmap)
        self.__create_element(self.horizontalSlider_SFM_2Dmap_rescaleImage_y, [465, 28, 90, 22], "horizontalSlider_SFM_2Dmap_rescaleImage_y")
        self.horizontalSlider_SFM_2Dmap_rescaleImage_y.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_y.setMinimum(1)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_y.setMaximum(15)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_y.setValue(1)

        # Export 2D map (we’ll move it to the top row in resizeEvent; this initial pos is OK)
        self.pushButton_SFM_2Dmap_export = QtWidgets.QPushButton(self.tab_2Dmap)
        self.__create_element(self.pushButton_SFM_2Dmap_export, [445, 555, 120, 25], "pushButton_SFM_2Dmap_export",
                            text="Export 2D map", font=font_button)

        self.tabWidget_SFM.addTab(self.tab_2Dmap, "")
        self.tabWidget_SFM.setTabText(self.tabWidget_SFM.indexOf(self.tab_2Dmap), "2D map")


        # StatusBar
        # Put the whole GUI inside a scroll area so that all controls remain
        # reachable on small laptop screens.
        self.scrollArea_main = QtWidgets.QScrollArea(MainWindow)
        self.scrollArea_main.setObjectName("scrollArea_main")
        self.scrollArea_main.setWidgetResizable(True)
        self.scrollArea_main.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea_main.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea_main.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea_main.setWidget(self.centralwidget)
        MainWindow.setCentralWidget(self.scrollArea_main)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # MenuBar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.__create_element(self.menubar, [0, 0, 1000, 21], "menubar")
        self.menu_help = QtWidgets.QMenu(self.menubar)
        self.__create_element(self.menu_help, [999, 999, 999, 999], "menu_help", title="Help")
        MainWindow.setMenuBar(self.menubar)
        self.action_version = QtWidgets.QAction(MainWindow)
        self.__create_element(self.action_version, [999, 999, 999, 999], "action_version", text="V1.5.1")
        self.menu_help.addAction(self.action_version)
        self.menubar.addAction(self.menu_help.menuAction())

        self.tabWidget_reductions.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    ##<--

class GUI(Ui_MainWindow):

    dir_current = ""
    if platform.system() == 'Windows': dir_current = os.getcwd().replace("\\", "/") + "/"
    else:
        for i in sys.argv[0].split("/")[:-4]: dir_current += i + "/"

    def __init__(self):

        super(GUI, self).__init__()
        self.setupUi(self)
        self._apply_layout_refactor()
        self.f_sampleGeometry_ui_update()

        # Some parameters
        self.roiLocked = []
        self.SFM_FILE, self.SFMFileAlreadyAnalized, self.SFMFile2dCalculatedParams = "", "", []  # current file in Single File Mode
        self.SFM_psdUU, self.SFM_psdDU, self.SFM_psdUD, self.SFM_psdDD = [], [], [], []             # 2d arrays of pol detector
        self.th_current = ""                                                                            # current th point
        self.dict_overillCoeff = {}                                                                     # write calculated overillumination coefficients into library
        self.DB_INFO, self.dbAlreadyAnalized = {}, []                                                 # Write DB info into library
        self.roi_draw, self.roi_draw_bkg, self.roi_draw_2Dmap = [], [], []                             # ROI frames
        self.roi_oldCoord_Y, self.roi_draw_int = [], []                                                # Recalc intens if Y roi is changed
        self.trigger_showDetInt = True                                                                # Trigger to switch the detector image view
        self.res_aif = []                                                                               # Alpha_i vs Alpha_f array
        self.sampleCurvature_last = []                                                               # Last sample curvature (lets avoid extra recalcs)
        self.SFM_monitors_Y_selection = []  # List of column indices chosen for Y

        # Triggers
        self.action_version.triggered.connect(self.f_menu_info)

        # Triggers: Buttons
        self.pushButton_importScans.clicked.connect(self.f_button_importRemoveScans)
        self.pushButton_deleteScans.clicked.connect(self.f_button_importRemoveScans)
        self.pushButton_importDB.clicked.connect(self.f_button_importRemoveDB)
        self.pushButton_deleteDB.clicked.connect(self.f_button_importRemoveDB)
        self.toolButton_saveAt.clicked.connect(self.f_button_saveDir)
        self.pushButton_reduceAll.clicked.connect(self.f_button_reduceAll)
        self.pushButton_reduceSFM.clicked.connect(self.f_button_reduceSFM)
        self.pushButton_clear.clicked.connect(self.f_button_clear)
        self.pushButton_SFM_2Dmap_export.clicked.connect(self.f_SFM_2Dmap_export)
        self.pushButton_SFM_detectorImage_showIntegratedRoi.clicked.connect(self.f_SFM_detectorImage_draw)
        self.horizontalSlider_SFM_detectorImage_index.valueChanged.connect(self.f_SFM_detectorImage_sliderChanged)

        # Triggers: LineEdits
        arr_LE_roi = [self.lineEdit_SFM_detectorImage_roiX_left, self.lineEdit_SFM_detectorImage_roiX_right, self.lineEdit_SFM_detectorImage_roiY_bottom, self.lineEdit_SFM_detectorImage_roiY_top, self.lineEdit_SFM_detectorImage_roi_bkgX_right]
        arr_LE_instr = [self.lineEdit_instrument_wavelength, self.lineEdit_instrument_distanceSampleToDetector, self.lineEdit_instrument_sampleCurvature]
        arr_LE_otherParam = [self.lineEdit_sampleLen, self.lineEdit_reductions_attenuatorDB, self.lineEdit_reductions_scaleFactor, self.lineEdit_reductions_subtractBkg_Skip,  self.lineEdit_instrument_wavelengthResolution, self.lineEdit_instrument_distanceS1ToSample, self.lineEdit_instrument_distanceS2ToSample, self.lineEdit_instrument_offsetFull, self.lineEdit_SFM_reflectivityPreview_skipPoints_right, self.lineEdit_SFM_reflectivityPreview_skipPoints_left]

        [i.editingFinished.connect(self.f_SFM_roi_update) for i in arr_LE_roi]
        [i.editingFinished.connect(self.f_SFM_reflectivityPreview_load) for i in arr_LE_otherParam + arr_LE_instr]
        [i.editingFinished.connect(self.f_SFM_2Dmap_draw) for i in arr_LE_instr + arr_LE_roi]

        # Triggers: ComboBoxes
        arr_CoB_detectorImage = [self.comboBox_SFM_detectorImage_incidentAngle, self.comboBox_SFM_detectorImage_polarisation, self.comboBox_SFM_detectorImage_colorScheme]
        arr_CoB_reflectivityPreview = [self.comboBox_reductions_divideByMonitorOrTime, self.comboBox_export_angle, self.comboBox_SFM_DB, self.comboBox_SFM_scan, self.comboBox_SFM_reflectivityPreview_view_angle, self.comboBox_SFM_reflectivityPreview_view_reflectivity]
        arr_CoB_2dmap = [self.comboBox_SFM_2Dmap_QxzThreshold, self.comboBox_SFM_2Dmap_polarisation, self.comboBox_SFM_2Dmap_axes, self.comboBox_SFM_scan,
                        self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy, self.comboBox_SFM_2Dmap_view_scale]

        self.comboBox_SFM_scan.currentIndexChanged.connect(self.f_SFM_detectorImage_load)
        self.comboBox_reductions_divideByMonitorOrTime.currentIndexChanged.connect(self.f_DB_analaze)
        self.tabWidget_SFM.currentChanged.connect(self.f__maybe_refresh_monitors_tab)
        self.pushButton_SFM_monitors_export.clicked.connect(self.f_SFM_monitors_export)
        self.comboBox_SFM_scan.currentIndexChanged.connect(self.f_SFM_monitors_refresh)
        # Monitors table: context menu + header select + plot
        self.tableWidget_SFM_monitors.customContextMenuRequested.connect(self.f_SFM_monitors_contextMenu)
        self.tableWidget_SFM_monitors.horizontalHeader().sectionClicked.connect(self.f_SFM_monitors_selectColumn)
        self.tableWidget_SFM_monitors.verticalHeader().sectionClicked.connect(self.f_SFM_monitors_selectRow)
        self.pushButton_SFM_monitors_plot.clicked.connect(self.f_SFM_monitors_plot_matplotlib)
        self.comboBox_SFM_monitors_xaxis.currentIndexChanged.connect(self.f_SFM_monitors_plot_matplotlib)
        self.checkBox_SFM_monitors_logY.stateChanged.connect(self.f_SFM_monitors_plot_matplotlib)
        self.pushButton_SFM_monitors_chooseY.clicked.connect(self.f_SFM_monitors_chooseY)
        self.comboBox_SFM_monitors_style.currentIndexChanged.connect(self.f_SFM_monitors_plot_matplotlib)
        for i in arr_CoB_detectorImage:
            i.currentIndexChanged.connect(self.f_SFM_detectorImage_draw)
        self.comboBox_SFM_detectorImage_incidentAngle.currentIndexChanged.connect(self.f_SFM_detectorImage_syncSliderFromCombo)
        for i in arr_CoB_reflectivityPreview:
            i.currentIndexChanged.connect(self.f_SFM_reflectivityPreview_load)
        self.comboBox_sampleShape.currentIndexChanged.connect(self.f_sampleGeometry_ui_update)
        self.comboBox_sampleShape.currentIndexChanged.connect(self.f_SFM_reflectivityPreview_load)
        for i in arr_CoB_2dmap:
            i.currentIndexChanged.connect(self.f_SFM_2Dmap_draw)

        self.f_SFM_monitors_refresh()

        # Triggers: CheckBoxes
        self.checkBox_reductions_divideByMonitorOrTime.stateChanged.connect(self.f_DB_analaze)
        self.checkBox_reductions_normalizeByDB.stateChanged.connect(self.f_DB_analaze)
        self.checkBox_SFM_2Dmap_flip.stateChanged.connect(self.f_SFM_2Dmap_draw)

        arr_ChB_reflectivityPreview = [self.checkBox_reductions_divideByMonitorOrTime, self.checkBox_reductions_normalizeByDB, self.checkBox_reductions_attenuatorDB, self.checkBox_reductions_overilluminationCorr, self.checkBox_reductions_subtractBkg, self.checkBox_SFM_reflectivityPreview_showOverillumination, self.checkBox_SFM_reflectivityPreview_showZeroLevel, self.checkBox_SFM_reflectivityPreview_includeErrorbars, self.checkBox_rearrangeDbAfter, self.checkBox_reductions_scaleFactor, self.checkBox_export_resolutionLikeSared, self.checkBox_export_addResolutionColumn]
        [i.stateChanged.connect(self.f_SFM_reflectivityPreview_load) for i in arr_ChB_reflectivityPreview]

        self.checkBox_rearrangeDbAfter.stateChanged.connect(self.f_DB_assign)
        self.checkBox_reductions_subtractBkg.stateChanged.connect(self.f_SFM_detectorImage_draw)

        # Triggers: Sliders
        self.horizontalSlider_SFM_2Dmap_rescaleImage_x.valueChanged.connect(self.f_SFM_2Dmap_draw)
        self.horizontalSlider_SFM_2Dmap_rescaleImage_y.valueChanged.connect(self.f_SFM_2Dmap_draw)

    def _set_layout_margins(self, layout, margins=(8, 8, 8, 8), spacing=8):
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)

    def _wrap_controls_row(self, parent, widgets):
        row = QtWidgets.QWidget(parent)
        lay = QtWidgets.QHBoxLayout(row)
        self._set_layout_margins(lay, (0, 0, 0, 0), 8)
        for item in widgets:
            if item == 'stretch':
                lay.addStretch(1)
            elif isinstance(item, tuple) and len(item) == 2 and item[0] == 'spacing':
                lay.addSpacing(item[1])
            else:
                lay.addWidget(item, 0)
        return row

    def _refactor_simple_groups(self):
        if self.groupBox_sampleLen.layout() is None:
            form = QtWidgets.QFormLayout(self.groupBox_sampleLen)
            self._set_layout_margins(form, (10, 10, 10, 10), 8)
            self.label_sampleLen.setMinimumWidth(150)
            self.lineEdit_sampleLen.setMinimumWidth(90)
            self.lineEdit_sampleLen.setMaximumWidth(120)
            self.comboBox_sampleShape.setMinimumWidth(140)
            form.addRow(self.label_sampleLen, self.lineEdit_sampleLen)
            form.addRow(self.label_sampleShape, self.comboBox_sampleShape)

        if self.groupBox_saveAt.layout() is None:
            lay = QtWidgets.QHBoxLayout(self.groupBox_saveAt)
            self._set_layout_margins(lay, (10, 10, 10, 10), 8)
            lay.addWidget(self.lineEdit_saveAt, 1)
            lay.addWidget(self.toolButton_saveAt, 0)

        if self.groupBox_recheckFilesInSFM.layout() is None:
            lay = QtWidgets.QVBoxLayout(self.groupBox_recheckFilesInSFM)
            self._set_layout_margins(lay, (10, 10, 10, 10), 6)
            lay.addWidget(self.listWidget_recheckFilesInSFM, 1)

        if self.groupBox_SFM_scan.layout() is None:
            lay = QtWidgets.QHBoxLayout(self.groupBox_SFM_scan)
            self._set_layout_margins(lay, (10, 10, 10, 10), 8)
            self.comboBox_SFM_scan.setMinimumWidth(260)
            self.comboBox_SFM_DB.setMinimumWidth(85)
            lay.addWidget(self.label_SFM_scan, 0)
            lay.addWidget(self.comboBox_SFM_scan, 1)
            lay.addSpacing(8)
            lay.addWidget(self.label_SFM_DB, 0)
            lay.addWidget(self.comboBox_SFM_DB, 0)

    def _refactor_data_group(self):
        if self.groupBox_data.layout() is not None:
            return

        self.tableWidget_scans.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tableWidget_DB.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tableWidget_scans.setMinimumHeight(220)
        self.tableWidget_DB.setMinimumHeight(150)

        top_buttons = self._wrap_controls_row(self.groupBox_data, [self.pushButton_deleteScans, 'stretch', self.pushButton_importScans])
        db_buttons = self._wrap_controls_row(self.groupBox_data, [self.pushButton_deleteDB, 'stretch', self.pushButton_importDB])

        lay = QtWidgets.QVBoxLayout(self.groupBox_data)
        self._set_layout_margins(lay, (10, 10, 10, 10), 6)
        lay.addWidget(self.label_dataFiles, 0)
        lay.addWidget(self.tableWidget_scans, 5)
        lay.addWidget(top_buttons, 0)
        lay.addWidget(self.label_DB_files, 0)
        lay.addWidget(self.checkBox_rearrangeDbAfter, 0)
        lay.addWidget(self.tableWidget_DB, 3)
        lay.addWidget(db_buttons, 0)

    def _refactor_reductions_tabs(self):
        if self.tab_reductions.layout() is None:
            lay = QtWidgets.QVBoxLayout(self.tab_reductions)
            self._set_layout_margins(lay, (10, 10, 10, 10), 8)
            row_div = self._wrap_controls_row(self.tab_reductions, [self.checkBox_reductions_divideByMonitorOrTime, self.comboBox_reductions_divideByMonitorOrTime, 'stretch'])
            lay.addWidget(row_div, 0)
            lay.addWidget(self.checkBox_reductions_normalizeByDB, 0)
            lay.addWidget(self.checkBox_reductions_attenuatorDB, 0)
            lay.addWidget(self.lineEdit_reductions_attenuatorDB, 0)
            lay.addWidget(self.checkBox_reductions_scaleFactor, 0)
            lay.addWidget(self.lineEdit_reductions_scaleFactor, 0)
            lay.addWidget(self.checkBox_reductions_subtractBkg, 0)
            lay.addWidget(self.lineEdit_reductions_subtractBkg_Skip, 0)
            lay.addWidget(self.checkBox_reductions_overilluminationCorr, 0)
            lay.addStretch(1)

        if self.tab_instrumentSettings.layout() is None:
            form = QtWidgets.QFormLayout(self.tab_instrumentSettings)
            self._set_layout_margins(form, (10, 10, 10, 10), 8)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            form.addRow(self.label_instrument_wavelength, self.lineEdit_instrument_wavelength)
            form.addRow(self.label_instrument_wavelengthResolution, self.lineEdit_instrument_wavelengthResolution)
            form.addRow(self.label_instrument_distanceS1ToSample, self.lineEdit_instrument_distanceS1ToSample)
            form.addRow(self.label_instrument_distanceS2ToSample, self.lineEdit_instrument_distanceS2ToSample)
            form.addRow(self.label_instrument_distanceSampleToDetector, self.lineEdit_instrument_distanceSampleToDetector)
            form.addRow(self.label_instrument_sampleCurvature, self.lineEdit_instrument_sampleCurvature)
            form.addRow(self.label_instrument_offsetFull, self.lineEdit_instrument_offsetFull)

        if self.tab_exportOptions.layout() is None:
            lay = QtWidgets.QVBoxLayout(self.tab_exportOptions)
            self._set_layout_margins(lay, (10, 10, 10, 10), 8)
            lay.addWidget(self.checkBox_export_addResolutionColumn, 0)
            lay.addWidget(self.checkBox_export_resolutionLikeSared, 0)
            lay.addWidget(self.checkBox_export_removeZeros, 0)
            row = self._wrap_controls_row(self.tab_exportOptions, [self.label_export_angle, self.comboBox_export_angle, 'stretch'])
            lay.addWidget(row, 0)
            lay.addStretch(1)

    def _refactor_detector_tab(self):
        if self.tab_SFM_detectorImage.layout() is not None:
            return

        self.graphicsView_SFM_detectorImage.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.graphicsView_SFM_detectorImage_roi.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.graphicsView_SFM_detectorImage_roi.setMinimumHeight(90)

        top_row = self._wrap_controls_row(
            self.tab_SFM_detectorImage,
            [
                self.label_SFM_detectorImage_polarisation,
                self.comboBox_SFM_detectorImage_polarisation,
                self.label_SFM_detectorImage_colorScheme,
                self.comboBox_SFM_detectorImage_colorScheme,
                'stretch',
                self.pushButton_SFM_detectorImage_showIntegratedRoi,
            ],
        )

        navigation_row = self._wrap_controls_row(
            self.tab_SFM_detectorImage,
            [
                self.label_SFM_detectorImage_cursor,
                self.horizontalSlider_SFM_detectorImage_index,
                self.label_SFM_detectorImage_incidentAngle,
                self.comboBox_SFM_detectorImage_incidentAngle,
                'stretch',
            ],
        )

        info_widget = QtWidgets.QWidget(self.tab_SFM_detectorImage)
        info_grid = QtWidgets.QGridLayout(info_widget)
        self._set_layout_margins(info_grid, (0, 0, 0, 0), 6)
        row = 0
        info_grid.addWidget(self.label_SFM_detectorImage_roi, row, 0)
        info_grid.addWidget(self.checkBox_SFM_detectorImage_lockRoi, row, 1)
        info_grid.addWidget(self.label_SFM_detectorImage_roiX_left, row, 2)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_roiX_left, row, 3)
        info_grid.addWidget(self.label_SFM_detectorImage_roiY_bottom, row, 4)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_roiY_bottom, row, 5)
        info_grid.addWidget(self.label_SFM_detectorImage_roi_bkg, row, 6)
        info_grid.addWidget(self.label_SFM_detectorImage_roi_bkgX_left, row, 7)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_roi_bkgX_left, row, 8)
        info_grid.addWidget(self.label_SFM_detectorImage_time, row, 9)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_time, row, 10)
        info_grid.addWidget(self.label_SFM_detectorImage_slits, row, 11)
        info_grid.addWidget(self.label_SFM_detectorImage_slits_s1hg, row, 12)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_slits_s1hg, row, 13)
        row = 1
        info_grid.addWidget(self.label_SFM_detectorImage_roiX_right, row, 2)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_roiX_right, row, 3)
        info_grid.addWidget(self.label_SFM_detectorImage_roiY_top, row, 4)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_roiY_top, row, 5)
        info_grid.addWidget(self.label_SFM_detectorImage_roi_bkgX_right, row, 7)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_roi_bkgX_right, row, 8)
        info_grid.addWidget(self.label_SFM_detectorImage_slits_s2hg, row, 12)
        info_grid.addWidget(self.lineEdit_SFM_detectorImage_slits_s2hg, row, 13)
        info_grid.setColumnStretch(14, 1)

        lay = QtWidgets.QVBoxLayout(self.tab_SFM_detectorImage)
        self._set_layout_margins(lay, (6, 6, 6, 6), 6)
        lay.addWidget(top_row, 0)
        lay.addWidget(navigation_row, 0)
        lay.addWidget(self.graphicsView_SFM_detectorImage, 1)
        lay.addWidget(self.graphicsView_SFM_detectorImage_roi, 0)
        lay.addWidget(info_widget, 0)

        self.graphicsView_SFM_detectorImage_roi.setVisible(False)

    def _refactor_reflectivity_tab(self):
        if self.tab_SFM_reflectivityPreview.layout() is not None:
            return

        top_row = self._wrap_controls_row(
            self.tab_SFM_reflectivityPreview,
            [
                self.checkBox_SFM_reflectivityPreview_showOverillumination,
                self.checkBox_SFM_reflectivityPreview_showZeroLevel,
                'stretch',
                self.label_SFM_reflectivityPreview_view_reflectivity,
                self.comboBox_SFM_reflectivityPreview_view_reflectivity,
                self.label_SFM_reflectivityPreview_view_angle,
                self.comboBox_SFM_reflectivityPreview_view_angle,
            ],
        )
        bottom_row = self._wrap_controls_row(
            self.tab_SFM_reflectivityPreview,
            [
                self.checkBox_SFM_reflectivityPreview_includeErrorbars,
                'stretch',
                self.label_SFM_reflectivityPreview_skipPoints_left,
                self.lineEdit_SFM_reflectivityPreview_skipPoints_left,
                self.label_SFM_reflectivityPreview_skipPoints_right,
                self.lineEdit_SFM_reflectivityPreview_skipPoints_right,
            ],
        )

        lay = QtWidgets.QVBoxLayout(self.tab_SFM_reflectivityPreview)
        self._set_layout_margins(lay, (6, 6, 6, 6), 6)
        lay.addWidget(top_row, 0)
        lay.addWidget(bottom_row, 0)
        lay.addWidget(self.graphicsView_SFM_reflectivityPreview, 1)

    def _refactor_monitors_tab(self):
        if self.tab_SFM_monitors.layout() is not None:
            return

        header = self._wrap_controls_row(self.tab_SFM_monitors, [self.label_SFM_monitors, 'stretch', self.pushButton_SFM_monitors_export])
        controls = self._wrap_controls_row(
            self.tab_SFM_monitors,
            [
                self.label_SFM_monitors_xaxis,
                self.comboBox_SFM_monitors_xaxis,
                self.checkBox_SFM_monitors_logY,
                self.pushButton_SFM_monitors_plot,
                self.pushButton_SFM_monitors_chooseY,
                'stretch',
                self.label_SFM_monitors_style,
                self.comboBox_SFM_monitors_style,
            ],
        )

        lay = QtWidgets.QVBoxLayout(self.tab_SFM_monitors)
        self._set_layout_margins(lay, (6, 6, 6, 6), 6)
        lay.addWidget(header, 0)
        lay.addWidget(self.tableWidget_SFM_monitors, 3)
        lay.addWidget(controls, 0)
        lay.addWidget(self.frame_SFM_monitors_sep, 0)
        lay.addWidget(self.toolbar_SFM_monitors, 0)
        lay.addWidget(self.canvas_SFM_monitors, 2)

    def _refactor_2d_tab(self):
        if self.tab_2Dmap.layout() is not None:
            return

        row1 = self._wrap_controls_row(
            self.tab_2Dmap,
            [
                self.checkBox_SFM_2Dmap_flip,
                self.label_SFM_2Dmap_QxzThreshold,
                self.comboBox_SFM_2Dmap_QxzThreshold,
                'stretch',
                self.label_SFM_2Dmap_view_scale,
                self.comboBox_SFM_2Dmap_view_scale,
                self.label_SFM_2Dmap_polarisation,
                self.comboBox_SFM_2Dmap_polarisation,
                self.label_SFM_2Dmap_axes,
                self.comboBox_SFM_2Dmap_axes,
            ],
        )
        row2 = self._wrap_controls_row(
            self.tab_2Dmap,
            [
                self.label_SFM_2Dmap_lowerNumberOfPointsBy,
                self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy,
                'stretch',
                self.label_SFM_2Dmap_rescaleImage_x,
                self.horizontalSlider_SFM_2Dmap_rescaleImage_x,
                self.label_SFM_2Dmap_rescaleImage_y,
                self.horizontalSlider_SFM_2Dmap_rescaleImage_y,
                self.pushButton_SFM_2Dmap_export,
            ],
        )

        self._2d_display_container = QtWidgets.QWidget(self.tab_2Dmap)
        self._2d_display_stack = QtWidgets.QStackedLayout(self._2d_display_container)
        self._set_layout_margins(self._2d_display_stack, (0, 0, 0, 0), 0)
        self._2d_display_stack.addWidget(self.graphicsView_SFM_2Dmap)
        self._2d_display_stack.addWidget(self.graphicsView_SFM_2Dmap_Qxz_theta)

        lay = QtWidgets.QVBoxLayout(self.tab_2Dmap)
        self._set_layout_margins(lay, (6, 6, 6, 6), 6)
        lay.addWidget(row1, 0)
        lay.addWidget(row2, 0)
        lay.addWidget(self._2d_display_container, 1)

    def _apply_layout_refactor(self):
        self._refactor_simple_groups()
        self._refactor_data_group()
        self._refactor_reductions_tabs()
        self._refactor_detector_tab()
        self._refactor_reflectivity_tab()
        self._refactor_monitors_tab()
        self._refactor_2d_tab()

        if self.centralwidget.layout() is not None:
            return

        # Prefer 280 px, but allow the splitter/scroll area to work better on small screens.
        self.groupBox_data.setMinimumWidth(240)
        self.groupBox_data.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.tabWidget_reductions.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.groupBox_recheckFilesInSFM.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.tabWidget_SFM.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self._set_layout_margins(main_layout, (10, 8, 10, 8), 10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        # Allow users on small screens to temporarily collapse side panels.
        splitter.setChildrenCollapsible(True)
        main_layout.addWidget(splitter, 1)

        left_pane = QtWidgets.QWidget(splitter)
        middle_pane = QtWidgets.QWidget(splitter)
        right_pane = QtWidgets.QWidget(splitter)

        left_layout = QtWidgets.QVBoxLayout(left_pane)
        middle_layout = QtWidgets.QVBoxLayout(middle_pane)
        right_layout = QtWidgets.QVBoxLayout(right_pane)
        for lay in (left_layout, middle_layout, right_layout):
            self._set_layout_margins(lay, (0, 0, 0, 0), 8)

        left_layout.addWidget(self.label_h5Scans, 0)
        left_layout.addWidget(self.groupBox_data, 1)

        middle_layout.addWidget(self.label_sample, 0)
        middle_layout.addWidget(self.groupBox_sampleLen, 0)
        middle_layout.addWidget(self.label_reductions, 0)
        middle_layout.addWidget(self.tabWidget_reductions, 1)
        middle_layout.addWidget(self.label_saveAt, 0)
        middle_layout.addWidget(self.groupBox_saveAt, 0)

        middle_buttons = self._wrap_controls_row(middle_pane, [self.pushButton_clear, 'stretch', self.pushButton_reduceAll])
        middle_layout.addWidget(middle_buttons, 0)

        middle_layout.addWidget(self.label_recheckFilesInSFM, 0)
        middle_layout.addWidget(self.groupBox_recheckFilesInSFM, 1)

        right_header = self._wrap_controls_row(right_pane, [self.label_SFM, 'stretch', self.pushButton_reduceSFM])
        right_layout.addWidget(right_header, 0)
        right_layout.addWidget(self.groupBox_SFM_scan, 0)
        right_layout.addWidget(self.tabWidget_SFM, 1)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 5)
        splitter.setSizes([290, 290, 600])

        self.setStyleSheet(self.styleSheet() + """
            QGroupBox {
                margin-top: 6px;
            }
            QTableWidget, QListWidget, QTabWidget::pane {
                border-radius: 4px;
            }
            QLineEdit {
                min-height: 22px;
            }
        """)

    def resizeEvent(self, event):
        super(GUI, self).resizeEvent(event)
        try:
            vp_w = max(180, self.tableWidget_scans.viewport().width())
            col0 = min(200, max(140, vp_w - 70))
            self.tableWidget_scans.setColumnWidth(0, col0)
            self.tableWidget_scans.setColumnWidth(1, max(45, vp_w - col0 - 4))
            self.tableWidget_scans.setColumnWidth(2, 0)
            self.tableWidget_DB.setColumnWidth(0, max(120, self.tableWidget_DB.viewport().width()))
            self.tableWidget_DB.setColumnWidth(1, 0)
        except Exception:
            pass

    def _parse_positive_factor(self, raw_text, default_value, field_label):
        txt = str(raw_text).strip() if raw_text is not None else ""
        try:
            value = float(default_value) if txt == "" else float(txt)
        except Exception:
            raise ValueError(f"Error: Recheck '{field_label}' field. Use a finite number > 0.")

        if (not np.isfinite(value)) or value <= 0:
            raise ValueError(f"Error: Recheck '{field_label}' field. Use a finite number > 0.")
        return value

    def _shape_requires_transverse_field(self, shape_mode):
        return False

    def f_sampleGeometry_ui_update(self):
        try:
            shape_mode = self.comboBox_sampleShape.currentText()
        except Exception:
            shape_mode = "Segment / Rectangle / Square"

        if shape_mode == "Disk / Ellipse":
            self.label_sampleLen.setText("Disk diameter / Ellipse major axis (mm)")
        elif shape_mode == "Square rotated by 45° / Diamond":
            self.label_sampleLen.setText("Square diagonal / Diamond longitudinal diagonal (mm)")
        else:
            self.label_sampleLen.setText("Sample length (mm)")

        self.label_sampleDy.setVisible(False)
        self.lineEdit_sampleDy.setVisible(False)
    ##--> menu options
    def f_menu_info(self):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setWindowIcon(QtGui.QIcon(self.iconpath))
        msgBox.setText("pySAred " + self.action_version.text() + "\n\n"
                       "Alexey.Klechikov@gmail.com\n\n"
                       "Check new version at https://github.com/Alexey-Klechikov/pySAred/releases")
        msgBox.exec_()
    ##<--

    ##--> Main window buttons
    def f_button_importRemoveScans(self):

        if self.sender().objectName() == "pushButton_importScans":

            files_import = QtWidgets.QFileDialog().getOpenFileNames(None, "FileNames", self.dir_current, ".h5 (*.h5)")
            if files_import[0] == []: return
            # Next "Import scans" will open last dir
            self.dir_current = files_import[0][0][:files_import[0][0].rfind("/")]

            for FILE in files_import[0]:
                self.tableWidget_scans.insertRow(self.tableWidget_scans.rowCount())
                self.tableWidget_scans.setRowHeight(self.tableWidget_scans.rowCount()-1, 10)
                # File name (row 0) and full path (row 2)
                for j in range(0, 3): self.tableWidget_scans.setItem(self.tableWidget_scans.rowCount()-1, j, QtWidgets.QTableWidgetItem())
                self.tableWidget_scans.item(self.tableWidget_scans.rowCount() - 1, 0).setText(FILE[FILE.rfind("/") + 1:])
                self.tableWidget_scans.item(self.tableWidget_scans.rowCount() - 1, 2).setText(FILE)

                # add file into SFM / Scan ComboBox
                self.comboBox_SFM_scan.addItem(str(FILE[FILE.rfind("/") + 1:]))

                self.f_DB_analaze()
                self.f_SFM_reflectivityPreview_load()

        if self.sender().objectName() == "pushButton_deleteScans":

            files_remove = self.tableWidget_scans.selectedItems()
            if not files_remove: return

            for FILE in files_remove:
                self.tableWidget_scans.removeRow(self.tableWidget_scans.row(FILE))

            # update SFM list
            self.comboBox_SFM_scan.clear()
            for i in range(0, self.tableWidget_scans.rowCount()):
                self.comboBox_SFM_scan.addItem(self.tableWidget_scans.item(i, 2).text()[self.tableWidget_scans.item(i, 2).text().rfind("/") + 1:])

    def f_button_importRemoveDB(self):

        if self.sender().objectName() == "pushButton_importDB":

            files_import = QtWidgets.QFileDialog().getOpenFileNames(None, "FileNames", self.dir_current, ".h5 (*.h5)")
            if files_import[0] == []: return
            # Next "Import scans" will open last dir
            self.dir_current = files_import[0][0][:files_import[0][0].rfind("/")]

            # I couldnt make tablewidget sorting work when adding files to not empty table, so this is the solution for making the list of DB files sorted
            for i in range(self.tableWidget_DB.rowCount()-1, -1, -1):
                files_import[0].append(self.tableWidget_DB.item(i, 1).text())
                self.tableWidget_DB.removeRow(i)

            for FILE in sorted(files_import[0]):
                self.tableWidget_DB.insertRow(self.tableWidget_DB.rowCount())
                self.tableWidget_DB.setRowHeight(self.tableWidget_DB.rowCount()-1, 10)
                # File name (row 0) and full path (row 2)
                for j in range(0, 2): self.tableWidget_DB.setItem(self.tableWidget_DB.rowCount()-1, j, QtWidgets.QTableWidgetItem())
                self.tableWidget_DB.item(self.tableWidget_DB.rowCount() - 1, 0).setText(FILE[FILE.rfind("/") + 1:])
                self.tableWidget_DB.item(self.tableWidget_DB.rowCount() - 1, 1).setText(FILE)

                # add file into SFM / DB ComboBox
                self.comboBox_SFM_DB.addItem(str(FILE[FILE.rfind("/") + 1:][:5]))

            self.f_DB_analaze()
            self.f_SFM_reflectivityPreview_load()

        elif self.sender().objectName() == "pushButton_deleteDB":

            files_remove = self.tableWidget_DB.selectedItems()
            if not files_remove: return

            for FILE in files_remove: self.tableWidget_DB.removeRow(self.tableWidget_DB.row(FILE))

            # update SFM list
            self.comboBox_SFM_DB.clear()
            for i in range(0, self.tableWidget_DB.rowCount()):
                self.comboBox_SFM_DB.addItem(self.tableWidget_DB.item(i, 1).text()[self.tableWidget_DB.item(i, 1).text().rfind("/") + 1:][:5])

            self.f_DB_analaze()

    def f_button_saveDir(self):
        saveAt = QtWidgets.QFileDialog().getExistingDirectory()
        if not saveAt: return
        self.lineEdit_saveAt.setText(str(saveAt) + ("" if str(saveAt)[-1] == "/" else "/"))

    def f_button_reduceAll(self):
        self.listWidget_recheckFilesInSFM.clear()
        bkg_skip = float(self.lineEdit_reductions_subtractBkg_Skip.text()) if self.lineEdit_reductions_subtractBkg_Skip.text() else 0
        dir_saveFile = self.lineEdit_saveAt.text() if self.lineEdit_saveAt.text() else self.dir_current
        if self.statusbar.currentMessage().find("Error") == 0: return

        try:
            self._get_shape_geometry(strict=True)
        except ValueError as ex:
            self.statusbar.showMessage(str(ex))
            return

        try:
            scaleFactor = self._parse_positive_factor(self.lineEdit_reductions_scaleFactor.text(), 10.0, 'Scale Factor') if self.checkBox_reductions_scaleFactor.isChecked() else 1.0
        except ValueError as ex:
            self.statusbar.showMessage(str(ex))
            return

        if self.checkBox_reductions_normalizeByDB.isChecked():
            self.f_DB_analaze()
            try:
                DB_attenFactor = self._parse_positive_factor(self.lineEdit_reductions_attenuatorDB.text(), 10.0, 'Direct Beam Attenuator Factor') if self.checkBox_reductions_attenuatorDB.isChecked() else 1.0
            except ValueError as ex:
                self.statusbar.showMessage(str(ex))
                return
        else:
            DB_attenFactor = 1.0

        # iterate through table with scans
        for i in range(0, self.tableWidget_scans.rowCount()):
            file_full = self.tableWidget_scans.item(i, 2).text()
            file_name = file_full[file_full.rfind("/") + 1: -3]
            FILE_DB = self.tableWidget_scans.item(i, 1).text() if self.checkBox_reductions_normalizeByDB.isChecked() else ""

            with h5py.File(file_full, 'r') as FILE:
                SCAN = FILE[list(FILE.keys())[0]]
                R = H5Resolver(SCAN)

                th_list = R.th_list() or []
                s1hg_list, s2hg_list = R.slit_lists()
                pols = R.pol_list()
                # ROI from file or default from image shape
                stk0 = R.detector_stack(pols[0] if len(pols) > 1 else None)
                H, W = (stk0.shape[1], stk0.shape[2]) if stk0 is not None else (1024, 1024)
                y0, y1, x0, x1 = R.roi((H, W))

                # build Y-integrated stacks per pol
                stacks_intY = {}
                for p in pols:
                    st = R.detector_stack(p if len(pols) > 1 else None)
                    if st is None:
                        stacks_intY[p] = None; continue
                    tmp = []
                    for j in range(st.shape[0]):
                        tmp.append(np.asarray(st[j, y0:y1, :]).sum(axis=0))
                    stacks_intY[p] = np.vstack(tmp)

                # monitors/time per pol
                mon0, sec0 = R.monitor_and_time(None)
                mon_map, time_list = {}, sec0
                for p in pols:
                    mon_map[p], _sec = R.monitor_and_time(p if len(pols) > 1 else None)
                    if mon_map[p] is None: mon_map[p] = mon0
                    if time_list is None: time_list = _sec
                if time_list is None: time_list = np.ones_like(th_list)

                checkThisFile = 0
                for p in pols:
                    if stacks_intY[p] is None: continue
                    scan_intens = stacks_intY[p]   # shape (N, W)

                    # detector name in output consistent with original naming
                    detector_tag = ("psd_" + p) if len(pols) > 1 else "psd"

                    with open(dir_saveFile + file_name + "_" + detector_tag + " (" + FILE_DB + ")" + ".dat", "w") as new_file:
                        for idx, th in enumerate(th_list):
                            th = th - float(self.lineEdit_instrument_offsetFull.text())
                            Qz = (4 * np.pi / float(self.lineEdit_instrument_wavelength.text())) * np.sin(np.radians(th))
                            s1hg, s2hg = (s1hg_list[idx] if s1hg_list is not None else 0), (s2hg_list[idx] if s2hg_list is not None else 0)

                            # intensity in ROI X (+ optional background block identical to original)
                            Intens = float(np.sum(scan_intens[idx][int(x0): int(x1)]))
                            if Intens == 0 and self.checkBox_export_removeZeros.isChecked(): continue
                            IntensErr = 1 if Intens == 0 else np.sqrt(Intens)

                            # "is middle of ROI around Qz~0.02?" check (unchanged)
                            if round(Qz, 3) > 0.015 and round(Qz, 3) < 0.03 and checkThisFile == 0:
                                scanData_0_015 = scan_intens[idx][int(x0): int(x1)]
                                if not max(scanData_0_015) == max(scanData_0_015[round((len(scanData_0_015) / 3)):-round((len(scanData_0_015) / 3))]):
                                    self.listWidget_recheckFilesInSFM.addItem(file_name); checkThisFile = 1

                            overillCorr, FWHM_proj = self.f_overilluminationCorrCoeff(s1hg, s2hg, round(th, 4))
                            if not self.checkBox_reductions_overilluminationCorr.isChecked(): overillCorr = 1

                            # Resolution as before
                            if self.checkBox_export_resolutionLikeSared.isChecked():
                                Resolution = np.sqrt(((2 * np.pi / float(self.lineEdit_instrument_wavelength.text())) ** 2) * ((np.cos(np.radians(th))) ** 2) * (0.68 ** 2) * ((s1hg ** 2) + (s2hg ** 2)) /
                                                    ((float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())) ** 2) +
                                                    ((float(self.lineEdit_instrument_wavelengthResolution.text()) ** 2) * (Qz ** 2)))
                            else:
                                d_alpha = np.arctan((s1hg + [s2hg if FWHM_proj == s2hg else FWHM_proj][0]) /
                                        ((float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())) * 2))
                                if self.comboBox_export_angle.currentText() == "Qz":
                                    k_0 = 2 * np.pi / float(self.lineEdit_instrument_wavelength.text())
                                    Resolution = np.sqrt((k_0 ** 2) * ((((np.cos(np.radians(th))) ** 2) * d_alpha ** 2) + ((float(self.lineEdit_instrument_wavelengthResolution.text()) ** 2) * ((np.sin(np.radians(th))) ** 2))))
                                else:
                                    Resolution = d_alpha if self.comboBox_export_angle.currentText() == "Radians" else np.degrees(d_alpha)
                            Resolution = Resolution / (2 * np.sqrt(2 * np.log(2)))

                            # background subtract
                            if self.checkBox_reductions_subtractBkg.isChecked() and Qz > bkg_skip:
                                w = int(x1) - int(x0)
                                bkg_left0 = int(x0) - 2 * w
                                bkg_left1 = int(x0) - w
                                if bkg_left0 >= 0:
                                    Intens_bkg = float(np.sum(scan_intens[idx][bkg_left0:bkg_left1]))
                                    if Intens_bkg > 0:
                                        IntensErr = np.sqrt(Intens + Intens_bkg)
                                        Intens = Intens - Intens_bkg

                            # monitor/time division
                            if self.checkBox_reductions_divideByMonitorOrTime.isChecked():
                                if self.comboBox_reductions_divideByMonitorOrTime.currentText() == "monitor":
                                    monitor = mon_map[p][idx] if mon_map.get(p) is not None else 1
                                    IntensErr = IntensErr / monitor if Intens == 0 else (Intens / monitor) * np.sqrt((IntensErr / Intens) ** 2 + (1 / monitor))
                                    Intens = Intens / monitor
                                else:
                                    monitor = time_list[idx]
                                    IntensErr = IntensErr / monitor; Intens = Intens / monitor

                            # overillumination correction
                            if self.checkBox_reductions_overilluminationCorr.isChecked() and overillCorr > 0:
                                IntensErr, Intens = IntensErr / overillCorr, Intens / overillCorr

                            # DB normalization
                            if self.checkBox_reductions_normalizeByDB.isChecked():
                                try:
                                    db_key = str(FILE_DB) + ";" + str(s1hg) + ";" + str(s2hg)
                                    DB_intens = float(self.DB_INFO[db_key].split(";")[0]) * DB_attenFactor
                                    DB_err = float(self.DB_INFO[db_key].split(";")[1]) * self.DB_attenFactor
                                    IntensErr = IntensErr + DB_err if Intens == 0 else (Intens / DB_intens) * np.sqrt((DB_err / DB_intens) ** 2 + (IntensErr / Intens) ** 2)
                                    Intens = Intens / DB_intens
                                except:
                                    if checkThisFile == 0:
                                        self.listWidget_recheckFilesInSFM.addItem(file_name); checkThisFile = 1
                                self.checkBox_reductions_scaleFactor.setChecked(False)

                            if self.checkBox_reductions_scaleFactor.isChecked():
                                IntensErr, Intens = IntensErr / scaleFactor, Intens / scaleFactor

                            # angle in output (unchanged)
                            if self.comboBox_export_angle.currentText() == "Qz": angle = round(Qz, 10)
                            elif self.comboBox_export_angle.currentText() == "Degrees": angle = round(np.degrees(np.arcsin(Qz * float(self.lineEdit_instrument_wavelength.text()) / (4 * np.pi))), 10)
                            elif self.comboBox_export_angle.currentText() == "Radians": angle = round(np.arcsin(Qz * float(self.lineEdit_instrument_wavelength.text()) / (4 * np.pi)), 10)

                            if ((idx == 0 and getattr(self, 'export_skip_first_point', True)) or (Intens == 0 and self.checkBox_export_removeZeros.isChecked())):
                                continue
                            if not (np.isfinite(angle) and np.isfinite(Intens) and np.isfinite(IntensErr) and np.isfinite(Resolution)):
                                continue

                            new_file.write(f"{angle} {Intens} {IntensErr} ")
                            if self.checkBox_export_addResolutionColumn.isChecked(): new_file.write(str(Resolution))
                            new_file.write('\n')

                    # if file empty, write the message as before
                    out_path = dir_saveFile + file_name + "_" + detector_tag + " (" + FILE_DB + ")" + ".dat"
                    if os.stat(out_path).st_size == 0:
                        with open(out_path, "w") as empty_file:
                            empty_file.write("All points are either zeros or negatives.")

        self.statusbar.showMessage(str(self.tableWidget_scans.rowCount()) + " files reduced, " + str(self.listWidget_recheckFilesInSFM.count()) + " file(s) might need extra care.")

    def f_button_reduceSFM(self):

        dir_saveFile = self.lineEdit_saveAt.text() if self.lineEdit_saveAt.text() else self.dir_current

        # polarisation order - uu, dd, ud, du
        detector = ["uu", "du", "ud", "dd"]

        for i in range(0, len(self.SFM_export_Qz)):

            SFM_DB_file_export = self.SFM_DB_FILE if self.checkBox_reductions_normalizeByDB.isChecked() else ""

            with open(dir_saveFile + self.SFM_FILE[self.SFM_FILE.rfind("/") + 1 : -3] + "_" + str(detector[i]) + " (" + SFM_DB_file_export + ")" + " SFM.dat", "w") as new_file:
                for j in range(0, len(self.SFM_export_Qz[i])):
                    if self.SFM_export_I[i][j] == 0 and self.checkBox_export_removeZeros.isChecked(): continue

                    if self.comboBox_export_angle.currentText() == "Qz": angle = round(self.SFM_export_Qz[i][j], 10)
                    elif self.comboBox_export_angle.currentText() == "Degrees":
                        angle = round(np.degrees(np.arcsin(self.SFM_export_Qz[i][j] * float(self.lineEdit_instrument_wavelength.text())/ (4* np.pi))), 10)
                    elif self.comboBox_export_angle.currentText() == "Radians":
                        angle = round(np.arcsin(self.SFM_export_Qz[i][j] * float(self.lineEdit_instrument_wavelength.text()) / (4 * np.pi)), 10)

                    new_file.write(str(angle) + ' ' + str(self.SFM_export_I[i][j]) + ' ' + str(self.SFM_export_dI[i][j]) + ' ')

                    if self.checkBox_export_addResolutionColumn.isChecked(): new_file.write(str(self.SFM_export_resolution[i][j]))
                    new_file.write('\n')

            # close new file
            new_file.close()

        self.statusbar.showMessage(self.SFM_FILE[self.SFM_FILE.rfind("/") + 1:] + " file is reduced in SFM.")

    def f_button_clear(self):

        for item in (self.comboBox_SFM_scan, self.listWidget_recheckFilesInSFM, self.graphicsView_SFM_detectorImage, self.graphicsView_SFM_2Dmap, self.graphicsView_SFM_reflectivityPreview.getPlotItem(),self.comboBox_SFM_detectorImage_incidentAngle, self.comboBox_SFM_detectorImage_polarisation, self.comboBox_SFM_2Dmap_polarisation):
            item.clear()
        try:
            self.horizontalSlider_SFM_detectorImage_index.blockSignals(True)
            self.horizontalSlider_SFM_detectorImage_index.setMinimum(0)
            self.horizontalSlider_SFM_detectorImage_index.setMaximum(0)
            self.horizontalSlider_SFM_detectorImage_index.setValue(0)
        finally:
            self.horizontalSlider_SFM_detectorImage_index.blockSignals(False)

        for i in range(self.tableWidget_scans.rowCount(), -1, -1): self.tableWidget_scans.removeRow(i)
        for i in range(self.tableWidget_DB.rowCount(), -1, -1): self.tableWidget_DB.removeRow(i)
    ##<--

    ##--> extra functions to shorten the code
    def _sample_shape_mode(self):
        try:
            return self.comboBox_sampleShape.currentText()
        except Exception:
            return "Segment / Rectangle / Square"

    def _shape_integration_difference_significant(self, coarse_value, fine_value):
        coarse_value = float(coarse_value)
        fine_value = float(fine_value)
        threshold = max(1e-7, 1e-4 * max(abs(fine_value), 1e-3))
        return abs(fine_value - coarse_value) > threshold

    def _get_shape_geometry(self, strict=False):
        shape_mode = self._sample_shape_mode()

        try:
            sample_len = float(self.lineEdit_sampleLen.text())
        except Exception:
            if strict:
                raise ValueError("Error: Recheck 'Sample length' field. Use a finite number > 0.")
            return None, shape_mode, None

        if (not np.isfinite(sample_len)) or sample_len <= 0:
            if strict:
                raise ValueError("Error: Recheck 'Sample length' field. Use a finite number > 0.")
            return None, shape_mode, None

        return sample_len, shape_mode, None

    def _overillumination_segment_coeff(self, sample_len, AO, OB, OC, FWHM_beam, s2hg, th):
        coeff = [0.0, 0.0]
        BC = OC - OB
        sampleLen_relative = float(sample_len) * np.sin(np.radians(np.fabs(th if not th == 0 else 0.00001)))

        if sampleLen_relative / 2 >= OC:
            coeff[0] = 1.0
        else:
            if sampleLen_relative / 2 <= OB:
                coeff[0] = AO * sampleLen_relative / 2
            else:
                coeff[0] = AO * (OB + BC / 2 - ((OC - sampleLen_relative / 2) ** 2) / (2 * BC))

        coeff[1] = s2hg if sampleLen_relative / 2 >= FWHM_beam else sampleLen_relative
        return coeff

    def _overillumination_segment_coeff0_array(self, sample_len_array, AO, OB, OC, th):
        BC = OC - OB
        th_eff = np.fabs(th if not th == 0 else 0.00001)
        sampleLen_relative = np.asarray(sample_len_array, dtype=float) * np.sin(np.radians(th_eff))
        a = sampleLen_relative / 2.0
        coeff0 = np.empty_like(a, dtype=float)

        mask_full = a >= OC
        mask_linear = (~mask_full) & (a <= OB)
        mask_quad = (~mask_full) & (~mask_linear)

        coeff0[mask_full] = 1.0
        coeff0[mask_linear] = AO * a[mask_linear]
        coeff0[mask_quad] = AO * (OB + BC / 2.0 - ((OC - a[mask_quad]) ** 2) / (2.0 * BC))
        return coeff0

    def _integrated_shape_coeff0(self, sample_len, sample_dy, AO, OB, OC, th, shape_mode, n_points):
        n_points = max(3, int(n_points))

        if shape_mode == "Disk / Ellipse":
            radius = sample_len / 2.0
            y = np.linspace(0.0, radius, n_points)
            local_lengths = 2.0 * np.sqrt(np.clip(radius * radius - y * y, 0.0, None))
            local_coeff = self._overillumination_segment_coeff0_array(local_lengths, AO, OB, OC, th)
            coeff0 = (2.0 / sample_len) * float(np.trapezoid(local_coeff, y))
        elif shape_mode == "Square rotated by 45° / Diamond":
            half_diag = sample_len / 2.0
            y = np.linspace(0.0, half_diag, n_points)
            local_lengths = 2.0 * np.clip(half_diag - y, 0.0, None)
            local_coeff = self._overillumination_segment_coeff0_array(local_lengths, AO, OB, OC, th)
            coeff0 = (2.0 / sample_len) * float(np.trapezoid(local_coeff, y))
        else:
            coeff0 = float(self._overillumination_segment_coeff(sample_len, AO, OB, OC, 0.0, 0.0, th)[0])

        return float(np.clip(coeff0, 0.0, 1.0))

    def _average_segment_coeff_for_shape(self, sample_len, sample_dy, AO, OB, OC, FWHM_beam, s2hg, th, shape_mode):
        base_coeff = self._overillumination_segment_coeff(sample_len, AO, OB, OC, FWHM_beam, s2hg, th)

        if shape_mode == "Segment / Rectangle / Square":
            return base_coeff

        coeff0_1500 = self._integrated_shape_coeff0(sample_len, sample_dy, AO, OB, OC, th, shape_mode, 1500)
        coeff0_2500 = self._integrated_shape_coeff0(sample_len, sample_dy, AO, OB, OC, th, shape_mode, 2500)

        if self._shape_integration_difference_significant(coeff0_1500, coeff0_2500):
            coeff0 = self._integrated_shape_coeff0(sample_len, sample_dy, AO, OB, OC, th, shape_mode, 4000)
        else:
            coeff0 = coeff0_1500

        return [coeff0, base_coeff[1]]

    def f_overilluminationCorrCoeff(self, s1hg, s2hg, th):

        sample_len, shape_mode, sample_dy = self._get_shape_geometry(strict=False)
        if sample_len is None:
            return [1, s2hg]
        config = str(s1hg) + " " + str(s2hg) + " " + str(th) + " " + str(sample_len) + " " + str(shape_mode) + " " + str(sample_dy) + " " + self.lineEdit_instrument_distanceS1ToSample.text() + " " + self.lineEdit_instrument_distanceS2ToSample.text()

        if config in self.dict_overillCoeff:
            coeff = self.dict_overillCoeff[config]
        else:
            coeff = [0.0, 0.0]

            if s1hg < s2hg:
                OB = abs(((float(self.lineEdit_instrument_distanceS1ToSample.text()) * (s2hg - s1hg)) / (2 * (float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())))) + s1hg / 2)
                OC = ((float(self.lineEdit_instrument_distanceS1ToSample.text()) * (s2hg + s1hg)) / (2 * (float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())))) - s1hg / 2
            elif s1hg > s2hg:
                OB = abs(((s2hg * float(self.lineEdit_instrument_distanceS1ToSample.text())) - (s1hg * float(self.lineEdit_instrument_distanceS2ToSample.text()))) / (2 * (float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text()))))
                OC = (float(self.lineEdit_instrument_distanceS1ToSample.text()) / (float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text()))) * (s2hg + s1hg) / 2 - (s1hg / 2)
            else:
                OB = s1hg / 2
                OC = s1hg * (float(self.lineEdit_instrument_distanceS1ToSample.text()) / (float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())) - 1 / 2)

            BC = OC - OB
            AO = 1 / (BC / 2 + OB)
            FWHM_beam = BC / 2 + OB

            coeff = self._average_segment_coeff_for_shape(sample_len, sample_dy, AO, OB, OC, FWHM_beam, s2hg, th, shape_mode)
            self.dict_overillCoeff[config] = coeff

        return coeff

    def f_DB_analaze(self):
        self.DB_INFO = {}

        for i in range(0, self.tableWidget_DB.rowCount()):
            with h5py.File(self.tableWidget_DB.item(i,1).text(), 'r') as FILE_DB:
                SCAN = FILE_DB[list(FILE_DB.keys())[0]]
                R = H5Resolver(SCAN)

                th_list = R.th_list() or []
                s1hg_list, s2hg_list = R.slit_lists()
                mon_list, time_list = R.monitor_and_time(None)
                if mon_list is None: mon_list = np.ones_like(th_list)
                if time_list is None: time_list = np.ones_like(th_list)

                # ROI counter if available; else compute from images using same ROI logic as scans
                intens_list = None
                for scalers in [g for g in R._scalers_candidates()]:
                    d = R._ds(scalers, "data"); names = R._ds(scalers, "SPEC_counter_mnemonics")
                    if isinstance(d, h5py.Dataset) and isinstance(names, h5py.Dataset):
                        arr = np.array(d).T
                        labels = [R._as_str(x).strip().lower() for x in names]
                        if "roi" in labels:
                            intens_list = arr[labels.index("roi")]
                            break

                if intens_list is None:
                    stack = R.detector_stack(None)
                    if stack is not None:
                        H, W = stack.shape[1], stack.shape[2]
                        y0, y1, x0, x1 = R.roi((H, W))
                        vals = []
                        for k in range(stack.shape[0]):
                            vals.append(np.asarray(stack[k, y0:y1, x0:x1]).sum())
                        intens_list = np.array(vals)
                    else:
                        intens_list = np.ones_like(th_list)  # safe fallback

                # monitor selector
                if self.checkBox_reductions_divideByMonitorOrTime.isChecked():
                    monitor = mon_list if self.comboBox_reductions_divideByMonitorOrTime.currentText() == "monitor" else time_list
                else:
                    monitor = np.ones_like(intens_list)

                for j in range(0, len(th_list)):
                    DB_intens = float(intens_list[j]) / float(monitor[j])
                    if self.checkBox_reductions_divideByMonitorOrTime.isChecked() and self.comboBox_reductions_divideByMonitorOrTime.currentText() == "monitor":
                        DB_err = DB_intens * np.sqrt(1/max(float(intens_list[j]),1.0) + 1/max(float(monitor[j]),1.0))
                    else:
                        DB_err = np.sqrt(max(float(intens_list[j]),1.0)) / float(monitor[j])

                    scan_slitsMonitor = self.tableWidget_DB.item(i, 0).text()[:5] + ";" + str(s1hg_list[j]) + ";" + str(s2hg_list[j])
                    self.DB_INFO[scan_slitsMonitor] = f"{DB_intens};{DB_err}"

        if self.tableWidget_DB.rowCount() == 0: return
        else: self.f_DB_assign()

    def f_DB_assign(self):

        DB_list = []
        for DB_scan_number in self.DB_INFO: DB_list.append(DB_scan_number.split(";")[0])

        for i in range(self.tableWidget_scans.rowCount()):
            scan_number = self.tableWidget_scans.item(i, 0).text()[:5]

            # find nearest DB file if there are several of them
            if len(DB_list) == 0: FILE_DB = ""
            elif len(DB_list) == 1: FILE_DB = DB_list[0][:5]
            else:
                if self.checkBox_rearrangeDbAfter.isChecked():
                    for j, DB_scan in enumerate(DB_list):
                        FILE_DB = DB_scan[:5]
                        if int(DB_scan[:5]) > int(scan_number[:5]): break
                else:
                    for j, DB_scan in enumerate(reversed(DB_list)):
                        FILE_DB = DB_scan[:5]
                        if int(DB_scan[:5]) < int(scan_number[:5]): break

            self.tableWidget_scans.item(i, 1).setText(FILE_DB)

    ##<--

    ##--> SFM
    def f_SFM_detectorImage_load(self):
        if self.comboBox_SFM_scan.currentText() == "": return

        self.comboBox_SFM_detectorImage_incidentAngle.clear()
        self.comboBox_SFM_detectorImage_polarisation.clear()
        self.comboBox_SFM_2Dmap_polarisation.clear()
        try:
            self.horizontalSlider_SFM_detectorImage_index.blockSignals(True)
            self.horizontalSlider_SFM_detectorImage_index.setMinimum(0)
            self.horizontalSlider_SFM_detectorImage_index.setMaximum(0)
            self.horizontalSlider_SFM_detectorImage_index.setValue(0)
        finally:
            self.horizontalSlider_SFM_detectorImage_index.blockSignals(False)

        # Resolve full path for selected SFM file
        for i in range(0, self.tableWidget_scans.rowCount()):
            if self.tableWidget_scans.item(i, 0).text() == self.comboBox_SFM_scan.currentText():
                self.SFM_FILE = self.tableWidget_scans.item(i, 2).text()
                break

        with h5py.File(self.SFM_FILE, 'r') as FILE:
            SCAN = FILE[list(FILE.keys())[0]]
            R = H5Resolver(SCAN)

            # instrument-sensitive slit labels in UI
            if R.instrument_name() == "MiniADAM":
                self.label_SFM_detectorImage_slits_s1hg.setText("s3hg")
                self.label_SFM_detectorImage_slits_s2hg.setText("s4hg")
            else:
                self.label_SFM_detectorImage_slits_s1hg.setText("s1hg")
                self.label_SFM_detectorImage_slits_s2hg.setText("s2hg")

            # get an image stack for shape and ROI
            stack = R.detector_stack(pol=None)
            H, W = (stack.shape[1], stack.shape[2]) if stack is not None else (1024, 1024)

            if not self.roiLocked == [] and self.checkBox_SFM_detectorImage_lockRoi.isChecked():
                original_roi_coord = np.array(self.roiLocked[0], dtype=object)
                original_roi_coord = [int(str(x).split(".")[0]) for x in original_roi_coord]
            else:
                original_roi_coord = R.roi((H, W))

            roi_width = int(original_roi_coord[3]) - int(original_roi_coord[2])

            # ROI fields
            self.lineEdit_SFM_detectorImage_roiX_left.setText(str(original_roi_coord[2]))
            self.lineEdit_SFM_detectorImage_roiX_right.setText(str(original_roi_coord[3]))
            self.lineEdit_SFM_detectorImage_roiY_bottom.setText(str(original_roi_coord[1]))
            self.lineEdit_SFM_detectorImage_roiY_top.setText(str(original_roi_coord[0]))

            # BKG ROI (left of the main ROI by one ROI width)
            if not self.roiLocked == [] and self.checkBox_SFM_detectorImage_lockRoi.isChecked():
                self.lineEdit_SFM_detectorImage_roi_bkgX_right.setText(str(self.roiLocked[1]))
            else:
                self.lineEdit_SFM_detectorImage_roi_bkgX_right.setText(str(int(self.lineEdit_SFM_detectorImage_roiX_left.text()) - roi_width))
            self.lineEdit_SFM_detectorImage_roi_bkgX_left.setText(str(int(self.lineEdit_SFM_detectorImage_roi_bkgX_right.text()) - roi_width))

            # Incident angles list (th)
            th = R.th_list() or []
            for v in th:
                self.comboBox_SFM_detectorImage_incidentAngle.addItem(str(round(float(v), 3)))
            try:
                self.horizontalSlider_SFM_detectorImage_index.blockSignals(True)
                if len(th) > 0:
                    self.horizontalSlider_SFM_detectorImage_index.setMinimum(0)
                    self.horizontalSlider_SFM_detectorImage_index.setMaximum(len(th) - 1)
                    self.horizontalSlider_SFM_detectorImage_index.setEnabled(True)
                    self.horizontalSlider_SFM_detectorImage_index.setValue(min(self.comboBox_SFM_detectorImage_incidentAngle.currentIndex(), len(th) - 1))
                else:
                    self.horizontalSlider_SFM_detectorImage_index.setMinimum(0)
                    self.horizontalSlider_SFM_detectorImage_index.setMaximum(0)
                    self.horizontalSlider_SFM_detectorImage_index.setValue(0)
                    self.horizontalSlider_SFM_detectorImage_index.setEnabled(False)
            finally:
                self.horizontalSlider_SFM_detectorImage_index.blockSignals(False)

            # Polarisation list
            for p in R.pol_list():
                for cb in (self.comboBox_SFM_detectorImage_polarisation, self.comboBox_SFM_2Dmap_polarisation):
                    cb.addItem(p)

            self.comboBox_SFM_detectorImage_polarisation.setCurrentIndex(0)
            self.comboBox_SFM_2Dmap_polarisation.setCurrentIndex(0)

    def f_SFM_detectorImage_sliderChanged(self, value):
        try:
            if self.comboBox_SFM_detectorImage_incidentAngle.count() == 0:
                return
            value = max(0, min(int(value), self.comboBox_SFM_detectorImage_incidentAngle.count() - 1))
            if self.comboBox_SFM_detectorImage_incidentAngle.currentIndex() != value:
                self.comboBox_SFM_detectorImage_incidentAngle.setCurrentIndex(value)
        except Exception:
            pass

    def f_SFM_detectorImage_syncSliderFromCombo(self, index):
        try:
            if self.horizontalSlider_SFM_detectorImage_index.maximum() < 0:
                return
            self.horizontalSlider_SFM_detectorImage_index.blockSignals(True)
            self.horizontalSlider_SFM_detectorImage_index.setValue(max(0, int(index)))
        except Exception:
            pass
        finally:
            try:
                self.horizontalSlider_SFM_detectorImage_index.blockSignals(False)
            except Exception:
                pass

    def f_SFM_monitors_refresh(self):
        """Rebuild the Monitors/Time table for the currently selected SFM file.
        Uses the same resolver + selection logic as the 'Divide by' normalization."""
        try:
            self.tableWidget_SFM_monitors.clear()
            self.tableWidget_SFM_monitors.setRowCount(0); self.tableWidget_SFM_monitors.setColumnCount(0)
            if self.comboBox_SFM_scan.currentText() == "": return

            # resolve path of selected SFM file
            for i in range(0, self.tableWidget_scans.rowCount()):
                if self.tableWidget_scans.item(i, 0).text() == self.comboBox_SFM_scan.currentText():
                    self.SFM_FILE = self.tableWidget_scans.item(i, 2).text()
                    break

            with h5py.File(self.SFM_FILE, 'r') as FILE:
                SCAN = FILE[list(FILE.keys())[0]]
                R = H5Resolver(SCAN)
                th_list = R.th_list() or []
                pols = R.pol_list()

                # Build columns: idx, th(deg), then for each pol: mon_<pol>, time_<pol>, and finally slits at the end
                s1_list, s2_list = R.slit_lists()
                # Label slits based on instrument identity
                if R.instrument_name() == "MiniADAM":
                    s1_hdr, s2_hdr = "s3hg(mm)", "s4hg(mm)"
                else:
                    s1_hdr, s2_hdr = "s1hg(mm)", "s2hg(mm)"

                headers = ["idx", "th(deg)"]
                if len(pols) > 1:
                    for p in pols:
                        headers += [f"mon_{p}", f"time_{p}"]
                else:
                    headers += ["mon_uu", "time_uu"]  # NR path

                # append slits at the very end
                headers += [s1_hdr, s2_hdr]

                self.tableWidget_SFM_monitors.setColumnCount(len(headers))
                for c, h in enumerate(headers):
                    self.tableWidget_SFM_monitors.setHorizontalHeaderItem(c, QtWidgets.QTableWidgetItem(h))

                # Obtain per-pol monitors/time exactly like the Divide-by code
                # (same fallback: mon_p[p] or mon0; time uses the 'None' request)
                mon0, sec0 = R.monitor_and_time(None)
                mon_p = {}
                time_any = sec0  # used for time columns when pol-specific time is missing

                for p in pols:
                    m, s = R.monitor_and_time(p if len(pols) > 1 else None)
                    if m is None: m = mon0
                    mon_p[p] = m
                    if time_any is None: time_any = s

                if time_any is None:
                    # final fallback like in reductions: ones with length of th_list
                    time_any = np.ones_like(np.array(th_list, dtype=float))

                # Fill rows
                N = len(th_list)
                self.tableWidget_SFM_monitors.setRowCount(N)
                for i in range(N):
                    # idx, th
                    self.tableWidget_SFM_monitors.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
                    th_val = round(float(th_list[i]), 6) if len(th_list) > i else ""
                    self.tableWidget_SFM_monitors.setItem(i, 1, QtWidgets.QTableWidgetItem(str(th_val)))

                    # monitors/time start right after idx,th
                    col = 2
                    if len(pols) > 1:
                        for p in pols:
                            mon_arr = mon_p.get(p)
                            mon_val = "" if mon_arr is None or len(mon_arr) <= i else str(mon_arr[i])
                            t_val = "" if time_any is None or len(time_any) <= i else str(time_any[i])
                            self.tableWidget_SFM_monitors.setItem(i, col, QtWidgets.QTableWidgetItem(mon_val)); col += 1
                            self.tableWidget_SFM_monitors.setItem(i, col, QtWidgets.QTableWidgetItem(t_val)); col += 1
                    else:
                        mon_arr = mon_p.get("uu") if "uu" in mon_p else mon_p.get(pols[0])
                        mon_val = "" if mon_arr is None or len(mon_arr) <= i else str(mon_arr[i])
                        t_val = "" if time_any is None or len(time_any) <= i else str(time_any[i])
                        self.tableWidget_SFM_monitors.setItem(i, col, QtWidgets.QTableWidgetItem(mon_val)); col += 1
                        self.tableWidget_SFM_monitors.setItem(i, col, QtWidgets.QTableWidgetItem(t_val)); col += 1

                    # Write slits at the very end
                    s1_val = "" if s1_list is None or len(s1_list) <= i else str(s1_list[i])
                    s2_val = "" if s2_list is None or len(s2_list) <= i else str(s2_list[i])
                    last_col = self.tableWidget_SFM_monitors.columnCount() - 2
                    self.tableWidget_SFM_monitors.setItem(i, last_col,   QtWidgets.QTableWidgetItem(s1_val))
                    self.tableWidget_SFM_monitors.setItem(i, last_col+1, QtWidgets.QTableWidgetItem(s2_val))

                self.tableWidget_SFM_monitors.resizeColumnsToContents()
                try:
                    self.f_SFM_monitors_plot_matplotlib()
                except Exception:
                    pass

                # Drop any stale Y choices (columns may have changed with another file)
                self.SFM_monitors_Y_selection = []

        except Exception as ex:
            # non-fatal; just show message
            try:
                self.statusbar.showMessage(f"Error: could not build Monitors/Time preview ({ex})")
            except Exception:
                pass
            
    def f__maybe_refresh_monitors_tab(self, idx: int):
        """Refresh Monitors/Time preview when the Monitors tab becomes active."""
        try:
            if self.tabWidget_SFM.widget(idx) is self.tab_SFM_monitors:
                self.f_SFM_monitors_refresh()
        except Exception:
            pass


    def f_SFM_monitors_export(self):
        """Export the currently previewed Monitors/Time table as a tab-separated .dat."""
        try:
            if self.tableWidget_SFM_monitors.columnCount() == 0 or self.tableWidget_SFM_monitors.rowCount() == 0:
                self.statusbar.showMessage("Nothing to export: Monitors/Time table is empty.")
                return

            # default save dir same as other exports
            dir_saveFile = self.lineEdit_saveAt.text() if self.lineEdit_saveAt.text() else self.dir_current
            base = self.comboBox_SFM_scan.currentText()
            # strip '.h5' in a conservative way
            if base.lower().endswith(".h5"): base = base[:-3]
            out_path = os.path.join(dir_saveFile, f"{base}_monitors_time.dat")

            with open(out_path, "w", encoding="utf-8") as f:
                # header
                headers = [self.tableWidget_SFM_monitors.horizontalHeaderItem(c).text()
                        for c in range(self.tableWidget_SFM_monitors.columnCount())]
                f.write("\t".join(headers) + "\n")
                # rows
                for r in range(self.tableWidget_SFM_monitors.rowCount()):
                    row_vals = []
                    for c in range(self.tableWidget_SFM_monitors.columnCount()):
                        item = self.tableWidget_SFM_monitors.item(r, c)
                        row_vals.append("" if item is None else item.text())
                    f.write("\t".join(row_vals) + "\n")

            self.statusbar.showMessage(f"Exported monitors/time to: {out_path}")

        except Exception as ex:
            self.statusbar.showMessage(f"Error: export failed ({ex})")

    def f_SFM_monitors_contextMenu(self, pos):
        """Context menu to copy the current selection in the Monitors/Time table."""
        try:
            menu = QtWidgets.QMenu(self.tableWidget_SFM_monitors)
            act_copy = menu.addAction("Copy selection")
            action = menu.exec_(self.tableWidget_SFM_monitors.viewport().mapToGlobal(pos))
            if action == act_copy:
                self.f_SFM_monitors_copySelection()
        except Exception as ex:
            self.statusbar.showMessage(f"Copy failed: {ex}")

    def f_SFM_monitors_copySelection(self):
        """Copy selected cells as tab-separated text; supports rectangular and sparse selections."""
        try:
            sel = self.tableWidget_SFM_monitors.selectedIndexes()
            if not sel:
                return
            # Build a rectangular grid if possible; else fallback to sparse TSV
            rows = sorted(set(i.row() for i in sel))
            cols = sorted(set(i.column() for i in sel))
            # check rectangular
            rectangular = (len(sel) == len(rows) * len(cols))
            if rectangular:
                out_lines = []
                for r in rows:
                    vals = []
                    for c in cols:
                        it = self.tableWidget_SFM_monitors.item(r, c)
                        vals.append("" if it is None else it.text())
                    out_lines.append("\t".join(vals))
                text = "\n".join(out_lines)
            else:
                # sparse: sort by (row,col) and write one row per selected row
                per_row = {}
                for i in sel:
                    per_row.setdefault(i.row(), []).append(i)
                out_lines = []
                for r in sorted(per_row.keys()):
                    cols_in_row = sorted(per_row[r], key=lambda x: x.column())
                    vals = []
                    for i in cols_in_row:
                        it = self.tableWidget_SFM_monitors.item(i.row(), i.column())
                        vals.append("" if it is None else it.text())
                    out_lines.append("\t".join(vals))
                text = "\n".join(out_lines)
            QtWidgets.QApplication.clipboard().setText(text)
            self.statusbar.showMessage("Selection copied to clipboard.")
        except Exception as ex:
            self.statusbar.showMessage(f"Copy failed: {ex}")

    def f_SFM_monitors_chooseY(self):
        """
        Pop up a simple dialog with checkboxes for data columns (mon_* and time_*).
        Enforce homogeneous units: all monitor or all time.
        """
        try:
            Ncols = self.tableWidget_SFM_monitors.columnCount()
            if Ncols <= 2:
                self.statusbar.showMessage("No data columns to choose from.")
                return

            # Build list of candidate columns (>=2 are data columns)
            headers = []
            for c in range(Ncols):
                itm = self.tableWidget_SFM_monitors.horizontalHeaderItem(c)
                headers.append("" if itm is None else itm.text())

            # Allow monitors/time and slit columns as plottable Y
            data_cols = [(c, h) for c, h in enumerate(headers)
                        if (h.startswith("mon_")
                            or h.startswith("time_")
                            or h.startswith("s1hg") or h.startswith("s2hg")
                            or h.startswith("s3hg") or h.startswith("s4hg"))]
            if not data_cols:
                self.statusbar.showMessage("No data columns to choose from.")
                return

            # Small modal dialog
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Choose Y columns")
            dlg.setModal(True)
            layout = QtWidgets.QVBoxLayout(dlg)

            info = QtWidgets.QLabel("Select one or more columns (must share the same unit).")
            layout.addWidget(info)

            # A list with checkboxes
            lst = QtWidgets.QListWidget(dlg)
            lst.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            for c, h in data_cols:
                it = QtWidgets.QListWidgetItem(h)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if c in self.SFM_monitors_Y_selection else QtCore.Qt.Unchecked)
                it.setData(QtCore.Qt.UserRole, int(c))
                lst.addItem(it)
            layout.addWidget(lst)

            # OK/Cancel
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
            layout.addWidget(btns)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)

            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return

            # Gather checked columns
            chosen = []
            for i in range(lst.count()):
                it = lst.item(i)
                if it.checkState() == QtCore.Qt.Checked:
                    chosen.append(it.data(QtCore.Qt.UserRole))

            if not chosen:
                self.SFM_monitors_Y_selection = []
                self.f_SFM_monitors_plot_matplotlib()
                return

            self.SFM_monitors_Y_selection = sorted(chosen)
            self.f_SFM_monitors_plot_matplotlib()

        except Exception as ex:
            self.statusbar.showMessage(f"Y selection failed: {ex}")

    def f_SFM_monitors_selectColumn(self, col: int):
        """Click on header to select an entire column (makes copy/plotting easy)."""
        try:
            self.tableWidget_SFM_monitors.clearSelection()
            self.tableWidget_SFM_monitors.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            Nrows = self.tableWidget_SFM_monitors.rowCount()
            for r in range(Nrows):
                idx = self.tableWidget_SFM_monitors.model().index(r, col)
                self.tableWidget_SFM_monitors.selectionModel().select(
                    idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows | QtCore.QItemSelectionModel.Columns
                )
        except Exception:
            pass

    def f_SFM_monitors_selectRow(self, row: int):
        """Click on vertical header to select an entire row."""
        try:
            self.tableWidget_SFM_monitors.clearSelection()
            self.tableWidget_SFM_monitors.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            Ncols = self.tableWidget_SFM_monitors.columnCount()
            for c in range(Ncols):
                idx = self.tableWidget_SFM_monitors.model().index(row, c)
                self.tableWidget_SFM_monitors.selectionModel().select(
                    idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows | QtCore.QItemSelectionModel.Columns
                )
        except Exception:
            pass

    def f_SFM_monitors_plot(self):
        """Plot selected table columns vs selected X (Index or th(deg))."""
        try:
            plt = self.graphicsView_SFM_monitors_plot.getPlotItem()
            plt.clear()

            # Determine X
            x_mode = self.comboBox_SFM_monitors_xaxis.currentText() if self.comboBox_SFM_monitors_xaxis.count() else "Index"

            Nrows = self.tableWidget_SFM_monitors.rowCount()
            Ncols = self.tableWidget_SFM_monitors.columnCount()
            if Nrows == 0 or Ncols == 0:
                return

            if x_mode == "th(deg)":
                # X is column 1 by construction ("th(deg)")
                x = []
                for r in range(Nrows):
                    it = self.tableWidget_SFM_monitors.item(r, 1)
                    try:
                        x.append(float(it.text()) if it is not None and it.text() != "" else np.nan)
                    except Exception:
                        x.append(np.nan)
                x = np.array(x, dtype=float)
                plt.setLabel('bottom', 'th (deg)')
            else:
                x = np.arange(Nrows, dtype=float)
                plt.setLabel('bottom', 'Index')

            # Prefer explicit Y selection (from the Y… dialog); else fall back:
            if getattr(self, "SFM_monitors_Y_selection", None):
                sel_cols = list(self.SFM_monitors_Y_selection)
            else:
                # Fallback: selection in table or first data column
                sel_cols = sorted(set(i.column() for i in self.tableWidget_SFM_monitors.selectedIndexes()))
                sel_cols = [c for c in sel_cols if c >= 2]
                if not sel_cols and Ncols > 2:
                    sel_cols = [2]

            # Plot each selected column
            for c in sel_cols:
                y = []
                for r in range(Nrows):
                    it = self.tableWidget_SFM_monitors.item(r, c)
                    try:
                        y.append(float(it.text()) if it is not None and it.text() != "" else np.nan)
                    except Exception:
                        y.append(np.nan)
                y = np.array(y, dtype=float)

                name = self.tableWidget_SFM_monitors.horizontalHeaderItem(c).text() if self.tableWidget_SFM_monitors.horizontalHeaderItem(c) else f"col {c}"
                curve = pg.PlotCurveItem(x=x, y=y, name=name, connect="finite")
                plt.addItem(curve)

            try:
                plt.setLogMode(x=False, y=bool(self.checkBox_SFM_monitors_logY.isChecked())) # Log Y if requested
            except Exception:
                pass

            plt.autoRange()
        except Exception as ex:
            self.statusbar.showMessage(f"Plot failed: {ex}")

    def f_SFM_monitors_plot_matplotlib(self):
        """Interactive Monitors/Time plot using Matplotlib + toolbar (a.u. on Y)."""
        try:
            fig = self.canvas_SFM_monitors.figure
            fig.clf()
            ax = fig.add_subplot(111)

            # Determine X
            x_mode = self.comboBox_SFM_monitors_xaxis.currentText() if self.comboBox_SFM_monitors_xaxis.count() else "Index"
            Nrows = self.tableWidget_SFM_monitors.rowCount()
            Ncols = self.tableWidget_SFM_monitors.columnCount()
            if Nrows == 0 or Ncols == 0:
                self.canvas_SFM_monitors.draw()
                return

            if x_mode == "th(deg)":
                x = []
                for r in range(Nrows):
                    it = self.tableWidget_SFM_monitors.item(r, 1)
                    try:
                        x.append(float(it.text()) if it is not None and it.text() != "" else np.nan)
                    except Exception:
                        x.append(np.nan)
                x = np.array(x, dtype=float)
                ax.set_xlabel('th (deg)')
            else:
                x = np.arange(Nrows, dtype=float)
                ax.set_xlabel('Index')

            # Which Y columns?
            if getattr(self, "SFM_monitors_Y_selection", None):
                sel_cols = list(self.SFM_monitors_Y_selection)
            else:
                sel_cols = sorted(set(i.column() for i in self.tableWidget_SFM_monitors.selectedIndexes()))
                sel_cols = [c for c in sel_cols if c >= 2]
                if not sel_cols and Ncols > 2:
                    sel_cols = [2]

            # Plot all chosen Y columns (Y in a.u.)
            for c in sel_cols:
                y = []
                for r in range(Nrows):
                    it = self.tableWidget_SFM_monitors.item(r, c)
                    try:
                        y.append(float(it.text()) if it is not None and it.text() != "" else np.nan)
                    except Exception:
                        y.append(np.nan)
                y = np.array(y, dtype=float)
                label = self.tableWidget_SFM_monitors.horizontalHeaderItem(c).text() if self.tableWidget_SFM_monitors.horizontalHeaderItem(c) else f"col {c}"
                style = self.comboBox_SFM_monitors_style.currentText() if hasattr(self, "comboBox_SFM_monitors_style") else "line+scatter"
                if style == "line+scatter":
                    ax.plot(x, y, linestyle='-', marker='', label=label)
                    ax.scatter(x, y, s=12)  # no label to avoid legend duplicates
                elif style == "scatter":
                    ax.scatter(x, y, s=12, label=label)
                else:  # "line"
                    ax.plot(x, y, linestyle='-', marker='', label=label)

            # Y axis in arbitrary units
            ax.set_ylabel('a.u.')
            if bool(self.checkBox_SFM_monitors_logY.isChecked()):
                ax.set_yscale('log')

            ax.grid(True, which='both', alpha=0.35)
            if sel_cols:
                ax.legend(loc='best', fontsize=8)

            fig.tight_layout()
            self.canvas_SFM_monitors.draw()
        except Exception as ex:
            try:
                self.statusbar.showMessage(f"Plot failed: {ex}")
            except Exception:
                pass

    def f_SFM_detectorImage_draw(self):
        if self.comboBox_SFM_detectorImage_polarisation.currentText() == "" or self.comboBox_SFM_detectorImage_incidentAngle.currentText() == "":
            return

        for item in (self.graphicsView_SFM_detectorImage, self.graphicsView_SFM_detectorImage_roi): item.clear()
        if self.SFM_FILE == "": return

        with h5py.File(self.SFM_FILE, 'r') as FILE:
            SCAN = FILE[list(FILE.keys())[0]]
            R = H5Resolver(SCAN)

            self.th_current = self.comboBox_SFM_detectorImage_incidentAngle.currentText()
            pol = self.comboBox_SFM_detectorImage_polarisation.currentText()
            # pick stack for selected polarisation (PNR) or NR
            stack = R.detector_stack(pol if len(R.pol_list()) > 1 else None)
            if stack is None: return

            th_list = R.th_list() or []
            tth_list = R.tth_list() or []
            s1hg_list, s2hg_list = R.slit_lists()
            mon_list, time_list = R.monitor_and_time(pol if len(R.pol_list()) > 1 else None)
            if time_list is None:
                time_list = np.ones_like(th_list)

            # find index of selected th
            idx = None
            for i, th in enumerate(th_list):
                if self.th_current == str(round(float(th), 3)):
                    idx = i; break
            if idx is None: return

            self.lineEdit_SFM_detectorImage_slits_s1hg.setText(str(s1hg_list[idx]) if s1hg_list is not None else "")
            self.lineEdit_SFM_detectorImage_slits_s2hg.setText(str(s2hg_list[idx]) if s2hg_list is not None else "")
            self.lineEdit_SFM_detectorImage_time.setText(str(time_list[idx]))

            img = np.asarray(stack[idx], dtype=int)
            img = np.subtract(img, np.zeros_like(img))  # keep behavior identical

            # integrate in Y across ROI to get 1D profile (same as original)
            y_top = int(self.lineEdit_SFM_detectorImage_roiY_top.text())
            y_bot = int(self.lineEdit_SFM_detectorImage_roiY_bottom.text())
            x_left = int(self.lineEdit_SFM_detectorImage_roiX_left.text())
            x_right = int(self.lineEdit_SFM_detectorImage_roiX_right.text())
            prof = img[y_top:y_bot, :].sum(axis=0).astype(int)

            # display
            self.graphicsView_SFM_detectorImage.setImage(img, axes={'x':1, 'y':0}, levels=(0, 0.1))

            # colors identical to original
            if self.comboBox_SFM_detectorImage_colorScheme.currentText() == "White / Black":
                color_det_image = np.array([[0, 0, 0, 255], [255, 255, 255, 255], [255, 255, 255, 255]], dtype=np.ubyte)
            else:
                color_det_image = np.array([[0, 0, 255, 255], [255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.ubyte)
            pos = np.array([0.0, 0.1, 1.0])
            self.graphicsView_SFM_detectorImage.setColorMap(pg.ColorMap(pos, color_det_image))

            # ROI outline
            try:
                if self.roi_draw: self.graphicsView_SFM_detectorImage.removeItem(self.roi_draw)
                if self.roi_draw_bkg: self.graphicsView_SFM_detectorImage.removeItem(self.roi_draw_bkg)
            except Exception: pass

            spots = {'x': (x_left, x_right, x_right, x_left, x_left), 'y': (y_top, y_top, y_bot, y_bot, y_top)}
            self.roi_draw = pg.PlotDataItem(spots, pen=pg.mkPen(255, 255, 255), connect="all")
            self.graphicsView_SFM_detectorImage.addItem(self.roi_draw)

            # background ROI (if enabled)
            if self.checkBox_reductions_subtractBkg.isChecked():
                bxr = int(self.lineEdit_SFM_detectorImage_roi_bkgX_right.text())
                bxl = int(self.lineEdit_SFM_detectorImage_roi_bkgX_left.text())
                spots_b = {'x': (bxl, bxr, bxr, bxl, bxl), 'y': (y_top, y_top, y_bot, y_bot, y_top)}
                self.roi_draw_bkg = pg.PlotDataItem(spots_b, pen=pg.mkPen(color=(255, 255, 255), style=QtCore.Qt.DashLine), connect="all")
                self.graphicsView_SFM_detectorImage.addItem(self.roi_draw_bkg)

            # Add integrated profile + vertical ROI markers
            self.graphicsView_SFM_detectorImage_roi.addItem(pg.PlotCurveItem(y=prof, pen=pg.mkPen(color=(0, 0, 0), width=2), brush=pg.mkBrush(color=(255, 0, 0), width=3)))

            if self.roi_draw_int: self.graphicsView_SFM_detectorImage_roi.removeItem(self.roi_draw_int)
            dots = []
            for i in range(0, int(prof.max()) if prof.size else 0):
                dots.append({'x': x_left, 'y': i})
                dots.append({'x': x_right, 'y': i})
            self.roi_draw_int = pg.ScatterPlotItem(spots=dots, size=1, pen=pg.mkPen(255, 0, 0))
            self.graphicsView_SFM_detectorImage_roi.addItem(self.roi_draw_int)

        # "Integrated ROI" toggle adapted to layout-managed UI
        if self.sender().objectName() == "pushButton_SFM_detectorImage_showIntegratedRoi":
            show_profile = True if self.trigger_showDetInt else False
            self.graphicsView_SFM_detectorImage_roi.setVisible(show_profile)
            self.trigger_showDetInt = not show_profile

    def f_SFM_reflectivityPreview_load(self):
        self.graphicsView_SFM_reflectivityPreview.getPlotItem().clear()
        # --- explicit X-axis label (θᵢ or Qz) ---
        try:
            angle_mode = self.comboBox_SFM_reflectivityPreview_view_angle.currentText()
        except Exception:
            angle_mode = 'Qz'
        ax_label = 'Qz (Å⁻¹)' if angle_mode == 'Qz' else 'θᵢ (deg)'
        try:
            self.graphicsView_SFM_reflectivityPreview.getPlotItem().setLabel('bottom', ax_label)
        except Exception:
            pass

        bkg_skip = 0
        self.SFM_export_Qz, self.SFM_export_I, self.SFM_export_dI, self.SFM_export_resolution = [], [], [], []

        # UI toggles identical
        self.checkBox_export_resolutionLikeSared.setEnabled(True if self.checkBox_export_addResolutionColumn.isChecked() else False)
        if self.checkBox_reductions_normalizeByDB.isChecked():
            hidden = [False, False, True, True]; self.checkBox_reductions_scaleFactor.setChecked(False)
        else:
            hidden = [True, True, False, False]
        for idx, el in enumerate([self.checkBox_reductions_attenuatorDB, self.lineEdit_reductions_attenuatorDB, self.checkBox_reductions_scaleFactor, self.lineEdit_reductions_scaleFactor]):
            el.setHidden(hidden[idx])

        if self.comboBox_SFM_scan.currentText() == "": return
        self.statusbar.clearMessage()

        # input checks (unchanged)
        try:
            if (self.checkBox_reductions_overilluminationCorr.isChecked() or self.checkBox_SFM_reflectivityPreview_showOverillumination.isChecked()):
                self._get_shape_geometry(strict=True)
        except ValueError as ex:
            self.statusbar.showMessage(str(ex))

        if self.checkBox_reductions_normalizeByDB.isChecked() and self.tableWidget_DB.rowCount() == 0:
            self.statusbar.showMessage("Error: Direct beam file is missing.")

        if int(self.lineEdit_SFM_detectorImage_roiX_left.text()) > int(self.lineEdit_SFM_detectorImage_roiX_right.text()) or \
        int(self.lineEdit_SFM_detectorImage_roiY_bottom.text()) < int(self.lineEdit_SFM_detectorImage_roiY_top.text()) or \
        (self.checkBox_reductions_subtractBkg.checkState() and int(self.lineEdit_SFM_detectorImage_roi_bkgX_left.text()) < 0):
            self.statusbar.showMessage("Error: Recheck your ROI input.")

        self.scaleFactor = 1.0
        if self.checkBox_reductions_scaleFactor.isChecked():
            try:
                self.scaleFactor = self._parse_positive_factor(self.lineEdit_reductions_scaleFactor.text(), 10.0, 'Scale Factor')
            except ValueError as ex:
                self.statusbar.showMessage(str(ex))

        self.DB_attenFactor = 1.0
        if self.checkBox_reductions_attenuatorDB.isChecked():
            try:
                self.DB_attenFactor = self._parse_positive_factor(self.lineEdit_reductions_attenuatorDB.text(), 10.0, 'Direct Beam Attenuator Factor')
            except ValueError as ex:
                self.statusbar.showMessage(str(ex))

        if self.lineEdit_reductions_subtractBkg_Skip.text():
            try: bkg_skip = float(self.lineEdit_reductions_subtractBkg_Skip.text())
            except: self.statusbar.showMessage("Error: Recheck 'Skip background' field.")

        try:
            _ = 1/float(self.lineEdit_instrument_wavelength.text())
            _ = float(self.lineEdit_instrument_wavelengthResolution.text())
            _ = float(self.lineEdit_instrument_distanceS1ToSample.text())
            _ = float(self.lineEdit_instrument_distanceS2ToSample.text())
            _ = float(self.lineEdit_instrument_distanceSampleToDetector.text())
            _ = float(self.lineEdit_instrument_sampleCurvature.text())
            _ = float(self.lineEdit_instrument_offsetFull.text())
        except:
            self.statusbar.showMessage("Error: Recheck 'Instrument / Corrections' tab for typos.")

        if self.statusbar.currentMessage(): return

        # define file + DB id
        for i in range(0, self.tableWidget_scans.rowCount()):
            if self.tableWidget_scans.item(i, 0).text() == self.comboBox_SFM_scan.currentText():
                self.SFM_FILE = self.tableWidget_scans.item(i, 2).text()
                break
        self.SFM_DB_FILE = self.comboBox_SFM_DB.currentText()

        with h5py.File(self.SFM_FILE, 'r') as FILE:
            SCAN = FILE[list(FILE.keys())[0]]
            R = H5Resolver(SCAN)
            pols = R.pol_list()

            y_top = int(self.lineEdit_SFM_detectorImage_roiY_top.text())
            y_bot = int(self.lineEdit_SFM_detectorImage_roiY_bottom.text())
            xL = int(self.lineEdit_SFM_detectorImage_roiX_left.text())
            xR = int(self.lineEdit_SFM_detectorImage_roiX_right.text())
            bL = int(self.lineEdit_SFM_detectorImage_roi_bkgX_left.text())
            bR = int(self.lineEdit_SFM_detectorImage_roi_bkgX_right.text())

            # re-sum triggers
            if [y_top, y_bot] != getattr(self, "roi_oldCoord_Y", []):
                self.SFMFileAlreadyAnalized = ""
            self.roi_oldCoord_Y = [y_top, y_bot]

            # read monitors/time
            mon0, sec0 = R.monitor_and_time(None)
            mon_p = {}
            for p in pols:
                mon_p[p], _sec = R.monitor_and_time(p if len(pols) > 1 else None)
                if mon_p[p] is None:
                    mon_p[p] = mon0

            # --- Sample curvature correction bookkeeping ---
            roi_coord_X = [
                int(self.lineEdit_SFM_detectorImage_roiX_left.text()),
                int(self.lineEdit_SFM_detectorImage_roiX_right.text())
            ]
            roi_coord_X_bkg = [
                int(self.lineEdit_SFM_detectorImage_roi_bkgX_left.text()),
                int(self.lineEdit_SFM_detectorImage_roi_bkgX_right.text())
            ]
            apply_bkg_curvature = (
                self.checkBox_reductions_subtractBkg.isChecked()
                and roi_coord_X_bkg[0] >= 0
                and roi_coord_X_bkg[1] > roi_coord_X_bkg[0]
            )
            roi_coord_X_curvature_windows = [roi_coord_X]
            if apply_bkg_curvature:
                roi_coord_X_curvature_windows.append(roi_coord_X_bkg)

            curv_txt = self.lineEdit_instrument_sampleCurvature.text().strip()
            if not curv_txt:
                curv_txt = "0"
            curv = float(curv_txt)

            sampleCurvature_key = [
                self.lineEdit_instrument_sampleCurvature.text(),
                self.lineEdit_SFM_detectorImage_roiX_left.text(),
                self.lineEdit_SFM_detectorImage_roiX_right.text(),
                self.lineEdit_SFM_detectorImage_roiY_bottom.text(),
                self.lineEdit_SFM_detectorImage_roiY_top.text(),
                str(int(self.checkBox_reductions_subtractBkg.isChecked())),
                self.lineEdit_SFM_detectorImage_roi_bkgX_left.text(),
                self.lineEdit_SFM_detectorImage_roi_bkgX_right.text(),
            ]

            sampleCurvature_recalc = (self.sampleCurvature_last == sampleCurvature_key)

            # IMPORTANT: rebuild the Y-integrated detector arrays from the raw file
            # whenever the curvature / ROI state changes. Otherwise the curvature warp
            # is applied on top of already-warped data and cannot be cleanly undone by
            # setting the curvature back to 0.
            need_rebuild_from_raw = (
                self.SFM_FILE != self.SFMFileAlreadyAnalized or (not sampleCurvature_recalc)
            )

            if need_rebuild_from_raw:
                self.SFM_psdUU = self.SFM_psdDD = self.SFM_psdUD = self.SFM_psdDU = []
                for p in pols:
                    stack = R.detector_stack(p if len(pols) > 1 else None)
                    if stack is None:
                        setattr(self, f"SFM_psd{p.upper()}", [])
                        continue
                    arr = []
                    for i in range(stack.shape[0]):
                        arr.append(np.asarray(stack[i, y_top:y_bot, :]).sum(axis=0))
                    arr = np.vstack(arr)
                    if p == "uu":
                        self.SFM_psdUU = arr
                    elif p == "dd":
                        self.SFM_psdDD = arr
                    elif p == "ud":
                        self.SFM_psdUD = arr
                    elif p == "du":
                        self.SFM_psdDU = arr

                self.SFMFileAlreadyAnalized = self.SFM_FILE
                self.sampleCurvature_last = "0"

            if not sampleCurvature_recalc:
                for index, SFM_curv in enumerate([self.SFM_psdUU, self.SFM_psdDU, self.SFM_psdUD, self.SFM_psdDD]):
                    if curv == 0.0:
                        continue
                    if SFM_curv is None or getattr(SFM_curv, "size", 0) == 0:
                        continue

                    for roi_coord_X_window in roi_coord_X_curvature_windows:
                        # slice by current X-ROI. If background subtraction is enabled,
                        # apply the same correction independently to the background ROI too.
                        SFM_slice = SFM_curv[:, roi_coord_X_window[0]:roi_coord_X_window[1]]

                        # warp to account for curvature: build scattered grid then regrid
                        detImage_recalc = [[], [], []]  # x, y, value
                        for x, col in enumerate(np.flipud(np.rot90(SFM_slice))):
                            displacement = x * np.tan(curv)
                            for y, value in enumerate(col):
                                detImage_recalc[0].append(x)
                                detImage_recalc[1].append(y + displacement)
                                detImage_recalc[2].append(value)
                        np.rot90(SFM_slice, -1)  # keep numpy happy; original code did this

                        # zero level in new grid
                        roi_mid = (SFM_slice.shape[1]) / 2
                        zero_level = int(round(roi_mid * np.tan(curv) - min(detImage_recalc[1])))

                        grid_x, grid_y = np.mgrid[
                            0:SFM_slice.shape[1]:1,
                            min(detImage_recalc[1]):max(detImage_recalc[1]):1
                        ]
                        SFM_slice = np.flipud(np.rot90(
                            griddata((detImage_recalc[0], detImage_recalc[1]), detImage_recalc[2],
                                    (grid_x, grid_y), method="linear", fill_value=float(0))
                        ))[zero_level:zero_level + SFM_slice.shape[0], :]

                        # write the corrected slice back into the original array
                        SFM_curv[:, roi_coord_X_window[0]:roi_coord_X_window[1]] = SFM_slice

                    # assign back to the right channel
                    if index == 0: self.SFM_psdUU = SFM_curv
                    elif index == 1: self.SFM_psdDU = SFM_curv
                    elif index == 2: self.SFM_psdUD = SFM_curv
                    elif index == 3: self.SFM_psdDD = SFM_curv

                # remember the inputs that produced the current correction
                self.sampleCurvature_last = sampleCurvature_key
            # --- end curvature correction ---

            th_list = R.th_list() or []
            tth_list = R.tth_list() or []
            self.th_list, self.tth_list = th_list, tth_list
            s1hg_list, s2hg_list = R.slit_lists()
            self.s1hg_list, self.s2hg_list = s1hg_list, s2hg_list

            # sample curvature correction block left unchanged (operates on self.SFM_psd*)
            # ... keep your existing curvature code here without changes ...

            # now compute/refplot per channel (same math as original)
            for colorIndex, (p, data) in enumerate([
                ("uu", self.SFM_psdUU), ("du", self.SFM_psdDU), ("ud", self.SFM_psdUD), ("dd", self.SFM_psdDD)
            ]):
                if (data is None) or (hasattr(data, "size") and data.size == 0) or (hasattr(data, "__len__") and len(data) == 0):
                    continue

                if colorIndex == 0: color, monitorData = [0, 0, 0], (mon_p.get("uu") if mon_p.get("uu") is not None else mon0)
                elif colorIndex == 1: color, monitorData = [0, 0, 255], first_non_none(mon_p.get("du"), mon0)
                elif colorIndex == 2: color, monitorData = [0, 255, 0], first_non_none(mon_p.get("ud"), mon0)
                elif colorIndex == 3: color, monitorData = [255, 0, 0], first_non_none(mon_p.get("dd"), mon0)

                plot_I, plot_angle, plot_dI_err_bottom, plot_dI_err_top, plot_overillumination = [], [], [], [], []
                SFM_export_Qz_onePol, SFM_export_I_onePol, SFM_export_dI_onePol, SFM_export_resolution_onePol = [], [], [], []

                for i, th in enumerate(th_list):
                    th = th - float(self.lineEdit_instrument_offsetFull.text())
                    Qz = (4 * np.pi / float(self.lineEdit_instrument_wavelength.text())) * np.sin(np.radians(th))
                    s1hg, s2hg = (s1hg_list[i] if s1hg_list is not None else 0), (s2hg_list[i] if s2hg_list is not None else 0)
                    monitor = monitorData[i] if monitorData is not None else 1

                    # overillumination coeff + projected FWHM
                    overill_calc, FWHM_proj = self.f_overilluminationCorrCoeff(s1hg, s2hg, round(th, 4))
                    if not self.checkBox_reductions_overilluminationCorr.isChecked():
                        overillCorr = 1
                        overill_plot = overill_calc
                    else:
                        overillCorr = overill_calc
                        overill_plot = overillCorr

                    # Resolution (same branches as original)
                    if self.checkBox_export_resolutionLikeSared.isChecked():
                        Resolution = np.sqrt(((2 * np.pi / float(self.lineEdit_instrument_wavelength.text())) ** 2) *
                                            ((np.cos(np.radians(th))) ** 2) * (0.68 ** 2) *
                                            ((s1hg ** 2) + (s2hg ** 2)) /
                                            ((float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())) ** 2) +
                                            ((float(self.lineEdit_instrument_wavelengthResolution.text()) ** 2) * (Qz ** 2)))
                    else:
                        d_alpha = np.arctan((s1hg + [s2hg if FWHM_proj == s2hg else FWHM_proj][0]) /
                                            ((float(self.lineEdit_instrument_distanceS1ToSample.text()) - float(self.lineEdit_instrument_distanceS2ToSample.text())) * 2))
                        if self.comboBox_export_angle.currentText() == "Qz":
                            k_0 = 2 * np.pi / float(self.lineEdit_instrument_wavelength.text())
                            Resolution = np.sqrt(
                                (k_0 ** 2) * (
                                    (((np.cos(np.radians(th))) ** 2) * d_alpha**2) + 
                                                               (((float(self.lineEdit_instrument_wavelengthResolution.text()))**2) * ((np.sin(np.radians(th)))**2))
                                                               ))
                        else:
                            Resolution = d_alpha if self.comboBox_export_angle.currentText() == "Radians" else np.degrees(d_alpha)

                    Resolution = Resolution / (2 * np.sqrt(2 * np.log(2)))  # FWHM → sigma

                    # integrate X within ROI (+ optional BKG)
                    Intens = float(np.sum(data[i, xL:xR]))
                    Intens_bkg = float(np.sum(data[i, bL:bR]))

                    Intens = max(0.0, Intens)
                    IntensErr = 1.0 if Intens == 0 else np.sqrt(Intens)

                    if self.checkBox_reductions_subtractBkg.isChecked() and Qz > bkg_skip and Intens_bkg > 0:
                        IntensErr = np.sqrt(Intens + Intens_bkg)
                        Intens -= Intens_bkg

                    if self.checkBox_reductions_divideByMonitorOrTime.isChecked():
                        if self.comboBox_reductions_divideByMonitorOrTime.currentText() == "monitor":
                            if Intens == 0: IntensErr = IntensErr / monitor
                            else: IntensErr = (Intens / monitor) * np.sqrt((IntensErr / Intens) ** 2 + (1 / monitor))
                            Intens = Intens / monitor
                        elif self.comboBox_reductions_divideByMonitorOrTime.currentText() == "time":
                            _, time_list = R.monitor_and_time(None)
                            t = time_list[i] if time_list is not None else 1
                            IntensErr = IntensErr / t
                            Intens = Intens / t

                    if self.checkBox_reductions_overilluminationCorr.isChecked() and overillCorr > 0:
                        IntensErr /= overillCorr; Intens /= overillCorr

                    if self.checkBox_reductions_normalizeByDB.isChecked():
                        try:
                            DB_intens = float(self.DB_INFO[self.SFM_DB_FILE + ";" + str(s1hg) + ";" + str(s2hg)].split(";")[0]) * self.DB_attenFactor
                            DB_err = float(self.DB_INFO[self.SFM_DB_FILE + ";" + str(s1hg) + ";" + str(s2hg)].split(";")[1]) * self.DB_attenFactor
                            IntensErr = (Intens / DB_intens) * np.sqrt((DB_err / DB_intens) ** 2 + (IntensErr / Intens) ** 2)
                            Intens = Intens / DB_intens
                            self.statusbar.clearMessage()
                        except:
                            self.statusbar.showMessage("Error: Choose another DB file for this SFM data file.")
                            self.checkBox_reductions_normalizeByDB.setCheckState(0)

                    if self.checkBox_reductions_scaleFactor.isChecked():
                        IntensErr /= self.scaleFactor
                        Intens /= self.scaleFactor

                    try:
                        show_first = int(self.lineEdit_SFM_reflectivityPreview_skipPoints_left.text())
                        show_last = len(th_list) - int(self.lineEdit_SFM_reflectivityPreview_skipPoints_right.text())
                    except:
                        show_first, show_last = 0, len(th_list)

                    if not (np.isfinite(Intens) and np.isfinite(IntensErr) and np.isfinite(Qz) and np.isfinite(Resolution)):
                        continue

                    if Intens >= 0 and i < show_last and i > show_first:
                        SFM_export_Qz_onePol.append(Qz)
                        SFM_export_I_onePol.append(Intens)
                        SFM_export_dI_onePol.append(IntensErr)
                        SFM_export_resolution_onePol.append(Resolution)

                        if Intens > 0:
                            if self.comboBox_SFM_reflectivityPreview_view_reflectivity.currentText() == "Lin":
                                plot_I.append(Intens); plot_dI_err_top.append(IntensErr); plot_dI_err_bottom.append(IntensErr)
                            else:
                                plot_I.append(np.log10(Intens))
                                plot_dI_err_top.append(abs(np.log10(Intens + IntensErr) - np.log10(Intens)))
                                plot_dI_err_bottom.append(0 if Intens <= IntensErr else (np.log10(Intens) - np.log10(Intens - IntensErr)))
                            plot_angle.append(Qz if self.comboBox_SFM_reflectivityPreview_view_angle.currentText() == "Qz" else th)
                            plot_overillumination.append(overill_plot)

                # export buffers per pol
                self.SFM_export_Qz.append(SFM_export_Qz_onePol)
                self.SFM_export_I.append(SFM_export_I_onePol)
                self.SFM_export_dI.append(SFM_export_dI_onePol)
                self.SFM_export_resolution.append(SFM_export_resolution_onePol)

                # plot (sanitized to avoid passing nan/inf values into pyqtgraph)
                plot_angle_arr = np.asarray(plot_angle, dtype=float)
                plot_I_arr = np.asarray(plot_I, dtype=float)
                plot_dI_err_top_arr = np.asarray(plot_dI_err_top, dtype=float)
                plot_dI_err_bottom_arr = np.asarray(plot_dI_err_bottom, dtype=float)
                plot_overillumination_arr = np.asarray(plot_overillumination, dtype=float)

                point_mask = np.isfinite(plot_angle_arr) & np.isfinite(plot_I_arr)
                err_mask = point_mask & np.isfinite(plot_dI_err_top_arr) & np.isfinite(plot_dI_err_bottom_arr)
                overillumination_mask = np.isfinite(plot_angle_arr) & np.isfinite(plot_overillumination_arr)

                if self.checkBox_SFM_reflectivityPreview_includeErrorbars.isChecked() and np.any(err_mask):
                    s1 = pg.ErrorBarItem(
                        x=plot_angle_arr[err_mask],
                        y=plot_I_arr[err_mask],
                        top=plot_dI_err_top_arr[err_mask],
                        bottom=plot_dI_err_bottom_arr[err_mask],
                        pen=pg.mkPen(color[0], color[1], color[2]),
                        brush=pg.mkBrush(color[0], color[1], color[2]),
                    )
                    self.graphicsView_SFM_reflectivityPreview.addItem(s1)
                if np.any(point_mask):
                    s2 = pg.ScatterPlotItem(
                        x=plot_angle_arr[point_mask],
                        y=plot_I_arr[point_mask],
                        symbol="o",
                        size=4,
                        pen=pg.mkPen(color[0], color[1], color[2]),
                        brush=pg.mkBrush(color[0], color[1], color[2]),
                    )
                    self.graphicsView_SFM_reflectivityPreview.addItem(s2)

                if self.checkBox_SFM_reflectivityPreview_showOverillumination.isChecked() and np.any(overillumination_mask):
                    s3 = pg.PlotCurveItem(
                        x=plot_angle_arr[overillumination_mask],
                        y=plot_overillumination_arr[overillumination_mask],
                        pen=pg.mkPen(color=(255, 0, 0), width=1),
                    )
                    self.graphicsView_SFM_reflectivityPreview.addItem(s3)

                if self.checkBox_SFM_reflectivityPreview_showZeroLevel.isChecked():
                    level = np.array([1, 1]) if self.comboBox_SFM_reflectivityPreview_view_reflectivity.currentText() == "Lin" else np.array([0, 0])
                    s4 = pg.PlotCurveItem(x=_safe_xrange_from(plot_angle_arr[point_mask]), y=level, pen=pg.mkPen(color=(0, 0, 255), width=1))
                    self.graphicsView_SFM_reflectivityPreview.addItem(s4)

        # --- ensure axis labels and margins are visible (Reflectivity preview) ---
        try:
            pi = self.graphicsView_SFM_reflectivityPreview.getPlotItem()
            # Choose X label based on "vs Angle" selector
            angle_mode = self.comboBox_SFM_reflectivityPreview_view_angle.currentText() if self.comboBox_SFM_reflectivityPreview_view_angle.count() else "Qz"
            x_lbl = "Qz (Å⁻¹)" if angle_mode == "Qz" else "θ (deg)"
            pi.setLabel('bottom', x_lbl, **{'font-size': '11pt'})
            # Y label: reflectivity in a.u. for preview
            pi.setLabel('left', 'Reflectivity (a.u.)', **{'font-size': '11pt'})
            # Add some breathing room for labels
            pi.layout.setContentsMargins(8, 8, 8, 24)
        except Exception:
            pass
        # --- end labels/margins ---

    def f_SFM_2Dmap_draw(self):

        # --- explicit axis labels for 2D map ---
        try:
            mode = self.comboBox_SFM_2Dmap_axes.currentText()
        except Exception:
            mode = 'Pixel vs. Point'
        try:
            item = self.graphicsView_SFM_2Dmap_Qxz_theta.getPlotItem()
        except Exception:
            item = None
        if item is not None:
            if mode == 'Qx vs. Qz':
                item.setLabel('bottom', 'Qx (Å⁻¹)')
                item.setLabel('left', 'Qz (Å⁻¹)')
            elif mode == 'Alpha_i vs. Alpha_f':
                item.setLabel('bottom', 'θᵢ (deg)')
                item.setLabel('left', 'θ_f (deg)')
            else:
                item.setLabel('bottom', '')
                item.setLabel('left', '')

        self.SFM_intDetectorImage = []

        for item in (self.graphicsView_SFM_2Dmap_Qxz_theta, self.graphicsView_SFM_2Dmap): item.clear()

        # change interface for the selected 2D-map mode
        ELEMENTS = [self.label_SFM_2Dmap_rescaleImage_x, self.label_SFM_2Dmap_rescaleImage_y, self.horizontalSlider_SFM_2Dmap_rescaleImage_x, self.horizontalSlider_SFM_2Dmap_rescaleImage_y, self.label_SFM_2Dmap_lowerNumberOfPointsBy, self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy, self.label_SFM_2Dmap_QxzThreshold, self.comboBox_SFM_2Dmap_QxzThreshold, self.label_SFM_2Dmap_view_scale, self.comboBox_SFM_2Dmap_view_scale, self.checkBox_SFM_2Dmap_flip]

        if self.comboBox_SFM_2Dmap_axes.currentText() == "Pixel vs. Point":
            visible = [True, True, True, True, False, False, False, False, True, True, False]
            current_display = self.graphicsView_SFM_2Dmap
        elif self.comboBox_SFM_2Dmap_axes.currentText() == "Qx vs. Qz":
            visible = [False, False, False, False, True, True, True, True, False, False, False]
            current_display = self.graphicsView_SFM_2Dmap_Qxz_theta
        else:
            visible = [False, False, False, False, True, True, False, False, True, True, True]
            current_display = self.graphicsView_SFM_2Dmap

        for index, index_visible in enumerate(visible):
            ELEMENTS[index].setVisible(index_visible)
        try:
            self._2d_display_stack.setCurrentWidget(current_display)
        except Exception:
            pass

        if self.SFM_FILE == "": return

        # start over if we selected nes SFM scan
        if not self.SFMFile2dCalculatedParams == [] and not self.SFMFile2dCalculatedParams[0] == self.SFM_FILE:
            self.comboBox_SFM_2Dmap_axes.setCurrentIndex(0)
            self.SFMFile2dCalculatedParams, self.res_aif = [], []

        try:
            self.graphicsView_SFM_2Dmap.removeItem(self.roi_draw_2Dmap)
        except: True

        # load selected integrated detector image
        if self.comboBox_SFM_2Dmap_polarisation.count() == 1: self.SFM_intDetectorImage = self.SFM_psdUU
        else:
            if self.comboBox_SFM_2Dmap_polarisation.currentText() == "uu": self.SFM_intDetectorImage = self.SFM_psdUU
            elif self.comboBox_SFM_2Dmap_polarisation.currentText() == "du": self.SFM_intDetectorImage = self.SFM_psdDU
            elif self.comboBox_SFM_2Dmap_polarisation.currentText() == "ud": self.SFM_intDetectorImage = self.SFM_psdUD
            elif self.comboBox_SFM_2Dmap_polarisation.currentText() == "dd": self.SFM_intDetectorImage = self.SFM_psdDD

        if (self.SFM_intDetectorImage is None) or (hasattr(self.SFM_intDetectorImage, "size") and self.SFM_intDetectorImage.size == 0) or (hasattr(self.SFM_intDetectorImage, "__len__") and len(self.SFM_intDetectorImage) == 0):
            return

        # create log array for log view
        self.SFM_intDetectorImage_log = np.log10(np.where(self.SFM_intDetectorImage < 1, 0.1, self.SFM_intDetectorImage))

        # Pixel to Angle conversion for "Qx vs Qz" and "alpha_i vs alpha_f" 2d maps
        if self.comboBox_SFM_2Dmap_axes.currentText() in ["Qx vs. Qz", "Alpha_i vs. Alpha_f"]:
            # recalculate only if something was changed
            if _is_empty(self.res_aif) or not self.SFMFile2dCalculatedParams == [self.SFM_FILE, self.comboBox_SFM_2Dmap_polarisation.currentText(), self.lineEdit_SFM_detectorImage_roiX_left.text(), self.lineEdit_SFM_detectorImage_roiX_right.text(), self.lineEdit_instrument_wavelength.text(), self.lineEdit_instrument_distanceSampleToDetector.text(), self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy.currentText(), self.comboBox_SFM_2Dmap_QxzThreshold.currentText(), self.lineEdit_instrument_sampleCurvature.text(), self.checkBox_SFM_2Dmap_flip.checkState()]:
                self.spots_Qxz, self.SFM_intDetectorImage_Qxz, self.SFM_intDetectorImage_aif, self.SFM_intDetectorImage_values_array = [], [], [[],[]], []

                # flip image in Aif mode (checkbox) -- this requires another ROI
                if self.comboBox_SFM_2Dmap_axes.currentText() == "Alpha_i vs. Alpha_f":
                    # we need to flip the detector (X) for correct calculation
                    self.SFM_intDetectorImage = np.flip(self.SFM_intDetectorImage, 1)
                    roi_middle = round((self.SFM_intDetectorImage.shape[1] - float(self.lineEdit_SFM_detectorImage_roiX_left.text()) +
                                            self.SFM_intDetectorImage.shape[1] - float(self.lineEdit_SFM_detectorImage_roiX_right.text())) / 2)


                # --- ensure ROI center (in pixels) is available for both Aif and Qxz branches ---

                try:

                    roi_L = float(self.lineEdit_SFM_detectorImage_roiX_left.text())

                    roi_R = float(self.lineEdit_SFM_detectorImage_roiX_right.text())

                except Exception:

                    roi_L, roi_R = 0.0, float(self.SFM_intDetectorImage.shape[1] - 1)

                roi_middle = round((self.SFM_intDetectorImage.shape[1] - roi_L + self.SFM_intDetectorImage.shape[1] - roi_R) / 2)


                mm_per_pix = 300 / self.SFM_intDetectorImage.shape[1]

                for theta_i, tth_i, det_image_i in zip(self.th_list, self.tth_list, self.SFM_intDetectorImage):
                    for pixel_num, value in enumerate(det_image_i):
                        # Reduce number of points to draw (to save RAM)
                        if pixel_num % int(self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy.currentText()) == 0:

                            theta_f = tth_i - theta_i # theta F in deg
                            delta_theta_F_mm = (pixel_num - roi_middle) * mm_per_pix
                            delta_theta_F_deg = np.degrees(np.arctan(delta_theta_F_mm / float(self.lineEdit_instrument_distanceSampleToDetector.text()))) # calculate delta theta F in deg
                            theta_f = theta_f + delta_theta_F_deg * (-1 if self.checkBox_SFM_2Dmap_flip.isChecked() else 1) # final theta F in deg for the point

                            # convert to Q
                            Qx = (2 * np.pi / float(self.lineEdit_instrument_wavelength.text())) * (np.cos(np.radians(theta_f)) - np.cos(np.radians(theta_i)))
                            Qz = (2 * np.pi / float(self.lineEdit_instrument_wavelength.text())) * (np.sin(np.radians(theta_f)) + np.sin(np.radians(theta_i)))

                            for arr, val in zip((self.SFM_intDetectorImage_Qxz, self.SFM_intDetectorImage_aif[0], self.SFM_intDetectorImage_aif[1], self.SFM_intDetectorImage_values_array), ([Qx, Qz, value], theta_i, theta_f, value)): arr.append(val)

                            # define colors - 2 count+ -> green, [0,1] - blue
                            color = [0, 0, 255] if value < int(self.comboBox_SFM_2Dmap_QxzThreshold.currentText()) else [0, 255, 0]

                            self.spots_Qxz.append({'pos': (-Qx, Qz), 'pen': pg.mkPen(color[0], color[1], color[2])})

                if self.comboBox_SFM_2Dmap_axes.currentText() == "Alpha_i vs. Alpha_f":
                    # calculate required number of pixels in Y axis
                    self.resolution_x_pix_deg = self.SFM_intDetectorImage.shape[0] / (max(self.SFM_intDetectorImage_aif[0]) - min(self.SFM_intDetectorImage_aif[0]))
                    self.resolution_y_pix = int(round((max(self.SFM_intDetectorImage_aif[1]) - min(self.SFM_intDetectorImage_aif[1])) * self.resolution_x_pix_deg))

                    grid_x, grid_y = np.mgrid[min(self.SFM_intDetectorImage_aif[0]):max(self.SFM_intDetectorImage_aif[0]):((max(self.SFM_intDetectorImage_aif[0]) - min(self.SFM_intDetectorImage_aif[0]))/len(self.th_list)), min(self.SFM_intDetectorImage_aif[1]):max(self.SFM_intDetectorImage_aif[1]):(max(self.SFM_intDetectorImage_aif[1]) - min(self.SFM_intDetectorImage_aif[1]))/self.resolution_y_pix]

                    self.res_aif = griddata((np.array(self.SFM_intDetectorImage_aif[0]), np.array(self.SFM_intDetectorImage_aif[1])), np.array(self.SFM_intDetectorImage_values_array), (grid_x, grid_y), method="linear", fill_value=float(0))

                    # create log array for log view
                    self.res_aif_log = np.log10(np.where(self.res_aif < 1, 0.1, self.res_aif))

                # record params that we used for 2D maps calculation
                self.SFMFile2dCalculatedParams = [self.SFM_FILE, self.comboBox_SFM_2Dmap_polarisation.currentText(), self.lineEdit_SFM_detectorImage_roiX_left.text(), self.lineEdit_SFM_detectorImage_roiX_right.text(), self.lineEdit_instrument_wavelength.text(), self.lineEdit_instrument_distanceSampleToDetector.text(), self.comboBox_SFM_2Dmap_lowerNumberOfPointsBy.currentText(), self.comboBox_SFM_2Dmap_QxzThreshold.currentText(), self.lineEdit_instrument_sampleCurvature.text(),self.checkBox_SFM_2Dmap_flip.checkState()]

        # plot
        if self.comboBox_SFM_2Dmap_axes.currentText() == "Pixel vs. Point":

            image = self.SFM_intDetectorImage_log if self.comboBox_SFM_2Dmap_view_scale.currentText() == "Log" else self.SFM_intDetectorImage

            self.graphicsView_SFM_2Dmap.setImage(image, axes={'x': 1, 'y': 0}, levels=(0, np.max(image)), scale=(int(self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value()), int(self.horizontalSlider_SFM_2Dmap_rescaleImage_y.value())))
            # add ROI rectangular
            spots_ROI = {'x':(int(self.lineEdit_SFM_detectorImage_roiX_left.text()) * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value()), int(self.lineEdit_SFM_detectorImage_roiX_right.text()) * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value()), int(self.lineEdit_SFM_detectorImage_roiX_right.text()) * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value()), int(self.lineEdit_SFM_detectorImage_roiX_left.text()) * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value()), int(self.lineEdit_SFM_detectorImage_roiX_left.text()) * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value())), 'y':(0,0,self.SFM_intDetectorImage.shape[0] * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_y.value()),self.SFM_intDetectorImage.shape[0] * int(self.horizontalSlider_SFM_2Dmap_rescaleImage_y.value()),0)}

            self.roi_draw_2Dmap = pg.PlotDataItem(spots_ROI, pen=pg.mkPen(255, 255, 255), connect="all")
            self.graphicsView_SFM_2Dmap.addItem(self.roi_draw_2Dmap)

        elif self.comboBox_SFM_2Dmap_axes.currentText() == "Alpha_i vs. Alpha_f":

            image = self.res_aif_log if self.comboBox_SFM_2Dmap_view_scale.currentText() == "Log" else self.res_aif

            self.graphicsView_SFM_2Dmap.setImage(image, axes={'x': 0, 'y': 1}, levels=(0, np.max(image)))
            self.graphicsView_SFM_2Dmap.getImageItem().setRect(QtCore.QRectF(min(self.SFM_intDetectorImage_aif[0]), min(self.SFM_intDetectorImage_aif[1]), max(self.SFM_intDetectorImage_aif[0]) - min(self.SFM_intDetectorImage_aif[0]), max(self.SFM_intDetectorImage_aif[1]) - min(self.SFM_intDetectorImage_aif[1])))
            self.graphicsView_SFM_2Dmap.getView().enableAutoRange(True, True)

            # add line at 45 degrees at alfa_f==0
            spots_45 = {'x': (min(self.SFM_intDetectorImage_aif[0]), max(self.SFM_intDetectorImage_aif[0]) - min(self.SFM_intDetectorImage_aif[0])), 'y': (min(self.SFM_intDetectorImage_aif[0]), max(self.SFM_intDetectorImage_aif[0]) - min(self.SFM_intDetectorImage_aif[0])) }
            self.graphicsView_SFM_2Dmap.addItem(pg.PlotDataItem(spots_45, pen=pg.mkPen(255, 255, 255), connect = "all"))

        elif self.comboBox_SFM_2Dmap_axes.currentText() == "Qx vs. Qz":
            s0 = pg.ScatterPlotItem(spots=self.spots_Qxz, size=1)
            self.graphicsView_SFM_2Dmap_Qxz_theta.addItem(s0)

        # hide Y axis in 2D map if "rescale image" is used. Reason - misleading scale
        for item in (self.graphicsView_SFM_2Dmap.view.getAxis("left"), self.graphicsView_SFM_2Dmap.view.getAxis("bottom")): item.setTicks(None)
        if self.horizontalSlider_SFM_2Dmap_rescaleImage_x.value() > 1: self.graphicsView_SFM_2Dmap.view.getAxis("bottom").setTicks([])
        if self.horizontalSlider_SFM_2Dmap_rescaleImage_y.value() > 1: self.graphicsView_SFM_2Dmap.view.getAxis("left").setTicks([])

    def f_SFM_2Dmap_export(self):
        dir_saveFile = self.lineEdit_saveAt.text() if self.lineEdit_saveAt.text() else self.dir_current

        if self.comboBox_SFM_2Dmap_axes.currentText() == "Pixel vs. Point":
            with open(dir_saveFile + self.SFM_FILE[self.SFM_FILE.rfind("/") + 1 : -3] + "_" + self.comboBox_SFM_2Dmap_polarisation.currentText() + " 2Dmap(Pixel vs. Point).dat", "w") as newFile_2Dmap:
                for line in self.SFM_intDetectorImage:
                    for row in line: newFile_2Dmap.write(str(row) + " ")
                    newFile_2Dmap.write("\n")

        elif self.comboBox_SFM_2Dmap_axes.currentText() == "Alpha_i vs. Alpha_f":
            # Matrix
            with open(dir_saveFile + self.SFM_FILE[self.SFM_FILE.rfind("/") + 1 : -3] + "_" + self.comboBox_SFM_2Dmap_polarisation.currentText() + " 2Dmap_(Alpha_i vs. Alpha_f)).dat", "w") as newFile_2Dmap_aif:
                # header
                newFile_2Dmap_aif.write("Alpha_i limits: " + str(min(self.SFM_intDetectorImage_aif[0])) + " : " + str(max(self.SFM_intDetectorImage_aif[0])) +
                                        "   Alpha_f limits: " + str(min(self.SFM_intDetectorImage_aif[1])) + " : " + str(max(self.SFM_intDetectorImage_aif[1])) + " degrees\n")
                for line in np.rot90(self.res_aif):
                    for row in line: newFile_2Dmap_aif.write(str(row) + " ")
                    newFile_2Dmap_aif.write("\n")

            # Points (full)
            with open(dir_saveFile + self.SFM_FILE[self.SFM_FILE.rfind("/") + 1: -3] + "_" + self.comboBox_SFM_2Dmap_polarisation.currentText() + " 2Dmap_(Alpha_i vs. Alpha_f))_Points.dat", "w") as newFile_2Dmap_aifPoints:

                self.SFM_intDetectorImage_values_array, self.SFM_intDetectorImage_aif = [], [[], []]
                roi_middle = round((self.SFM_intDetectorImage.shape[1] - float(self.lineEdit_SFM_detectorImage_roiX_left.text()) +
                                    self.SFM_intDetectorImage.shape[1] - float(self.lineEdit_SFM_detectorImage_roiX_right.text())) / 2)

                mm_per_pix = 300 / self.SFM_intDetectorImage.shape[1]

                for theta_i, tth_i, det_image_i in zip(self.th_list, self.tth_list, self.SFM_intDetectorImage):
                    for pixel_num, value in enumerate(det_image_i):
                        theta_f = tth_i - theta_i # theta F in deg
                        delta_theta_F_mm = (pixel_num - roi_middle) * mm_per_pix
                        delta_theta_F_deg = np.degrees(np.arctan(delta_theta_F_mm / float(self.lineEdit_instrument_distanceSampleToDetector.text()))) # calculate delta theta F in deg
                        theta_f = theta_f + delta_theta_F_deg * (-1 if self.checkBox_SFM_2Dmap_flip.isChecked() else 1) # final theta F in deg for the point

                        for arr, val in zip((self.SFM_intDetectorImage_aif[0], self.SFM_intDetectorImage_aif[1], self.SFM_intDetectorImage_values_array), (theta_i, theta_f, value)): arr.append(val)

                for index in range(len(self.SFM_intDetectorImage_values_array)-1):
                    newFile_2Dmap_aifPoints.write(f"{str(self.SFM_intDetectorImage_aif[0][index])} {str(self.SFM_intDetectorImage_aif[1][index])} {str(self.SFM_intDetectorImage_values_array[index])} \n")

        elif self.comboBox_SFM_2Dmap_axes.currentText() in ["Qx vs. Qz"]:
            with open(dir_saveFile + self.SFM_FILE[self.SFM_FILE.rfind("/") + 1 : -3] + "_" + self.comboBox_SFM_2Dmap_polarisation.currentText() + " points_(Qx, Qz, intens).dat", "w") as newFile_2Dmap_Qxz:
                for line in self.SFM_intDetectorImage_Qxz: newFile_2Dmap_Qxz.write(str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n")

    def f_SFM_roi_update(self):

        roi_width = int(self.lineEdit_SFM_detectorImage_roiX_right.text()) - int(self.lineEdit_SFM_detectorImage_roiX_left.text())

        if not self.sender().objectName() == "lineEdit_SFM_detectorImage_roi_bkgX_right":
            self.lineEdit_SFM_detectorImage_roi_bkgX_left.setText(str(int(self.lineEdit_SFM_detectorImage_roiX_left.text()) - 2 * roi_width))
            self.lineEdit_SFM_detectorImage_roi_bkgX_right.setText(str(int(self.lineEdit_SFM_detectorImage_roiX_left.text()) - roi_width))
        else: self.lineEdit_SFM_detectorImage_roi_bkgX_left.setText(str(int(self.lineEdit_SFM_detectorImage_roi_bkgX_right.text()) - roi_width))

        # record ROI coord for "Lock ROI" checkbox
        self.roiLocked = [[self.lineEdit_SFM_detectorImage_roiY_top.text() + ". ", self.lineEdit_SFM_detectorImage_roiY_bottom.text() + ". ", self.lineEdit_SFM_detectorImage_roiX_left.text() + ". ", self.lineEdit_SFM_detectorImage_roiX_right.text() + ". "], self.lineEdit_SFM_detectorImage_roi_bkgX_right.text()]

        self.f_SFM_detectorImage_draw()
        self.f_SFM_reflectivityPreview_load()
        self.f_SFM_2Dmap_draw()

    ##<--

if __name__ == "__main__":
    QtWidgets.QApplication.setStyle("Fusion")
    app = QtWidgets.QApplication(sys.argv)
    prog = GUI()
    prog.show()
    sys.exit(app.exec_())
