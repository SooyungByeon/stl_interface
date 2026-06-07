"""
Click2STL authoring interface (PyQt6) — working interface for Example 1.

Mode-based gestures (paper Table 1). Pick a tool on the left, then act on the map:
  - Region : drag to draw a rectangle referent.
  - Reach  : click a region  -> ◇ reach requirement on it.
  - Keep   : click a region  -> □ keep requirement on it.
  - Or     : click two existing requirements (by their regions) -> merge them into
             one disjunction ◇/□(R1 ∨ R2 ∨ …); originals are removed (keep clicking
             to fold in more). Same operator + same window required.
  - Window : click a requirement's region -> type the [t1,t2] interval.
  - Plan   : translate gestures -> STL, solve MICP, draw trajectory; if infeasible,
             extract the IIS and highlight the responsible gestures (Section 4).

Start position and obstacles are environment-supplied, sampled with the same
planner routine used by run_example1, and drawn on the map. Fixed runways G1
(LAX) / G2 (Ontario) are pre-placed. Sequence / Until are stubbed.

Run:  ./run_interface.sh   (or python -m interface.app)
"""

from __future__ import annotations

import sys

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QStatusBar, QInputDialog,
)

from planner import la_frame as la
from planner.terrain_bg import load_la_dem
from planner.examples import GeometryConstraints
from planner.descriptor import _place_random_obstacles
from .gestures import GestureProgram
from .diagnosis import plan as plan_program

MODES = ["Rect", "Circle", "Reach", "Keep", "Sequence", "Until", "Or", "Window"]
MODE_HINT = {
    "Rect":   "drag to draw a rectangular region",
    "Circle": "drag from the centre outward to set the radius",
    "Reach":  "click a region  →  ◇ reach",
    "Keep":   "click a region  →  □ keep",
    "Or":     "click two requirements (their regions) to combine into ◇/□(· ∨ ·)",
    "Window": "click a requirement's region  →  type [t1, t2]",
}


class MapCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(8, 4.7))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.11)
        self._bg = load_la_dem()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Click2STL — Authoring Interface")
        self.mode = "Rect"

        # ---- mission state ----
        self.program = GestureProgram(obstacles=[])
        self.program.add_region(la.box(*la.G1_KM, 2.0), name="G1", is_runway=True)
        self.program.add_region(la.box(*la.G2_KM, 2.0), name="G2", is_runway=True)
        self.x0 = self._make_x0()
        self.bounds = la.make_bounds()
        self.program.obstacles = self._sample_obstacles()
        self.last_req = None
        self.or_anchor = None           # requirement being OR-combined (Or tool)
        self.result = None
        self.iis_regions: set[str] = set()
        self._press = None
        self._rubber = None

        # ---- widgets ----
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        palette = QVBoxLayout()
        palette.addWidget(QLabel("<b>Gestures</b>"))
        self.btns = {}
        for g in MODES:
            b = QPushButton(g); b.setCheckable(True)
            b.clicked.connect(lambda _c, gg=g: self.set_mode(gg))
            palette.addWidget(b); self.btns[g] = b
        for g in ("Sequence", "Until"):
            self.btns[g].setEnabled(False)
            self.btns[g].setToolTip("drag-arrow gesture — not yet wired")
        self.btns["Rect"].setChecked(True)
        palette.addStretch(1)
        self.clearstl_btn = QPushButton("Clear STL"); self.clearstl_btn.clicked.connect(self.action_clear_stl)
        palette.addWidget(self.clearstl_btn)
        self.clear_btn = QPushButton("Clear (all)"); self.clear_btn.clicked.connect(self.action_clear)
        palette.addWidget(self.clear_btn)
        self.plan_btn = QPushButton("Plan (solve)"); self.plan_btn.clicked.connect(lambda: self.action_plan())
        palette.addWidget(self.plan_btn)
        root.addLayout(palette, 0)

        self.canvas = MapCanvas()
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        root.addWidget(self.canvas, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>STL specification</b>"))
        self.formula = QTextEdit(); self.formula.setReadOnly(True)
        self.formula.setPlaceholderText("(authored STL appears here)")
        self.formula.setMinimumWidth(300)
        right.addWidget(self.formula, 3)
        right.addWidget(QLabel("<b>Infeasibility (IIS) resolution</b>"))
        self.iis = QTextEdit(); self.iis.setReadOnly(True)
        self.iis.setPlaceholderText("(on infeasible plans, the responsible gestures appear here)")
        right.addWidget(self.iis, 2)
        root.addLayout(right, 0)

        self.setStatusBar(QStatusBar())
        self.set_mode("Rect")
        self.redraw(); self.update_formula()

    # ====================================================================
    # environment (same as planner / run_example1)
    # ====================================================================
    def _make_x0(self) -> np.ndarray:
        sx, sy = 118.0, 100.0
        d = np.array([95.0 - sx, 112.0 - sy])
        v = d / (np.linalg.norm(d) + 1e-9) * 0.035
        return np.array([sx, sy, 1.2, v[0], v[1], 0.0])

    def _sample_obstacles(self):
        x_min, x_max, _, _ = self.bounds
        rng = np.random.default_rng(0)
        rules = GeometryConstraints(
            min_goal_obstacle_clearance=4.0, min_start_obstacle_clearance=4.0,
            require_rects_in_bounds=True, require_start_in_bounds=True)
        regions = {n: r.rect for n, r in self.program.regions.items()}  # G1, G2
        return _place_random_obstacles(
            rng=rng, x0=self.x0, regions=regions, n_obstacles=5,
            obstacle_w_min=6.0, obstacle_w_max=10.0,
            obstacle_h_min=6.0, obstacle_h_max=12.0,
            xlim=(x_min[0], x_max[0]), ylim=(x_min[1], x_max[1]),
            rules=rules, max_tries=8000,
            sample_xlim=(60.0, 135.0), sample_ylim=(95.0, 135.0))

    # ====================================================================
    # mode + gesture handling
    # ====================================================================
    def status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def set_mode(self, g: str) -> None:
        if not self.btns[g].isEnabled():
            return
        if g != "Or":
            self.or_anchor = None        # leaving Or finalizes the current combine
        self.mode = g
        for k, b in self.btns.items():
            b.setChecked(k == g)
        self.status(f"Mode: {g}   ·   {MODE_HINT.get(g, '')}")

    def hit_region(self, x, y):
        for name, r in self.program.regions.items():
            if r.shape == "circle" and r.circle is not None:
                cx, cy, rad = r.circle
                if np.hypot(x - cx, y - cy) <= rad:
                    return name
            else:
                x1, x2, y1, y2 = r.rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return name
        return None

    def find_requirement_for(self, name):
        for req in reversed(self.program.requirements):
            if name in req.region_names:
                return req
        return None

    def handle_region_click(self, name: str) -> None:
        """Apply the active tool to a clicked region (callable headlessly).

        Reach/Keep create a single-region requirement. Or combines two EXISTING
        same-operator, same-window requirements into one disjunction (and removes
        the originals), so there is no redundancy and no "Or first" ordering.
        """
        if self.mode in ("Reach", "Keep"):
            kind = "reach" if self.mode == "Reach" else "keep"
            self.last_req = self.program.add_requirement(kind, [name])
            self._invalidate()
            self.status(f"{kind} {name}   ({self.last_req.gid})")
            return
        if self.mode == "Or":
            req = self.find_requirement_for(name)
            if req is None:
                self.status("Or: reach/keep this region first — Or combines requirements")
                return
            if self.or_anchor is None:
                self.or_anchor = req
                self.redraw()
                self.status(f"Or: selected {req.kind} {'∨'.join(req.region_names)} "
                            f"({req.gid}) — click another to combine")
                return
            if req is self.or_anchor:
                return
            if not self.program.can_merge(self.or_anchor, req):
                self.status("Or: requirements must be the same operator and window")
                return
            self.program.merge_or(self.or_anchor, req)
            self.last_req = self.or_anchor
            self._invalidate()
            self.status(f"Or → {self.or_anchor.kind} ("
                        + " ∨ ".join(self.or_anchor.region_names) + f")  ({self.or_anchor.gid})")
            return
        if self.mode == "Window":
            req = self.find_requirement_for(name)
            if req is None:
                self.status(f"{name} has no requirement yet")
                return
            self.action_window(req=req)
        # Region mode handled by drag in on_press/on_release.

    def action_window(self, t1=None, t2=None, req=None):
        req = req or self.last_req
        if req is None:
            self.status("Window: create a requirement first")
            return
        if t1 is None or t2 is None:
            t1, ok1 = QInputDialog.getInt(self, "Window", "t1 (start step):", 0, 0, 100000)
            if not ok1:
                return
            t2, ok2 = QInputDialog.getInt(self, "Window", "t2 (end step):", max(int(t1), 1), int(t1), 100000)
            if not ok2:
                return
        self.program.set_window(req, int(t1), int(t2))
        self._invalidate(); self.status(f"window [{t1},{t2}] on {req.gid}")

    def action_clear_stl(self):
        """Clear the authored STL (requirements) but keep all regions."""
        self.program.requirements.clear()
        self.last_req = None; self.or_anchor = None
        self._invalidate(); self.iis.clear(); self.status("STL cleared (regions kept)")

    def action_clear(self):
        """Clear everything except the fixed runways."""
        self.program.requirements.clear()
        self.program.regions = {n: r for n, r in self.program.regions.items() if r.is_runway}
        self.last_req = None; self.or_anchor = None
        self._invalidate(); self.iis.clear(); self.status("cleared")

    def action_plan(self):
        if not self.program.requirements:
            self.status("Plan: author at least one requirement")
            return None
        self.status("Planning ..."); QApplication.processEvents()
        res = plan_program(self.program, x0=self.x0, bounds=self.bounds, dt=la.DT,
                           v_max=la.V_MAX, cruise_z=la.CRUISE_Z_REF)
        self.result = res
        if res["feasible"]:
            self.iis_regions.clear()
            self.iis.setPlainText("Feasible — trajectory found.")
            self.status(f"Feasible at T={res['T']}")
        else:
            rep = res["iis"]
            self.iis_regions = set(rep.region_names)
            self.iis.setPlainText(rep.text)
            self.status(f"INFEASIBLE at T={res['T']} — see IIS panel")
        self.redraw(); self.update_formula()
        return res

    def _invalidate(self):
        """A gesture edit invalidates any previous plan."""
        self.result = None; self.iis_regions.clear()
        self.redraw(); self.update_formula()

    # ====================================================================
    # rendering
    # ====================================================================
    def _region_role(self, name: str) -> str:
        for req in self.program.requirements:
            if name in req.region_names:
                return req.kind
        return "runway" if self.program.regions[name].is_runway else "region"

    def redraw(self) -> None:
        ax = self.canvas.ax; ax.clear()
        if self.canvas._bg is not None:
            elev, extent = self.canvas._bg
            cmap = plt.cm.terrain.copy(); cmap.set_bad("lightgrey")
            vmin = float(np.nanpercentile(elev, 2)); vmax = float(np.nanpercentile(elev, 98))
            ax.imshow(elev, origin="upper", extent=extent, cmap=cmap, vmin=vmin,
                      vmax=vmax, aspect="auto", interpolation="nearest", zorder=0)

        # obstacles (environment)
        for j, (x1, x2, y1, y2) in enumerate(self.program.obstacles):
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="salmon",
                         edgecolor="darkred", alpha=0.55, linewidth=1.0, zorder=3,
                         label="obstacle" if j == 0 else None))

        role_style = {
            "keep":   ("gold", "goldenrod", "--"),
            "reach":  ("deepskyblue", "blue", "--"),
            "runway": ("red", "darkred", "-"),
            "region": ("white", "black", "--"),
        }
        anchor_names = set(self.or_anchor.region_names) if self.or_anchor else set()
        for name, r in self.program.regions.items():
            fc, ec, ls = role_style[self._region_role(name)]
            if name in self.iis_regions:
                ec, lw, alpha = "red", 3.2, 0.40
            elif name in anchor_names:
                ec, lw, alpha = "cyan", 3.0, 0.38
            else:
                lw, alpha = 2.0, 0.35
            if r.shape == "circle" and r.circle is not None:
                cx, cy, rad = r.circle
                ax.add_patch(mpatches.Circle((cx, cy), rad, facecolor=fc, edgecolor=ec,
                             alpha=alpha, linewidth=lw, linestyle=ls, zorder=4))
            else:
                x1, x2, y1, y2 = r.rect
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=fc,
                             edgecolor=ec, alpha=alpha, linewidth=lw, linestyle=ls, zorder=4))
            ax.text(cx, cy, name, ha="center", va="center",
                    fontsize=10, fontweight="bold", zorder=7)

        ax.plot(self.x0[0], self.x0[1], "^", color="limegreen", markersize=11,
                markeredgecolor="black", zorder=8, label="start")

        if self.result and self.result.get("feasible"):
            X = self.result["X"]; n = len(X)
            seg = plt.cm.plasma(np.linspace(0, 1, max(n - 1, 1)))
            for i in range(n - 1):
                ax.plot(X[i:i+2, 0], X[i:i+2, 1], color="white", linewidth=3.5, zorder=9)
                ax.plot(X[i:i+2, 0], X[i:i+2, 1], color=seg[i], linewidth=2.0, zorder=10)
            ax.plot(X[-1, 0], X[-1, 1], "s", color="red", markersize=10,
                    markeredgecolor="black", zorder=11)

        ax.set_xlim(*la.VIEW_XLIM); ax.set_ylim(*la.VIEW_YLIM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East  px (km)"); ax.set_ylabel("North  py (km)")
        ax.set_title("Authoring canvas — Los Angeles")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        self.canvas.draw_idle()

    def update_formula(self) -> None:
        self.formula.setPlainText(self.program.to_text())

    # ====================================================================
    # mouse handling
    # ====================================================================
    def on_press(self, event) -> None:
        if event.inaxes is not self.canvas.ax or event.xdata is None:
            return
        if self.mode == "Rect":
            self._press = (event.xdata, event.ydata)
            self._rubber = mpatches.Rectangle((event.xdata, event.ydata), 0, 0, fill=False,
                                              edgecolor="black", linewidth=1.5, linestyle=":", zorder=12)
            self.canvas.ax.add_patch(self._rubber)
        elif self.mode == "Circle":
            self._press = (event.xdata, event.ydata)
            self._rubber = mpatches.Circle((event.xdata, event.ydata), 0.0, fill=False,
                                           edgecolor="black", linewidth=1.5, linestyle=":", zorder=12)
            self.canvas.ax.add_patch(self._rubber)
        else:
            name = self.hit_region(event.xdata, event.ydata)
            if name is not None:
                self.handle_region_click(name)
            else:
                self.status(f"Mode: {self.mode} — click on a region")

    def on_motion(self, event) -> None:
        if not self._press or event.inaxes is not self.canvas.ax or event.xdata is None:
            return
        x0, y0 = self._press
        if self.mode == "Rect":
            self._rubber.set_bounds(min(x0, event.xdata), min(y0, event.ydata),
                                    abs(event.xdata - x0), abs(event.ydata - y0))
            self.canvas.draw_idle()
        elif self.mode == "Circle":
            self._rubber.center = (x0, y0)
            self._rubber.set_radius(float(np.hypot(event.xdata - x0, event.ydata - y0)))
            self.canvas.draw_idle()

    def on_release(self, event) -> None:
        if self.mode not in ("Rect", "Circle") or not self._press:
            return
        x0, y0 = self._press; self._press = None
        if self._rubber is not None:
            self._rubber.remove(); self._rubber = None
        if event.xdata is None:
            self.redraw(); return
        if self.mode == "Rect":
            x1, x2 = sorted((x0, event.xdata)); y1, y2 = sorted((y0, event.ydata))
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                self.redraw(); return
            r = self.program.add_region((x1, x2, y1, y2))
        else:  # Circle
            rad = float(np.hypot(event.xdata - x0, event.ydata - y0))
            if rad < 1.0:
                self.redraw(); return
            r = self.program.add_circle_region(x0, y0, rad)
        self.last_req = None
        self._invalidate(); self.status(f"region {r.name} ({r.shape}) drawn")

    # convenience for headless tests
    def create_region(self, rect, name=None):
        r = self.program.add_region(rect, name=name)
        self._invalidate()
        return r.name

    def create_circle(self, cx, cy, rad, name=None):
        r = self.program.add_circle_region(cx, cy, rad, name=name)
        self._invalidate()
        return r.name


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(1280, 600); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
