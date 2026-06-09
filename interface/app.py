"""
Click2STL authoring interface (PyQt6) — working interface for Example 1.

Mode-based gestures (paper Table 1). Pick a tool on the left, then act on the map:
  - Region : drag to draw a rectangle referent.
  - Eventually : click a region  -> ◇ (reach) requirement on it.
  - Always     : click a region  -> □ (keep) requirement on it.
  - Until      : drag from a key region to a door region -> ¬door U key.
  - Or     : click two existing requirements (by their regions) -> merge them into
             one disjunction ◇/□(R1 ∨ R2 ∨ …); originals are removed (keep clicking
             to fold in more). Same operator + same window required.
  - Window : click a requirement's region -> type the [t1,t2] interval.
  - Plan   : translate gestures -> STL, solve MICP, draw trajectory; if infeasible,
             extract the IIS and highlight the responsible gestures (Section 4).

Start position and obstacles are environment-supplied, sampled with the same
planner routine used by run_example1, and drawn on the map. Fixed runways G1
(LAX) / G2 (Ontario) are pre-placed.

Run:  ./run_interface.sh   (or python -m interface.app)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
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

MODES = ["Rect", "Circle", "Eventually", "Always", "Until", "Or", "Window"]
MODE_HINT = {
    "Rect":       "drag to draw a rectangular region",
    "Circle":     "drag from the centre outward to set the radius",
    "Eventually": "click a region  →  ◇ eventually (reach)",
    "Always":     "click a region  →  □ always (keep)",
    "Until":      "drag from a key region to the door region  →  ¬door U key",
    "Or":         "click two requirements (their regions) to combine into ◇/□(· ∨ ·)",
    "Window":     "click a requirement's region  →  type [t1, t2]",
}
# button label -> requirement kind
_KIND = {"Eventually": "reach", "Always": "keep"}


class MapCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(8, 4.7))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.12)
        self._bg = load_la_dem()


class MainWindow(QMainWindow):
    def __init__(self, seed=None) -> None:
        super().__init__()
        # obstacle sampling seed: use the given one, else a fresh random seed
        # (shown in the title so a run can be reproduced with ./run_interface.sh <seed>)
        self.seed = int(seed) if seed is not None else int(np.random.default_rng().integers(1_000_000))
        self.setWindowTitle(f"Click2STL — Authoring Interface   (obstacle seed {self.seed})")
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
        self.iis_buttons: set[str] = set()   # palette buttons to flag red (e.g. Window)
        self._press = None
        self._rubber = None
        self._until_start = None      # (key_name, (cx,cy)) while dragging an Until arrow
        self._arrow = None

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
        self.formula.setMinimumWidth(200); self.formula.setMaximumWidth(240)
        right.addWidget(self.formula, 3)
        self.iis_label = QLabel("<b>Infeasibility resolution</b>")
        right.addWidget(self.iis_label)
        self.iis = QTextEdit(); self.iis.setReadOnly(True)
        self.iis.setPlaceholderText("(on infeasible plans, the responsible gestures appear here)")
        self.iis.setMinimumWidth(200); self.iis.setMaximumWidth(240)
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
        rng = np.random.default_rng(self.seed)
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

    def region_center(self, name):
        r = self.program.regions[name]
        if r.shape == "circle" and r.circle is not None:
            return (r.circle[0], r.circle[1])
        x1, x2, y1, y2 = r.rect
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def handle_region_click(self, name: str) -> None:
        """Apply the active tool to a clicked region (callable headlessly).

        Eventually/Always create a single-region requirement. Or combines two EXISTING
        same-operator, same-window requirements into one disjunction (and removes
        the originals), so there is no redundancy and no "Or first" ordering.
        """
        if self.mode in _KIND:
            kind = _KIND[self.mode]
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
        # Region / Until modes handled by drag in on_press/on_release.

    def handle_until(self, key: str, door: str):
        """Create an until requirement ¬door U key (callable headlessly).

        Absorbs a redundant same-window reach on the key, since ¬door U key
        already entails ◇key (keeps the authored spec free of redundant conjuncts).
        """
        if key == door:
            self.status("Until: key and door must differ")
            return None
        req = self.program.add_requirement("until", [key, door])
        removed = self.program.absorb_reach_into_until(req)
        self.last_req = req
        self._invalidate()
        msg = f"until  ¬{door} U {key}   ({req.gid})"
        if removed:
            msg += f"   (absorbed redundant reach {removed})"
        self.status(msg)
        return req

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
            self.iis_buttons.clear()
            self.iis.setPlainText("Feasible — trajectory found.")
            self._set_iis_infeasible_style(False)
            self.status(f"Feasible at T={res['T']}")
        else:
            rep = res["iis"]
            self.iis_regions = set(rep.region_names)
            # flag the Window button if any implicated requirement carries a window
            self.iis_buttons = {"Window"} if any(r.window_gid for r in rep.requirements) else set()
            self.iis.setPlainText(rep.text)
            self._set_iis_infeasible_style(True)
            self.status(f"INFEASIBLE at T={res['T']} — see IIS panel")
        self._apply_button_highlights()
        self.redraw(); self.update_formula()
        return res

    def _set_iis_infeasible_style(self, on: bool):
        """Turn the IIS label/box transparent red on infeasible plans; else restore."""
        if on:
            self.iis_label.setStyleSheet("color: red;")
            self.iis.setStyleSheet("background-color: rgba(255, 0, 0, 40);")
        else:
            self.iis_label.setStyleSheet("")
            self.iis.setStyleSheet("")

    def _invalidate(self):
        """A gesture edit invalidates any previous plan."""
        self.result = None; self.iis_regions.clear(); self.iis_buttons.clear()
        self._set_iis_infeasible_style(False)
        self._apply_button_highlights()
        self.redraw(); self.update_formula()

    def _apply_button_highlights(self):
        """Flag palette buttons in self.iis_buttons red; restore the rest."""
        for name, b in self.btns.items():
            if name in self.iis_buttons:
                b.setStyleSheet("background-color: #ff6b6b; font-weight: bold;")
            else:
                b.setStyleSheet("")

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
            "runway": ("royalblue", "navy", "-"),
            "until":  ("mediumseagreen", "darkgreen", "--"),
            "region": ("white", "black", "--"),
        }
        anchor_names = set(self.or_anchor.region_names) if self.or_anchor else set()
        for name, r in self.program.regions.items():
            fc, ec, ls = role_style[self._region_role(name)]
            failed = name in self.iis_regions
            hatch = None
            if failed:
                fc, ec, ls, lw, alpha, hatch = "red", "red", "-", 4.0, 0.50, "xx"
            elif name in anchor_names:
                ec, lw, alpha = "cyan", 3.0, 0.38
            else:
                lw, alpha = 2.0, 0.35
            if r.shape == "circle" and r.circle is not None:
                cx, cy, rad = r.circle
                ax.add_patch(mpatches.Circle((cx, cy), rad, facecolor=fc, edgecolor=ec,
                             alpha=alpha, linewidth=lw, linestyle=ls, hatch=hatch, zorder=4))
            else:
                x1, x2, y1, y2 = r.rect
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rad = max(x2 - x1, y2 - y1) / 2.0
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=fc,
                             edgecolor=ec, alpha=alpha, linewidth=lw, linestyle=ls,
                             hatch=hatch, zorder=4))
            if failed:
                # big, hard-to-miss failure indicator: red ring + bold ✗ badge
                ax.add_patch(mpatches.Circle((cx, cy), rad * 1.9, fill=False,
                             edgecolor="red", linewidth=3.0, linestyle="--", zorder=11))
                ax.plot(cx, cy, marker="X", markersize=26, markeredgewidth=2.5,
                        color="red", markeredgecolor="white", zorder=12)
                ax.text(cx, cy + rad * 2.1, name, ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="red", zorder=12,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
            else:
                ax.text(cx, cy, name, ha="center", va="center",
                        fontsize=10, fontweight="bold", zorder=7)

        # until arrows: key --> door  (¬door U key)
        for req in self.program.requirements:
            if req.kind == "until":
                key, door = req.region_names
                ax.add_patch(mpatches.FancyArrowPatch(
                    self.region_center(key), self.region_center(door),
                    arrowstyle="-|>", mutation_scale=14, color="purple",
                    linewidth=1.8, alpha=0.9, zorder=6))

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
        elif self.mode == "Until":
            name = self.hit_region(event.xdata, event.ydata)
            if name is None:
                self.status("Until: start the drag on a key region")
                return
            c = self.region_center(name)
            self._until_start = (name, c)
            self._arrow = mpatches.FancyArrowPatch(c, (event.xdata, event.ydata),
                                                   arrowstyle="-|>", mutation_scale=14,
                                                   color="purple", linewidth=1.8, zorder=13)
            self.canvas.ax.add_patch(self._arrow)
        else:
            name = self.hit_region(event.xdata, event.ydata)
            if name is not None:
                self.handle_region_click(name)
            else:
                self.status(f"Mode: {self.mode} — click on a region")

    def on_motion(self, event) -> None:
        if event.inaxes is not self.canvas.ax or event.xdata is None:
            return
        if self.mode == "Until" and self._until_start and self._arrow is not None:
            self._arrow.set_positions(self._until_start[1], (event.xdata, event.ydata))
            self.canvas.draw_idle()
            return
        if not self._press:
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
        if self.mode == "Until":
            if not self._until_start:
                return
            key = self._until_start[0]; self._until_start = None
            if self._arrow is not None:
                self._arrow.remove(); self._arrow = None
            door = None if event.xdata is None else self.hit_region(event.xdata, event.ydata)
            if door is None or door == key:
                self.status("Until: release on the door region (a different region)")
                self.redraw(); return
            self.handle_until(key, door)
            return
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
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", nargs="?", type=int, default=None,
                    help="obstacle sampling seed (omit for a random one)")
    args, _ = ap.parse_known_args()
    app = QApplication(sys.argv)
    win = MainWindow(seed=args.seed); win.resize(1040, 500); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
