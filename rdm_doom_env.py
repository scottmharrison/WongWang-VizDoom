"""
rdm_doom_env.py -- Doom-based "left/right" perceptual decision stimuli.

Task analogue of random-dot-motion (RDM):
  * The ViZDoom `basic.wad` scenario puts a stationary monster on the far wall.
  * Before each trial the (invisible) player is turned so the monster sits at a
    controlled horizontal offset from screen centre (the crosshair).
  * Difficulty = |offset| in pixels (small = hard, like low motion coherence).
  * Correct answer = which side of the crosshair the monster is on.

Two interchangeable back-ends:
  RDMDoomEnv    real ViZDoom
  SimulatedEnv  no-dependency fake frames, for testing the logging/trial logic
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Stim:
    frame: np.ndarray        # H x W x 3 uint8 RGB
    true_side: str           # "left" | "right" (monster relative to crosshair)
    offset_target_px: float  # signed target offset (+ = right)
    offset_measured_px: float  # signed, measured from the labels buffer
    width: int
    height: int


class RDMDoomEnv:
    def __init__(self, width=640, height=480, seed=None, visible=False):
        import vizdoom as vzd
        import os
        self.vzd = vzd
        self.W, self.H = width, height
        g = vzd.DoomGame()
        g.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
        g.set_doom_map("map01")
        res = getattr(vzd.ScreenResolution, f"RES_{width}X{height}", None)
        if res is None:
            raise ValueError(f"Unsupported resolution {width}x{height}; try 640x480, "
                             "800x600, 320x240, 1024x768")
        g.set_screen_resolution(res)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_labels_buffer_enabled(True)
        g.set_render_hud(False)
        g.set_render_weapon(False)
        g.set_render_crosshair(True)      # crosshair = the reference point
        g.set_render_decals(False)
        g.set_render_particles(False)
        g.set_window_visible(visible)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_episode_timeout(0)
        g.set_episode_start_time(1)
        g.clear_available_buttons()
        g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
        if seed is not None:
            g.set_seed(int(seed))
        g.init()
        self.game = g
        self._new_episode()
        self.gain = self._calibrate()   # px of screen shift per unit of turn

    # ---- low level -------------------------------------------------------
    def _new_episode(self):
        self.game.new_episode()

    def _monster_x(self):
        """Signed px offset of monster centre from screen centre, or None."""
        s = self.game.get_state()
        if s is None or s.labels_buffer is None:
            return None
        lb = s.labels_buffer
        best = None
        for lab in s.labels:
            if lab.object_name == "DoomPlayer":
                continue
            if best is None or lab.width * lab.height > best.width * best.height:
                best = lab
        if best is None:
            return None
        cols = np.where((lb == best.value).any(axis=0))[0]
        if cols.size == 0:
            return None
        return (cols.min() + cols.max()) / 2.0 + 0.5 - self.W / 2.0

    def _turn(self, d):
        self.game.make_action([float(d)], 1)
        self.game.make_action([0.0], 2)      # settle: let the turn finish before reading

    def _find_monster(self):
        x = self._monster_x()
        for _ in range(24):          # scan if it started off-screen
            if x is not None:
                return x
            self._turn(15)
            x = self._monster_x()
        raise RuntimeError("Could not find the monster in the labels buffer.")

    def _diag(self):
        s = self.game.get_state()
        if s is None:
            return "state is None (episode finished?)"
        names = [(l.object_name, l.value, l.x, l.y, l.width, l.height) for l in s.labels]
        lb = s.labels_buffer
        return (f"labels={names}; labels_buffer={'None' if lb is None else (lb.shape, sorted(set(lb.ravel().tolist()))[:10])}; "
                f"screen={s.screen_buffer.shape}")

    def _calibrate(self):
        """Measure px shift per unit of turn; also detect if fractional turns are honoured."""
        x0 = self._find_monster()
        gain = None
        for mag in (1.0, 3.0, 10.0, 30.0):
            self._turn(mag)
            x1 = self._monster_x()
            self._turn(-mag)
            if x1 is not None and abs(x1 - x0) > 1e-3:
                gain = (x1 - x0) / mag
                break
        if gain is None:
            raise RuntimeError("Turn calibration failed: no shift seen for turns of 1..30 units. "
                               f"x0={x0}. {self._diag()}")
        # does a fractional turn do anything?  (tests for integer-quantised input)
        xa = self._find_monster()
        self._turn(0.3)
        xb = self._monster_x()
        self._turn(-0.3)
        self.quantised = (xb is None) or abs(xb - xa) < 0.3 * abs(gain) * 0.3
        return gain

    def _align(self, target, tol=0.5, max_iter=10):
        if getattr(self, 'quantised', False):
            tol = max(tol, abs(self.gain) * 0.5)
        x = self._find_monster()
        for _ in range(max_iter):
            err = target - x
            if abs(err) <= tol:
                break
            self._turn(err / self.gain)
            x = self._find_monster()
        return x

    # ---- public ----------------------------------------------------------
    def make_trial(self, offset_px, side):
        """side: 'left' or 'right'; offset_px: magnitude (>0)."""
        target = abs(offset_px) * (1 if side == "right" else -1)
        self._new_episode()
        x = self._align(target)
        s = self.game.get_state()
        frame = s.screen_buffer
        if frame.ndim == 3 and frame.shape[0] == 3:      # CHW -> HWC
            frame = np.transpose(frame, (1, 2, 0))
        return Stim(np.ascontiguousarray(frame), "right" if x > 0 else "left",
                    target, float(x), self.W, self.H)

    def info(self):
        return {"backend": "vizdoom", "vizdoom_version": getattr(self.vzd, "__version__", "?"),
                "width": self.W, "height": self.H, "turn_gain_px_per_unit": float(self.gain), "turn_quantised": bool(self.quantised)}

    def close(self):
        self.game.close()


class SimulatedEnv:
    """Fake env: grey frame with a red box at the target offset."""
    def __init__(self, width=640, height=480, seed=None, visible=False):
        self.W, self.H = width, height
        self.rng = np.random.default_rng(seed)

    def make_trial(self, offset_px, side):
        target = abs(offset_px) * (1 if side == "right" else -1)
        x = target + self.rng.normal(0, 0.2)
        f = np.full((self.H, self.W, 3), 90, np.uint8)
        cx = int(round(self.W / 2 + x))
        f[self.H // 2 - 20:self.H // 2 + 20, max(cx - 15, 0):cx + 15] = (200, 30, 30)
        return Stim(f, "right" if x > 0 else "left", target, float(x), self.W, self.H)

    def info(self):
        return {"backend": "simulated", "width": self.W, "height": self.H}

    def close(self):
        pass
