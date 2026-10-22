"""
doom_recog_env.py -- multi-choice Doom recognition stimuli (RTify-style n-AFC).

Each trial: a monster of one of N types is summoned in front of the player in basic.wad
(sv_cheats 1 + summon, verified by your probe), the player walks backwards until the sprite is
about `target_h` px tall (so apparent size doesn't give the answer away), the scenario's stock
Cacodemon is painted out using the labels buffer, and the frame is optionally degraded
(contrast / noise) to make difficulty a controlled variable (the analogue of coherence).
"""
from dataclasses import dataclass
import numpy as np

CLASSES = ["Zombieman", "DoomImp", "Demon", "Cacodemon", "BaronOfHell"]
NICE = {"Zombieman": "Zombieman", "DoomImp": "Imp", "Demon": "Demon", "Cacodemon": "Cacodemon",
        "BaronOfHell": "Baron", "ShotgunGuy": "Shotgunner", "ChaingunGuy": "Chaingunner", "LostSoul": "Lost Soul"}


@dataclass
class RStim:
    frame: np.ndarray       # H x W x 3 uint8 (already degraded)
    true_idx: int
    true_name: str
    contrast: float
    noise: float
    bbox_w: int
    bbox_h: int
    back_tics: int


def degrade(frame, contrast, noise, rng):
    if contrast >= 0.999 and noise <= 0:
        return frame
    f = frame.astype(np.float32)
    m = f.mean()
    f = m + contrast * (f - m)
    if noise > 0:
        f = f + rng.normal(0, noise * 255.0, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def inpaint_rows(frame, mask):
    """Fill masked pixels from the nearest unmasked pixel to the left (else right) in each row."""
    H, W = mask.shape
    cols = np.arange(W)[None, :].repeat(H, 0)
    valid = ~mask
    left = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)
    right = np.minimum.accumulate(np.where(valid, cols, W + 10)[:, ::-1], axis=1)[:, ::-1]
    src = np.where(left >= 0, left, np.minimum(right, W - 1))
    out = frame.copy()
    rows = np.arange(H)[:, None].repeat(W, 1)
    out[mask] = frame[rows[mask], np.clip(src[mask], 0, W - 1)]
    return out


class DoomRecogEnv:
    def __init__(self, width=640, height=480, seed=None, classes=None, target_h=200):
        import os, vizdoom as vzd
        self.vzd, self.W, self.H = vzd, width, height
        self.classes = list(classes or CLASSES)
        self.target_h = target_h
        self.rng = np.random.default_rng(seed)
        g = vzd.DoomGame()
        g.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad")); g.set_doom_map("map01")
        g.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{width}X{height}"))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_labels_buffer_enabled(True)
        g.set_render_hud(False); g.set_render_weapon(False); g.set_render_crosshair(False)
        g.set_window_visible(False); g.set_mode(vzd.Mode.PLAYER); g.set_episode_timeout(0)
        g.clear_available_buttons()
        g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
        g.add_available_button(vzd.Button.MOVE_BACKWARD)
        if seed is not None:
            g.set_seed(int(seed))
        g.init()
        self.game = g

    def _act(self, back=0, tics=1):
        self.game.make_action([0.0, 1 if back else 0], tics)

    def _non_player(self):
        s = self.game.get_state()
        return [] if s is None else [l for l in s.labels if l.object_name != "DoomPlayer"]

    def make_trial(self, class_idx, contrast=1.0, noise=0.0):
        g, name = self.game, self.classes[class_idx]
        g.new_episode()
        for c in ("sv_cheats 1", "god", "notarget"):
            g.send_game_command(c)
        self._act(tics=2)
        stock = {l.object_id for l in self._non_player()}
        g.send_game_command(f"summon {name}")
        self._act(tics=3)

        def target():
            new = [l for l in self._non_player() if l.object_id not in stock]
            return max(new, key=lambda l: l.width * l.height) if new else None

        t, back, last_h = target(), 0, None
        if t is None:
            raise RuntimeError(f"summon {name} produced no new label; run probe_spawn.py again")
        while t.height > self.target_h and back < 90:
            self._act(back=True, tics=5); back += 5
            t = target()
            if t is None:
                raise RuntimeError("target lost while backing up")
            if last_h is not None and t.height >= last_h:     # stuck against a wall
                break
            last_h = t.height
        self._act(tics=2)
        t = target() or t
        s = g.get_state()
        frame = np.ascontiguousarray(s.screen_buffer)
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        lb = s.labels_buffer
        mask = np.zeros(lb.shape, bool)
        for l in s.labels:
            if l.object_name != "DoomPlayer" and l.object_id in stock:
                mask |= (lb == l.value)
        if mask.any():
            frame = inpaint_rows(frame, mask)
        frame = degrade(frame, contrast, noise, self.rng)
        return RStim(frame, class_idx, name, contrast, noise, int(t.width), int(t.height), back)

    def info(self):
        return {"backend": "vizdoom", "classes": self.classes, "target_h": self.target_h,
                "width": self.W, "height": self.H}

    def close(self):
        self.game.close()


class FakeRecogEnv:
    """Coloured rectangles per class -- pipeline test without ViZDoom."""
    def __init__(self, width=640, height=480, seed=None, classes=None, target_h=200):
        self.W, self.H, self.classes = width, height, list(classes or CLASSES)
        self.rng = np.random.default_rng(seed)

    def make_trial(self, class_idx, contrast=1.0, noise=0.0):
        f = np.full((self.H, self.W, 3), 70, np.uint8)
        col = [(200, 60, 60), (60, 200, 60), (60, 60, 200), (200, 200, 60), (200, 60, 200)][class_idx % 5]
        f[140:340, 270:370] = col
        return RStim(degrade(f, contrast, noise, self.rng), class_idx, self.classes[class_idx], contrast, noise, 100, 200, 0)

    def info(self):
        return {"backend": "fake", "classes": self.classes}

    def close(self):
        pass
