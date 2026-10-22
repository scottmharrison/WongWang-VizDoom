"""probe_spawn2.py -- spawn works; now: hide the stock Cacodemon, control distance, keep monsters still."""
import os
import numpy as np, vizdoom as vzd, pygame
os.makedirs("probe_frames2", exist_ok=True)
g = vzd.DoomGame()
g.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad")); g.set_doom_map("map01")
g.set_screen_resolution(vzd.ScreenResolution.RES_640X480); g.set_screen_format(vzd.ScreenFormat.RGB24)
g.set_labels_buffer_enabled(True); g.set_render_hud(False); g.set_render_weapon(False)
g.set_window_visible(False); g.set_mode(vzd.Mode.PLAYER); g.set_episode_timeout(0)
g.clear_available_buttons(); g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
g.add_available_button(vzd.Button.MOVE_BACKWARD); g.init()

def labs():
    s = g.get_state()
    return None if s is None else [(l.object_name, l.x, l.y, l.width, l.height) for l in s.labels if l.object_name != "DoomPlayer"]
def snap(tag):
    s = g.get_state()
    if s is None: print(f"  [{tag}] state None"); return
    print(f"  [{tag}] {labs()}")
    pygame.image.save(pygame.surfarray.make_surface(np.ascontiguousarray(s.screen_buffer).swapaxes(0, 1)), f"probe_frames2/{tag}.png")

for variant, cmds in (("A_god", ["sv_cheats 1", "god", "notarget"]), ("B_freeze", ["sv_cheats 1", "god", "notarget", "freeze"])):
    for back in (0, 10, 25):
        for actor in ("Zombieman", "DoomImp", "BaronOfHell"):
            try:
                g.new_episode()
                for c in cmds: g.send_game_command(c)
                g.make_action([60.0, 0], 3)                 # turn away from the stock Cacodemon
                before = labs()
                g.send_game_command(f"summon {actor}")
                g.make_action([0.0, 0], 2)
                if back: g.make_action([0.0, 1], back)       # walk backwards to get distance
                print(f"{variant} back={back} {actor}: before-summon labels={before}")
                snap(f"{variant}_back{back}_{actor}")
            except Exception as e:
                print(f"{variant} back={back} {actor}: ERROR {e!r}")
g.close(); print("done -> paste output; look at probe_frames2/")
