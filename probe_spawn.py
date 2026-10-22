"""probe_spawn.py -- can we make ViZDoom show different enemy types on demand (needed for a
multi-choice recognition task)?  Run:  python probe_spawn.py
Saves probe_frames/<name>.png so you can look at what each attempt produced."""
import os, sys
print("python", sys.version.split()[0])
import numpy as np, vizdoom as vzd, pygame
print("vizdoom", getattr(vzd, "__version__", "?"))
os.makedirs("probe_frames", exist_ok=True)

g = vzd.DoomGame()
g.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad")); g.set_doom_map("map01")
g.set_screen_resolution(vzd.ScreenResolution.RES_640X480); g.set_screen_format(vzd.ScreenFormat.RGB24)
g.set_labels_buffer_enabled(True); g.set_render_hud(False); g.set_render_weapon(False)
g.set_window_visible(False); g.set_mode(vzd.Mode.PLAYER); g.set_episode_timeout(0)
g.clear_available_buttons(); g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
g.init()

def snap(tag):
    s = g.get_state()
    if s is None:
        print(f"  [{tag}] state is None"); return
    labs = [(l.object_name, l.x, l.y, l.width, l.height) for l in s.labels if l.object_name != "DoomPlayer"]
    print(f"  [{tag}] labels: {labs}")
    pygame.image.save(pygame.surfarray.make_surface(np.ascontiguousarray(s.screen_buffer).swapaxes(0, 1)),
                      f"probe_frames/{tag}.png")

print("has send_game_command:", hasattr(g, "send_game_command"))
g.new_episode(); g.make_action([0.0], 2); snap("baseline")
for cmd_set in (["sv_cheats 1", "kill monsters"], ["sv_cheats 1", "removeall"], ["sv_cheats 1"]):
    for actor in ("Zombieman", "DoomImp", "Cacodemon", "Demon", "BaronOfHell"):
        try:
            g.new_episode()
            for c in cmd_set: g.send_game_command(c)
            g.make_action([0.0], 2)
            g.send_game_command(f"summon {actor}")
            g.make_action([0.0], 4)
            snap(f"{'_'.join(cmd_set).replace(' ', '')}__{actor}")
        except Exception as e:
            print(f"  [{cmd_set}/{actor}] ERROR {e!r}")
g.close()
print("done -> look at probe_frames/ and paste this output back")
