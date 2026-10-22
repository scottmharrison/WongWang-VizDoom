# WongWang-VizDoom

Wong-Wang (WW) decision networks, and an extended **Wong-Wang+** model (critical roaming + a Logan-style
instance route feeding a leaky competing accumulator), that **play ViZDoom** so you can watch their decision
dynamics (attractor rates, race between routes, roaming input, reaction times) in real time.

It is the experimental/demo platform for the *StarPower 2* project on reaction-time (RT) distribution shape and
heavy tails.

> **Status / honesty note.** Model code is tested headless (`--fake-env`). The pygame dashboard, the
> enemy-slowing mod, `--god` and `--infinite-ammo` were written without access to a real Doom install and may need
> small fixes on your machine. The tier presets are hand-tuned starting points, not fitted values (see below).

---

## 1. The models

All three share the same core and differ in how many mechanisms are switched on. They are called **tiers**:

| tier | name | what it is |
|---|---|---|
| `ww` | Wong-Wang | Reduced two-variable-per-pool attractor network (Wong & Wang 2006), generalised to *n* pools. Constant input level `u0`. |
| `ww_cr` | WW + critical roaming | As `ww`, but `u0` wanders from decision to decision (clipped Ornstein-Uhlenbeck chain) around the network's **critical point** `u0* ≈ 10.68`. |
| `ww_cr_lca` | **Wong-Wang+** | `ww_cr` plus a fast **instance-retrieval -> LCA** route that races the WW route. First route to threshold wins. |

### Wong-Wang core (`ww_npool.py`)
Each choice has a pool with NMDA-like synaptic gating `S_k`. In 1 ms Euler steps:

```
I_k  = I_bg + J_ext·u0·(1 + coh_k) + w_self·S_k + w_lat·(ΣS − S_k) + OU noise
r_k  = phi(I_k)                         # firing rate, Hz (Abbott-Chance f-I curve)
dS_k = −S_k/τ_s + γ(1 − S_k)·r_k/1000
```

A decision is made when any pool's rate reaches 15 Hz; **RT = decision time + non-decision time Ter**
(Gaussian, ~200 ms). `coh_k` is the evidence for choice *k*; `u0` is the overall input level (an arousal-like gain).
Near `u0*` the network's decision landscape flattens and decisions become slow.

### Critical roaming (`ww_cr`)
`u0` follows a clipped OU chain (θ = 0.1, stationary sd `u0_sd`, range [0.5, 33]). Epochs where `u0` drifts toward
`u0*` give very slow, lapse-like RTs.

### Instance / LCA fast route (`ww_cr_lca`)
Logan (1988) instance theory: with practice, `N` stored instances race to be retrieved; the fastest retrieval
wins, so retrieval time shrinks as a power of `N` (Weibull, `s_W·N^(−1/c)`). Retrieval strength grows as
`ρ_m = ρ_max·N/(N+200)`. The retrieved memory drives a **leaky competing accumulator** (Usher & McClelland 2001;
k = 0.2, β = 0.3, τ = 20 ms, threshold 1) whose input is `ρ_k = g·ρ_m·((1−bias)/n + bias·q_k)` with
`g = clip(u0/u0*, 0, 3)` and `q = softmax(logits)`. The LCA starts at the retrieval onset and races the WW route.
The LCA parameters come from the fitted Haddara E2 hybrid; the 5-way WW presets (`u0=16`, `gain=0.05`,
`noise=0.02`, `u0_sd=4`) are hand-tuned. Making the LCA drive depend on the evidence posterior `q` is a design choice
of this code, not something the original fits tested.

Evidence mapping (`sp2_nway.NWayAgent.decide`): `coh_k = clip(gain·(logit_k − mean(logit)), ±0.9)`.

---

## 2. Why Wong-Wang+? (theoretical motivation)

* **Base WW cannot fit our RT data.** In the StarCraft 2 data of StarPower 1 (Harrison, Chuong, O'Camb & Blair),
  RTs have a *fast front* (experts have ~23% of RTs under 140 ms) and a *heavy, power-law-like tail* (generalised
  beta-prime / GB2 fits dominate for novices; the thin-tailed ex-Gaussian takes over with skill). A single WW attractor
  gives neither the fast front nor the heavy tail.
* **Fast front -> instance route.** Practised responses become automatic by retrieval of stored instances
  (Logan 1988); a fast LCA accumulator driven by retrieval captures ultra-fast responses that grow with skill.
* **Heavy tail -> attention/arousal roaming.** Slow fluctuations of the network's overall excitability around a
  critical point (Pathak & Battaglia, critical roaming) produce lapses and very long RTs, consistent with engaged/
  disengaged latent states seen in behaviour (Ashwood et al. 2022).
* **Power-law-like tails as a mixture of exponentials.** Noise-driven escape from the spontaneous state has an
  approximately exponential waiting time (Martí et al. 2008). If `u0` roams, the escape rate `k` varies between
  trials; the survival function `S(t) = ∫ e^{−kt} dG(k)` is a *hyperexponential* mixture that mimics a power law over
  a wide range. In fixed-drive WW, `ln k ≈ −1.34 + 0.193·u0`, so Gaussian roaming of `u0` gives log-normal rates.
  This is the **working hypothesis**; in our analyses it beats pure *critical slowing down* as an explanation, but it
  is **still open** and a more complex picture may hold.
* Related landscape/flux work on WW decisions: Ye & Li (2021), Brinkman et al. (2022).

---

## 3. The ViZDoom demo: how it works

Scenario `defend_the_center` (agent stands in the middle, enemies approach). **Independent WW+ networks** are the
only decision-makers:

```
ViZDoom labels buffer -> nearest enemy's horizontal offset dx -> hand-written evidence logits
   -> net (ww | ww_cr | ww_cr_lca) -> choice + RT -> short key press (pulse controller) -> game
```

| net | choices | job |
|---|---|---|
| **MOVE** | LEFT vs RIGHT | If no enemy is visible it *searches* (arrow held, decisions pick/reverse the side); once an enemy is visible it turns toward it in short pulses. |
| **SHOOT** | no-shoot vs SHOOT | Decides when (and how often) to fire; evidence favours shooting when the enemy is near the crosshair. |

Real input is only `dx = (enemy_centre_x − W/2)/(W/2)` from the game's object-label buffer (nearest/tallest enemy).
Evidence for each net is a fixed function of `dx` (MOVE: `ev·min(|dx|/move_scale,1)` toward the enemy; SHOOT:
`ev·[al, 1−al]`, `al=min(|dx|/aim_tol,1)`; SEARCH: weak memory of the side the last enemy left by). Nets never see
health, ammo, distance, enemy count or each other. **No neural-network/CNN front end is used.** The pulse controller
(how long a key stays down) and the enemy mod are scaffolding.

The window shows, per net, seven live panels: WW rates, LCA activation, the WW-vs-LCA race, evidence/input, roaming
`u0` vs critical `u0*`, the effective attractor landscape, and an RT history. A screenshot of the dashboard is saved at
every committed decision.

---

## 4. Setup

Python 3.9+ (the ViZDoom wheel availability depends on your Python version; 3.9-3.12 are the safest, newer versions
have worked for some users). Use a virtual environment.

```bash
git clone https://github.com/scottmharrison/WongWang-VizDoom
cd WongWang-VizDoom
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, vizdoom, pygame
```

**ViZDoom** is a pip package (`pip install vizdoom`) that ships the Doom engine and the `defend_the_center` scenario.
* **Windows / macOS:** prebuilt wheels cover recent Python versions; if `pip` tries to compile and fails, switch to
  Python 3.10-3.12.
* **Linux:** wheels are on PyPI; if building from source you need `cmake`, a C++ compiler, SDL2, Boost and OpenAL
  (see the [ViZDoom installation docs](https://github.com/Farama-Foundation/ViZDoom)). A display is needed for the
  window; use `--png` / `--compare` for headless runs.
* Test: `python -c "import vizdoom; print(vizdoom.__version__)"`.

No-install smoke test (no Doom, no window):

```bash
python play_nets.py --compare oracle,ww,ww_cr,ww_cr_lca --episodes 2 --fake-env
```

---

## 5. Running it

### Watch the networks play (pygame window)
```bash
python play_nets.py                          # default tier ww_cr_lca, MOVE + SHOOT nets
python play_nets.py --speed 0.25             # slow motion to see the traces unfold
python play_nets.py --god --infinite-ammo    # keep playing so the dynamics keep going
```
Keys: `SPACE` pause · `1/2/3` switch tier (ww / ww_cr / ww_cr_lca) · `+/-` speed · `S` screenshot · `ESC` quit.

### Wong-Wang on its own vs Wong-Wang+
```bash
python play_nets.py --agent ww               # base Wong-Wang only
python play_nets.py --agent ww_cr            # + critical roaming
python play_nets.py --agent ww_cr_lca        # full Wong-Wang+ (default)
python play_nets.py --nets shoot             # only the SHOOT net (turning follows a hard rule)
python play_nets.py --nets move              # only the MOVE net (firing follows a hard rule)
```

### Compare tiers headless (table of accuracy and RT per net)
```bash
python play_nets.py --compare oracle,ww,ww_cr,ww_cr_lca --episodes 5 --skill 1
```
`oracle`/`idle` are non-model baselines. Add `--fake-env` to run without Doom.

### Change model/evidence parameters
```bash
python play_nets.py --u0 14 --gain 0.08 --noise 0.03 --ter 180 --ev 12 --aim-tol 0.15
python play_nets.py --params '{"shoot":{"u0":18},"move":{"u0_sd":6,"bias":0.6}}'
```
Other flags: `--timeout` (ms per decision), `--turn-gain`, `--turn-tics`, `--fire-tics`, `--deadband`,
`--search-ev`, `--lost-grace`, `--scenario`, `--skill`, `--seed`, `--csv FILE` / `--no-csv`, `--shots-dir`,
`--no-shots`. `python play_nets.py -h` lists everything. Results (one row per decision: tic, net, dx, response,
RT, route, u0) go to a CSV.

Dashboard image without a window: `python play_nets.py --png dash.png --snap 200,500 --fake-env`.

### Enemy changes (a ViZDoom mod)
By default the fast melee monsters (Demon, Spectre, chainsaw marine) are replaced by slow gun-carrying zombies so
the agent cannot be swarmed, via a small generated DECORATE `.pk3` loaded with `-file`. `--keep-melee` disables this,
`--demon-speed`, `--demon-health`, `--shooter-health` tweak the actors. *This mod is untested on real Doom.*

### Other entry points
```bash
python run_human_task.py --participant me                  # human RDM-analogue task (monster left/right of crosshair)
python run_human_task.py --agent ww_cr_lca --trials 600    # model does the same task; same CSV as a human
python run_human_task.py --agent ww_cr --agent-params '{"u0_sd": 8}' --watch --speed 3
python run_human_task.py --simulate                        # headless logic test
python run_recog_task.py -h                                # 5-way monster recognition task
python watch_ww.py --fake-env                              # watch a 5-way WW race trial by trial
```

### Use the models from Python
```python
import numpy as np
from sp2_nway import NWayAgent
from ww_npool import run

# full tier object: logits in, choice + RT + traces out
agent = NWayAgent("ww_cr_lca", params={"u0": 16, "gain": 0.05}, seed=0, n=5)
out = agent.decide(np.array([8, 0, 0, 0, 0.]), timeout_ms=3000)
print(out["choice"], out["rt_ms"], out["route"])      # route is "ww" or "lca"
# other keys: u0, coh, q, ww_r (rate trace), lca_x, t_ww, t_lca, dt, timeout

# bare Wong-Wang n-pool network
res = run(coh=[0.2, 0.0], u0=14.0, noise=0.02, t_max=3000)
print(res["choice"], res["t_ms"])                     # res["r"] = T x n rates (Hz), res["S"] = gating
```
Tiers: `ww`, `ww_cr`, `ww_cr_lca`. Defaults are in `sp2_nway.NW_PRESETS`; any key can be overridden via `params`.

---

## 6. Files

| file | purpose |
|---|---|
| `ww_npool.py` | n-pool Wong-Wang network |
| `sp2_nway.py` | the three tiers (5-way / n-way): roaming + LCA route |
| `sp2_agents.py` | trial-level 2-choice agents and shared constants |
| `play_nets.py` | **main demo**: MOVE + SHOOT nets playing `defend_the_center` |
| `play_ww.py` | ViZDoom env wrapper, enemy mod, older single-net player |
| `run_human_task.py`, `rdm_doom_env.py` | RDM-analogue task (human or model) |
| `run_recog_task.py`, `doom_recog_env.py`, `watch_ww.py` | 5-way recognition task |
| `proto_frontend.py` | prototype-matching evidence front end used by `watch_ww.py` |

## 7. Known limitations
* Presets are hand-tuned for the demo, not fitted to data.
* `--turn-gain` (degrees-per-tic of the pulse controller) is a guess; adjust if the agent over/undershoots.
* Pygame dashboard, the enemy mod, `--god` and `--infinite-ammo` have not been verified on real Doom by the author of this code.
* This is a demonstration of the model dynamics, not a good Doom-playing agent.

## References
* Ashwood, Z. C., et al. (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nature Neuroscience, 25*(2), 201-212.
* Brinkman, B. A. W., et al. (2022). Metastable dynamics of neural circuits and networks. *Applied Physics Reviews, 9*, 011313.
* Harrison, S. M., Chuong, O'Camb & Blair. *The Dynamics of Reaction Time Distributions: An Analysis of Cross-sectional and Longitudinal Telemetry Data from 25,364 Hours of Gameplay from the Strategy Video Game StarCraft 2* (in press, JEP: General). Data: doi:10.20383/103.01000.
* Kempka, M., Wydmuch, M., Runc, G., Toczek, J., & Jaśkowski, W. (2016). ViZDoom: A Doom-based AI research platform for visual reinforcement learning. *IEEE Conference on Computational Intelligence and Games*.
* Logan, G. D. (1988). Toward an instance theory of automatization. *Psychological Review, 95*(4), 492-527.
* Martí, D., Deco, G., Mattia, M., Gigante, G., & Del Giudice, P. (2008). A fluctuation-driven mechanism for slow decision processes in reverberant networks. *PLoS ONE, 3*(7), e2534.
* Pathak, A., & Battaglia, D. (2025). Critical roaming. *bioRxiv*, doi:10.64898/2025.12.29.696846.
* Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. *Psychological Review, 108*(3), 550-592.
* Wang, X.-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. *Neuron, 36*(5), 955-968.
* Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience, 26*(4), 1314-1328.
* Ye, W., & Li, Z. (2021). Physical mechanism of mind changes and tradeoffs among speed, accuracy and energy cost in brain decision making: landscape, flux and path perspectives. *Frontiers in Computational Neuroscience, 15*, 740601 (check exact citation in the project PDF).
