---
layout: page
title: Humanoid Localization & Mapping on Construction Sites
description: Unitree G1 in MuJoCo construction/energy sites. FMCW LiDAR Doppler + leg kinematics for localization; Doppler-aware mapping and MPPI navigation. Closed-loop open-site missions 0/3 → 3/3 vs geometry-only LIO.
img: assets/img/humanoid_g1/highlight.gif
importance: 2
category: work
links:
  - title: Code on GitHub
    url: https://github.com/enginksz/humanoid-g1-site-localization
tags:
  - Localization
  - Mapping
  - Humanoid
  - LiDAR
  - Sensor Fusion
  - Navigation
---

A **Unitree G1** walks simulated construction and energy sites using only its own sensors: a head-mounted **FMCW LiDAR** with per-point Doppler, an IMU, and joint encoders. The question is not "can the policy walk" — that policy is Unitree's pretrained one — but whether the robot **knows where it is**, whether its **map is safe to plan on**, and whether localization errors turn into map errors that make it stop or fall.

This page is a portfolio summary. Full numbers, parameters, and failure notes are in the [GitHub README](https://github.com/enginksz/humanoid-g1-site-localization) and the technical report in that repo.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/humanoid_g1/highlight.gif" title="Map + MPPI navigation demo" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Left: chase camera. Right: the robot's own map (LiDAR + Doppler), tracked people, MPPI samples and the selected trajectory. Reactive steering gets trapped in a U-obstacle; this stack completes the mission.*

## Stack

| Stage | Approach |
|---|---|
| Localization | FMCW-LIO with Doppler, plus **leg kinematics** and a **Doppler–leg χ² slip gate** |
| Baselines | FAST-LIO2 (geometry only), FMCW-LIO without the leg / slip additions |
| Mapping | Online obstacle map: Doppler moving-object removal + guarded raycast + short "recent observations" layer for collision |
| Navigation | Dijkstra navigation function (global) + **MPPI** on body velocity commands; people predicted over ~2.5 s |
| Platform | MuJoCo scenes (building floor, corridor, roof, open energy site, nav clutter); ROS-free estimator ports |

What was written here vs. reused: scenes, sensor models, closed loop, online map, MPPI navigator, leg odometry / slip gate patches, and all evaluation. Walking policy and algorithm cores of FMCW-LIO / FAST-LIO2 are upstream.

## Localization

Nine scenes × three seeds. Geometry-only LIO is strong where there is structure and fails on open, repetitive, or busy sites. Doppler + legs close most of that gap.

| ATE [m] (lower better) | building floor | workers & machines | long corridor | empty roof | open site |
|---|---|---|---|---|---|
| FAST-LIO2 | 0.030 | 0.694 | 0.298 | 5.86 | 0.660 |
| FMCW-LIO + Doppler | 0.030 | 0.035 | 0.147 | 3.25 | 0.039 |
| **+ legs + Doppler slip gate** | **0.027** | **0.033** | **0.066** | **0.72** | **0.031** |

- Slip gate (Doppler vs leg velocity): **88–97 %** precision, **79–84 %** recall on loose roof sheets.
- Root cause of the classic **6–8 m** open-site drift: a humanoid is never stationary. A short "standing still" IMU init while it steps in place leaves ~1.5° gravity error → roughly \(½ · 0.25\,\mathrm{m/s²} · (8\,\mathrm{s})² ≈ 8\,\mathrm{m}\).

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/humanoid_g1/summary_ate.png" title="ATE summary" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## Mapping for safe motion

The obstacle map is built from the robot's own pose and own LiDAR. Two failure modes dominate on a busy site: **pose error** (shifted / doubled structure) and **moving objects** (ghost walls from people and machines).

| Obstacle-map F1 (0.3 m tolerance vs GT) | building + workers / machines | open site + trucks / workers |
|---|---|---|
| FAST-LIO2 pose + raycast clearing | 0.24 | 0.19 |
| Perfect pose, no moving-object handling | 0.58 | 0.57 |
| **4D + legs pose, Doppler + guarded raycast** | **0.98** | **0.93** |

Doppler removes movers in the same scan (point precision **0.93–0.95**, recall **0.83–0.88**). Radial-only Doppler misses cross-beam motion, so the map keeps a short recent-observation layer: **people leave the static map but stay in collision checking.**

<div class="row justify-content-sm-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/humanoid_g1/elevmap_dynamic_stack.png" title="Map comparison on dynamic site" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

*Dark: obstacle within 0.3 m of truth. Red: ghost. Blue: missed.*

## Navigation

| 5 seeds (U-trap, 1 m gap, pallets, people) | success | robot-caused person contacts |
|---|---|---|
| Reactive steering on a ground-truth costmap | 1 / 5 | 0 |
| Own LiDAR map + MPPI, raycast only | 5 / 5 | 4 (closest 0.12 m) |
| **Own map + MPPI, Doppler + guarded raycast** | **5 / 5** | **0** (closest 0.52 m) |

Raycast-only clearing can erase people from both the map and awareness. With Doppler they leave the map but remain tracked and predicted.

<div class="row justify-content-sm-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/humanoid_g1/nav_bench_traj.png" title="Navigation benchmark trajectories" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## Closed loop on the open site

The robot walks on **its own** pose estimate and builds **its own** map. On a feature-rich building floor both stacks complete 5/5. On the open site the difference is the whole story:

| open site, 3 seeds | completed | falls | max pose error |
|---|---|---|---|
| Geometry-only LIO, no moving-object handling | **0 / 3** | 1 | 3.1 m |
| **Doppler + legs, Doppler + guarded raycast** | **3 / 3** | 0 | **0.20 m** |

Start-up gravity error → 2–3 m pose error → shifted structure and phantom walls in the map → stop or fall.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/humanoid_g1/outdoor_phantom_wall.png" title="Phantom wall from gravity error" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

*Geometry-only odometry on the open site: gravity error lifts far ground into the body-height band, so the robot's map invents walls.*

## Honest scope

- **Simulation only** (MuJoCo). Absolute metres are optimistic; rankings and failure mechanisms are the claim.
- Walking policy is Unitree's pretrained policy — not trained in this project.
- FMCW model is geometric (true radial velocity + noise); real sensors are noisier.
- Navigator commands body velocities; footstep planning is out of scope. Simulated people never yield.
- Code was developed with an AI coding assistant; design, experiments, and analysis were directed by the author.

## Links

- [GitHub repository](https://github.com/enginksz/humanoid-g1-site-localization) — code, reproduce scripts, report, demo video
