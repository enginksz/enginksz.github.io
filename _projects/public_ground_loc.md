---
layout: page
title: Ground Vehicle Localization Under GNSS Dropout (LiDAR–IMU + GNSS)
description: LiDAR–inertial odometry with gated GNSS fusion on MulRan and UrbanNav. Frame alignment, innovation gating, continuous odometry through outages, and drift / reacquisition metrics on open data.
img: assets/img/liosam.jpg
importance: 3
category: work
links:
  - title: MulRan
    url: https://sites.google.com/view/mulran-pr/dataset
  - title: UrbanNav
    url: https://github.com/IPNL-POLYU/UrbanNavDataset
tags:
  - Localization
  - SLAM
  - Sensor Fusion
  - GNSS-Denied
  - ROS 2
---

**Problem.** A ground vehicle needs a continuous pose, velocity and heading for its planner and controller. GNSS gives an absolute position but disappears in tunnels, degrades in urban canyons (multipath / NLOS), and can be jammed. LiDAR–inertial odometry (LIO) gives smooth, locally accurate motion but drifts without an absolute reference. The engineering question is not "which is better" but **when the filter may trust GNSS, and how to hand over between the two without disturbing the vehicle**.

This page studies that question on two public datasets so every number is reproducible. It uses no employer data, maps or software.

## Datasets

| | [MulRan](https://sites.google.com/view/mulran-pr/dataset) | [UrbanNav (Hong Kong)](https://github.com/IPNL-POLYU/UrbanNavDataset) |
|---|---|---|
| LiDAR | Ouster OS1-64 | Velodyne HDL-32E (+ others) |
| IMU | Xsens MTi-300 | Xsens MTi-10 |
| GNSS | u-blox EVK-7P (single-point) | u-blox receivers, raw + fix |
| Ground truth | 6-DoF baseline trajectory (rear-axle centre) | NovAtel SPAN-CPT (RTK GNSS/INS) |
| Sequences used | KAIST 01, DCC 01, Riverside 01 | HK-Medium-Urban-1, HK-Deep-Urban-1, HK-Tunnel-1 |
| Role here | Controlled study: **synthetic** GNSS outages on clean urban drives | **Real** degradation: canyon multipath and a 3.15 km tunnel run |

MulRan's GNSS is a consumer single-point receiver (metre-level), so it is a realistic test of covariance honesty: an RTK-grade $$\sigma$$ on that receiver would pull the estimate off a good LiDAR map.

## Architecture

```
  LiDAR (10 Hz)            IMU (100+ Hz)               GNSS fix (1–10 Hz)
       │                        │                             │
  deskew with IMU          preintegration                lever arm,
       │                   ΔR, Δv, Δp                    σ from receiver
       ▼                        ▼                             │
 ┌──────────────────────────────────────────────┐             │
 │ LiDAR–inertial odometry  (LIO-SAM / FAST-LIO2)│             │
 │ scan-to-map + IMU factors, local map          │             │
 │ degeneracy check on scan-matching Hessian     │             │
 └───────────────────────┬──────────────────────┘             │
                         │ odom → base_link  (continuous)     │
                         ▼                                    ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Global fusion (error-state EKF on SE(2) + z)                 │
 │ predict: LIO increment          update: GNSS position        │
 │ χ² innovation gate · adaptive R · outage / reacquire logic   │
 └───────────────────────┬──────────────────────────────────────┘
                         │ map → odom  (may correct, rate-limited)
                         ▼
        /odometry/local (smooth)     /odometry/global (absolute)
```

Two layers on purpose, following the ROS REP-105 split:

- **`odom → base_link`** comes from LIO only. It is continuous and never jumps, so the controller always has a smooth reference.
- **`map → odom`** carries the absolute correction from GNSS. When GNSS is lost this transform simply stops changing; when it returns, the correction is absorbed here — not injected as a step into the control loop.

The baseline it is compared against is **graph-coupled** fusion (GNSS as a unary factor inside the LIO factor graph, as in LIO-SAM's GPS factor). The loosely-coupled design gives up some optimality in exchange for an explicit place to gate, rate-limit and log GNSS decisions.

## Fusion layer

### Frame alignment

LIO runs in its own map frame; GNSS is in ENU. Before any update, the 4-DoF transform (translation + yaw; roll/pitch are observable from gravity) between them is estimated from the first stretch of good-quality fixes with a closed-form Umeyama fit, and only accepted once the trajectory is long enough and not straight-line (yaw is unobservable otherwise). The GNSS antenna lever arm $$\mathbf{l}$$ in the body frame is applied in the measurement model; skipping it shows up as a heading-dependent position bias.

### State and prediction

$$
\mathbf{x} = \begin{bmatrix} p_E & p_N & p_U & \psi \end{bmatrix}^\top
$$

Prediction uses the LIO relative increment $$(\Delta \mathbf{p}_k, \Delta\psi_k)$$ between consecutive odometry outputs, expressed in the body frame:

$$
\mathbf{p}_k = \mathbf{p}_{k-1} + \mathbf{R}(\psi_{k-1})\,\Delta\mathbf{p}_k, \qquad \psi_k = \psi_{k-1} + \Delta\psi_k
$$

Process noise $$\mathbf{Q}_k$$ scales with distance travelled and is inflated when the LIO degeneracy check fires (small eigenvalue of the scan-matching information matrix — tunnels and long corridors are the classic case). LIO is used **only** as a relative input, so the same information is never counted twice as both prediction and measurement.

### GNSS update and gating

$$
\mathbf{z}_k = \mathbf{p}_k + \mathbf{R}(\psi_k)\,\mathbf{l} + \mathbf{v}_k, \qquad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{\text{gnss}})
$$

- $$\mathbf{R}_{\text{gnss}}$$ comes from the receiver-reported covariance, with a floor per fix type (single-point, DGNSS, RTK float / fixed) and an inflation factor for few satellites or high DOP.
- Each fix is tested with the normalized innovation squared $$d^2 = \boldsymbol{\nu}^\top \mathbf{S}^{-1} \boldsymbol{\nu}$$ against a $$\chi^2$$ threshold (horizontal, 2 DoF: 9.21 at 99 %). Rejected fixes are logged, not silently dropped.
- Several consecutive rejects while the filter is confident trigger a **re-check** instead of permanent rejection, so a genuinely wrong filter state cannot lock GNSS out forever.

### Outage and reacquisition

| Mode | Entry condition | Behaviour |
|---|---|---|
| **Fused** | GNSS valid and passing the gate | Absolute updates; covariance stays bounded |
| **Dead-reckoning** | No fix, or fixes rejected for > $$T_{\text{out}}$$ | LIO-only prediction; covariance grows; downstream gets a `degraded` flag |
| **Reacquire** | Fixes return and pass a *wider* gate for $$N$$ consecutive epochs | Correction absorbed into `map → odom`, rate-limited (m/s, deg/s), then back to Fused |

A single good fix after a long tunnel is not trusted on its own: the first fixes out of a tunnel are often multipath-corrupted, and that is exactly where an ungated filter jumps several metres.

## Evaluation protocol

All trajectories are evaluated with [`evo`](https://github.com/MichaelGrupp/evo) against the dataset ground truth.

| Metric | Definition | Why |
|---|---|---|
| **ATE (aligned)** | RMSE after SE(3) Umeyama alignment | LIO-only accuracy; its frame is arbitrary |
| **ATE (ENU, unaligned)** | RMSE in the GNSS frame, no alignment | GNSS-fused accuracy; alignment would hide global error |
| **RPE / drift** | Translational error per 100 m segment, % of distance | Odometry quality independent of start point |
| **Outage drift** | Horizontal error at end of denial vs. denied distance | Core question: how long can the vehicle go without GNSS |
| **Reacquisition** | Largest pose step in `/odometry/global`, settling time | Is handover safe for the controller |
| **Consistency** | Fraction of NIS / NEES inside the 95 % bounds | Is the reported covariance honest |
| **Gate statistics** | Reject rate per sequence, false accept / reject vs. GT error | Does the gate remove the bad fixes and only those |

### Experiment matrix

| ID | Data | Configuration | Question |
|---|---|---|---|
| E1 | MulRan KAIST 01, DCC 01 | LIO only | Baseline drift of the odometry front end |
| E2 | same | LIO + GNSS, graph-coupled (GPS factor) | Reference fusion |
| E3 | same | LIO + GNSS, gated EKF | Does gating change accuracy on clean data (it should not) |
| E4 | same | E3 with GNSS removed for 60 / 120 / 300 s windows | Drift vs. outage length |
| E5 | UrbanNav Medium / Deep-Urban | E2 vs. E3, gate on / off | Multipath rejection in real canyons |
| E6 | UrbanNav HK-Tunnel-1 | E3, rate limit on / off | Real denial and post-tunnel reacquisition step |
| E7 | any | Lever arm or LiDAR–IMU time offset deliberately perturbed | Sensitivity to calibration errors |

## Failure modes this setup is designed to expose

1. **Calibration before algorithms.** A small LiDAR–IMU time offset or yaw error appears as along-track walk and is easily misread as "SLAM divergence". E7 quantifies it.
2. **Geometric degeneracy.** Tunnels and long straight corridors leave scan matching unconstrained along the direction of travel; the IMU and inflated $$\mathbf{Q}$$ must carry the estimate there.
3. **Over-confident GNSS.** Consumer single-point fixes configured with RTK-level covariance pull the filter off a correct LiDAR map in NLOS conditions.
4. **Unprotected reacquisition.** A multi-metre step into the controller right after a tunnel exit.
5. **Frame errors.** A wrong ENU ↔ map yaw looks like a growing position error that scales with distance from the origin.

## Reproducibility

Each run records the LIO configuration, extrinsics file, lever arm, $$\mathbf{Q}$$ / $$\mathbf{R}$$ settings, gate thresholds and outage windows alongside the `evo` outputs, so every run can be regenerated from the public logs.

## Related

- [GNSS-denied visual localization for UAVs (SIU 2026)](/projects/gnss_denied_localization/) — the aerial counterpart: absolute re-anchoring from a camera and a map.
- [Public-data UAV visual geo-localization](/projects/public_uav_geoloc/) — the same idea on open benchmarks.
- [LIO-SAM Gazebo ROS 2](/projects/LIO-SAM%20ROS2/) — earlier simulation bring-up of the LIO front end.
