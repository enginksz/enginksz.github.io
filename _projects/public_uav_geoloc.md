---
layout: page
title: UAV Visual Geo-Localization on Public Benchmarks (OrthoLoC + AerialVL)
description: Absolute UAV pose from camera-to-orthophoto registration on OrthoLoC and AerialVL. Prior-guided map warping, SuperPoint–LightGlue, DSM-lifted EPnP + RANSAC, and a gated track filter. The open-data counterpart of the SIU 2026 field system.
img: assets/img/navwogps/eb94e55c-d80c-4118-bb8f-171e4417d1ef.png
importance: 2
category: work
giscus_comments: true
links:
  - title: OrthoLoC
    url: https://deepscenario.github.io/OrthoLoC/
  - title: AerialVL
    url: https://github.com/hmf21/AerialVL
  - title: SIU 2026 (field)
    url: /projects/gnss_denied_localization/
tags:
  - Computer Vision
  - Localization
  - UAV Navigation
  - GNSS-Denied
---

**Problem.** Visual and visual–inertial odometry drift without bound. A UAV that must fly without GNSS needs an **absolute** reference, and the most widely available one is a georeferenced orthophoto or satellite image with an elevation model. The task: given a camera frame, register it to that map and recover the camera pose in world coordinates — robustly across seasons, altitudes and low-texture terrain.

The [SIU 2026 system](/projects/gnss_denied_localization/) solves this on our own field flights (10 m MAE / 14 m RMSE in X–Y at ~500 m AGL vs. RTK, ~5 Hz on Jetson Orin Nano). Those flights are not public. This page runs the same estimator on two open benchmarks so the method can be measured, compared and reproduced by anyone.

## Benchmarks

| | [OrthoLoC](https://deepscenario.github.io/OrthoLoC/) (NeurIPS 2025) | [AerialVL](https://github.com/hmf21/AerialVL) (RA-L 2024) |
|---|---|---|
| Query data | 16,425 UAV images, 47 locations, 19 regions (DE / US) | 11 flight sequences, 3.7–11 km each, ~70 km total; 18,361 frames |
| Map data | Orthophoto (DOP) + DSM per sample, intrinsics given | 14,096 map patches + raw satellite tiles (Google Earth, USGS) |
| Ground truth | 6-DoF camera pose | Single-point GNSS (NovAtel OEM718D, ~1.5 m RMS) |
| Splits / tasks | `train`, `val`, `test_inPlace` (seen locations), `test_outPlace` (unseen) | Visual place recognition (VPR) and sequential visual alignment (VAL) |
| What it tests here | **Single-frame 6-DoF registration**, domain shift | **Long-trajectory tracking**: re-anchoring, track loss and recovery |

Two consequences for the protocol:

- OrthoLoC pairs each query with its geodata crop, which isolates the **registration** stage from retrieval. Pose errors there are about matching and geometry, not about finding the right tile.
- AerialVL's ground truth is itself metre-level (single-point GNSS), so errors below ~2 m are not resolvable on it. It is used for track-level behaviour, with error thresholds of 5 m and above.

## Pipeline

```
   prior pose P̂(k−1), cov Σ        intrinsics K, altitude / attitude
                 │                               │
                 ▼                               ▼
   ┌──────────────────────────────────────────────────────┐
   │ 1. Map synthesis                                      │
   │    crop DOP/satellite around prior (radius from Σ)    │
   │    warp with homography H(alt, roll, pitch, yaw)      │──► I_map
   └───────────────────────────┬──────────────────────────┘
   I_cam ── CLAHE ──┐          │
                    ▼          ▼
   ┌──────────────────────────────────────────────────────┐
   │ 2. Matching   SuperPoint → LightGlue                  │
   │               (SIFT / ORB / LoFTR as baselines)       │──► (u,v)_cam ↔ (u,v)_map
   └───────────────────────────┬──────────────────────────┘
                               ▼  lift map pixels with georeference + DSM
   ┌──────────────────────────────────────────────────────┐
   │ 3. Pose       EPnP + RANSAC → Levenberg–Marquardt      │──► T_wc, inliers, reproj. error
   └───────────────────────────┬──────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────┐
   │ 4. Track filter   constant-velocity KF + χ² gate       │
   │    rate limit · optical-flow + Procrustes fallback    │──► P̂(k) → next crop
   └──────────────────────────────────────────────────────┘
```

On OrthoLoC (single frames, paired geodata) stages 1–3 run once per query. On AerialVL all four stages run sequentially, and the filter output defines the next map crop.

## Geometry

**Map warping.** A nadir orthophoto and an oblique, rotated, scaled camera frame differ by a large projective transform. Asking the matcher to absorb all of it in one shot costs inliers. Using the prior attitude and altitude, the map crop is warped by a homography $$\mathbf{H}$$ into an approximate camera view, so the matcher only has to resolve the residual. On the field data, switching $$\mathbf{H}$$ off raised X–Y MAE from **10.2 m to 23.1 m** ([SIU ablation](/projects/gnss_denied_localization/)). OrthoLoC's own refinement method, AdHoP, applies a related homography-based idea; comparing the two is one of the planned ablations.

**2D–3D correspondences.** Each matched map pixel $$(u_m, v_m)$$ is undone through $$\mathbf{H}^{-1}$$, mapped to world easting / northing with the geotransform, and lifted with the elevation model:

$$
\mathbf{X}_i = \begin{bmatrix} E(u_m, v_m) & N(u_m, v_m) & \mathrm{DSM}(u_m, v_m) \end{bmatrix}^\top
$$

**Pose.** With camera pixels $$\mathbf{u}_i$$, the pose minimizes reprojection error

$$
\mathbf{T}_{wc}^{*} = \arg\min_{\mathbf{R},\,\mathbf{t}} \sum_{i \in \mathcal{I}} \rho\!\left( \left\lVert \mathbf{u}_i - \pi\!\left(\mathbf{K}(\mathbf{R}\mathbf{X}_i + \mathbf{t})\right) \right\rVert^2 \right)
$$

initialized by EPnP inside RANSAC and refined with Levenberg–Marquardt on the inlier set $$\mathcal{I}$$, with a robust loss $$\rho$$. A solution is accepted only if it has enough inliers and a plausible altitude; otherwise the frame is marked as a failed registration rather than reported as a bad pose.

**Where the errors come from.** Horizontal position is constrained by many well-spread ground features. Height and the coupled tilt are weaker: they depend on DSM resolution and on the small depth variation seen from altitude. Results are therefore reported for X–Y and Z separately.

**Operating point (field system).** SuperPoint 800–1200 keypoints, LightGlue 80–200 matches, 15–40 RANSAC inliers used by EPnP. The same quantities are logged per frame on the benchmarks.

## Track filter (AerialVL)

AerialVL provides no IMU stream, so the filter is a constant-velocity model on horizontal position, height and yaw. Each registration result is tested with the normalized innovation squared against a $$\chi^2$$ threshold before it is accepted; a large but consistent jump over several frames widens the search window instead of being ignored indefinitely. When matching collapses (water, uniform fields), Lucas–Kanade optical flow with a Procrustes fit propagates the pose for a bounded number of frames; after that the track is declared lost and the next crop radius grows with the covariance.

## Evaluation protocol

| Metric | OrthoLoC | AerialVL |
|---|---|---|
| Translation error (m) | median and mean, per split | X–Y error vs. GNSS GT: MAE, RMSE |
| Rotation error (°) | median and mean | yaw only |
| Recall @ threshold | % of queries within (1 m, 1°), (3 m, 3°), (5 m, 5°) | % of frames within 5 / 10 / 20 m |
| Registration rate | % of queries with an accepted pose | % of frames with absolute registration (vs. filter-only) |
| Robustness | `inPlace` → `outPlace` gap | track losses per km, time to recover |
| Cost | matches, inliers, runtime per frame (desktop GPU and Jetson) | same |

Official OrthoLoC numbers are produced with the benchmark's own evaluation script; the recall thresholds above are reported in addition.

### Ablations

| ID | Change | Question |
|---|---|---|
| A1 | Matcher: SuperPoint–LightGlue vs. SIFT, ORB, LoFTR | Accuracy / runtime trade-off on open data |
| A2 | Homography warp off / on / AdHoP | How much does view synthesis buy, and is it matcher-independent |
| A3 | DSM vs. flat ground plane at mean height | Cost of ignoring terrain relief |
| A4 | Track filter and fallback off (AerialVL) | Contribution of temporal filtering to track continuity |
| A5 | `test_inPlace` vs. `test_outPlace` | Generalization to unseen locations |

## Reference figures from the field system

The figures below are from the SIU 2026 field flights, not from OrthoLoC or AerialVL. They show what the matching and tracking stages produce on real UAV-to-map data.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-34-40.png" title="Camera–map matches" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

*SuperPoint–LightGlue correspondences between a UAV frame and the warped satellite tile.*

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-45-19.png" title="Estimated vs RTK track" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

*Estimated track vs. RTK on the satellite map (field flight, ~500 m AGL: 10 m MAE / 14 m RMSE in X–Y).*

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/navwogps/09_ekim_figure_1.png" title="NED traces" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

*North, East and Down components vs. RTK. Horizontal axes follow the reference closely; height is the weaker axis.*

## Expected failure modes

1. **Appearance change.** Season, construction or a different capture year between query and map; the main driver of the `inPlace` → `outPlace` gap.
2. **Low texture.** Farmland, water, snow: too few repeatable keypoints. The fallback only delays drift; the track must re-anchor once texture returns.
3. **Wrong prior window.** A crop that does not contain the true footprint produces confident but wrong matches; the gate and the covariance-driven crop radius exist for this case.
4. **Elevation error.** A coarse or outdated DSM biases height and tilt first, then horizontal position at oblique views.
5. **Repetitive structure.** Field rows and building blocks create aliased matches that RANSAC can accept as a consistent wrong pose.

## Reproducibility

Matcher weights are frozen across all splits, and every run stores its configuration (keypoint budget, RANSAC threshold, inlier minimum, gate threshold) next to per-frame logs of matches, inliers, reprojection error and pose error.

## Related

- [GNSS-denied visual localization for UAVs (SIU 2026)](/projects/gnss_denied_localization/) — the field system and its published results.
- [Ground vehicle localization under GNSS dropout](/projects/public_ground_loc/) — the ground counterpart: LiDAR–inertial odometry with gated GNSS.
