---
layout: page
title: GNSS-Denied Visual Localization for UAVs using Satellite Imagery
description: SIU 2026. SuperPoint–LightGlue matching of UAV camera frames to satellite tiles; 10 m MAE / 14 m RMSE (X–Y) at ~500 m vs RTK, ~5 Hz on Jetson Orin Nano.
img: assets/img/navwogps/eb94e55c-d80c-4118-bb8f-171e4417d1ef.png
importance: 1
category: work
giscus_comments: true
links:
  - title: IEEE Xplore
    url: https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Deep+Visual+Localization+for+UAVs+in+GNSS-Denied+Environments+Using+Satellite+Imagery%22
tags:
  - Computer Vision
  - Deep Learning
  - Robotics
  - Localization
  - UAV Navigation
---

**[SIU 2026](https://siu2026.pirireis.edu.tr/)** — *Deep Visual Localization for UAVs in GNSS-Denied Environments Using Satellite Imagery* (Engin Öksüz, Barış Yalçın, Tolga Demirdal). Proceedings are submitted to IEEE Xplore; the button above is the Xplore search for this title.

Absolute position from a downward/oblique UAV camera matched to open satellite tiles. No GNSS in the estimator. Relative methods (VO / VIO / SLAM) drift; this pipeline re-anchors to a georeferenced map.

## Pipeline

Four stages, as in the paper:

1. **Map synthesis** — previous pose \(P\) and camera \(K\) select satellite tiles; a homography \(H\) warps the map toward the camera (altitude, roll, pitch, yaw).
2. **Matching** — CLAHE, then SuperPoint + LightGlue. SIFT and ORB are baselines on the same backend, not alternate front-ends in the deployed system.
3. **Pose** — EPnP + RANSAC on 2D–3D correspondences. Typical operating point: SuperPoint 800–1200 keypoints, LightGlue 80–200 inliers, EPnP uses 15–40.
4. **Fusion** — linear Kalman filter. When matches collapse (low texture), Lucas–Kanade optical flow + Procrustes is the backup. A rate limiter bounds jump updates.

Deployed with TensorRT on a **Jetson Orin Nano** at about **5 Hz** (~200 ms end-to-end). That is the number in the paper — not 10–30 Hz, not AGX.

## Field results (RTK truth)

Rural, low-texture flights: about **20 sorties / 12 hours**, **50–900 m** AGL. Headline accuracy is the **~500 m** band, **X–Y only**:

| | MAE | RMSE |
|---|-----|------|
| Proposed (SP + LightGlue) | **10 m** | **14 m** |

Ablation on a representative segment (X–Y MAE vs RTK):

| Configuration | MAE (m) |
|---|---|
| Full system | 10.2 |
| Homography off | 23.1 |
| Optical-flow backup off | 12.6 |

Matching on the same protocol:

| Front-end | mAP | Time (s) | Matches |
|---|---|---|---|
| SIFT | 0.650 | 0.043 | 24 |
| ORB | 0.717 | 0.036 | 78 |
| SuperPoint + LightGlue | **0.912** | 0.340 | 249 |

North and East tracks follow RTK. **Z is worse** — DEM resolution and monocular depth, not a solved axis. Homogeneous farmland still drops keypoints and produces short error spikes; Kalman + optical flow damp them, they do not disappear.

## What this is not

- Not 4.8 m mean error. That figure was on an older draft of this page and is not in the paper.
- Not a claim of consumer-GPS parity, indoor flight, urban air mobility, or night-as-primary.
- Not a full 12-state EKF with camera intrinsics in the state. The published filter is a linear Kalman on the pose stream.
- Employer name and platform internals are not discussed here.

## Figures

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-34-40.png" title="Camera–satellite matches" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*SuperPoint–LightGlue correspondences between the camera frame and the warped satellite tile.*

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-45-19.png" title="Estimated vs RTK track" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Estimated track on the satellite map vs RTK. Paper headline: 10 m MAE / 14 m RMSE in X–Y near 500 m.*

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/09_ekim_figure_1.png" title="NED traces" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*NED traces. Horizontal axes track RTK more closely than height.*

## Next (paper)

Other terrain and seasons, better Z, and a LoFTR (or similar) comparison. Not on this page until measured.
