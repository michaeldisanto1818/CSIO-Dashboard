"""
convert_predictions.py
=======================

Bridges the video->biomechanics prediction pipeline (whoever runs that model)
to the web dashboard (index.html).

WHY THIS EXISTS
----------------
The dashboard is a static HTML/JS page with no backend. It can't unpickle a
numpy .npz that contains a pickled Python dict (`np.load(..., allow_pickle=True)
['data'].item()`), because that requires running Python. This script does that
conversion ONE TIME, offline, and produces plain files a browser can read
directly:

    <vid_name>.json        <- time-series data (angles/torques/velos/accels/GRF/COP)
    <vid_name>_verts.bin   <- SMPL mesh vertices, float32, flat [T*6890*3] (optional)

Give this script's OUTPUT to the dashboard's "Upload Session" panel (or drop
the files next to index.html and register the session), and it will plug in
using the exact same code path as the built-in demo sessions.

USAGE
-----
    python convert_predictions.py path/to/s10_fencing10_Lungetwo_1_pred_mp.npz --outdir ./converted

Run from anywhere -- it only needs numpy. It does NOT need the rest of the
`fencing/` repo, pyrender, trimesh, etc. (the DOF list is inlined below so
this script has zero project-specific dependencies).

INPUT CONTRACT (what the model/pipeline needs to produce)
-----------------------------------------------------------
A .npz with a single pickled entry 'data' (a dict) OR (preferred, simpler)
a plain .npz of arrays, containing:

    fps         : int                      frames per second
    qfull       : float array [T, 57]      generalized coords (3 root trans + 54 rotational DOF, radians)
    tau_inv     : float array [T, 57]      inverse-dynamics joint torques (Nm), same layout as qfull
    GRF         : float array [T, 2, 3]    ground reaction force, [left, right] x [X,Y,Z] (N)
    COP         : float array [T, 2, 3]    center of pressure, [left, right] x [X,Y,Z] (m)
    bodymass    : float                    kg
    height      : float                    m (optional)
    vertices    : float array [T, 6890, 3] SMPL mesh vertices (optional -- enables 3D viewer)

    vid_name is taken from the filename (strip "_pred_mp.npz"); subject/action/
    trial are parsed from vid_name as "s{subject}_fencing{n}_{action}[_{trial}]".
    If your naming differs, pass --subject/--action/--trial explicitly.
"""

import argparse
import json
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# DOF ordering -- must match lib/utils/human_constants.py:MP_ROTDOFS exactly.
# qfull/tau_inv columns 0:3 are root translation (unused by the dashboard);
# columns 3:57 map 1:1 onto this 54-name list.
# ---------------------------------------------------------------------------
MP_ROTDOFS = [
    "root_RX_TiltRight", "root_RY_AxialLeft", "root_RZ_Posterior",
    "LeftHip_Abduction", "LeftHip_ExRotation", "LeftHip_Flexion",
    "RightHip_Adduction", "RightHip_InRotation", "RightHip_Flexion",
    "LumbarJoint_BendRight", "LumbarJoint_AxialLeft", "LumbarJoint_Extension",
    "LeftKnee_Abduction_unused", "LeftKnee_ExRotation_unused", "LeftKnee_Extension",
    "RightKnee_Adduction_unused", "RightKnee_InRotation_unused", "RightKnee_Extension",
    "ThoracicJoint_BendRight", "ThoracicJoint_AxialLeft", "ThoracicJoint_Extension",
    "LeftAnkle_Inversion", "LeftAnkle_Abduction", "LeftAnkle_Dorsiflexion",
    "RightAnkle_Eversion", "RightAnkle_Adduction", "RightAnkle_Dorsiflexion",
    "LeftCollar_Abduction", "LeftCollar_ExRotation", "LeftCollar_Flexion_unused",
    "RightCollar_Adduction", "RightCollar_InRotation", "RightCollar_Flexion_unused",
    "Neck_BendRight", "Neck_AxialLeft", "Neck_Flexion",
    "LeftShoulder_Abduction", "LeftShoulder_ExRotation", "LeftShoulder_Flexion",
    "RightShoulder_Adduction", "RightShoulder_InRotation", "RightShoulder_Flexion",
    "LeftElbow_Valgus_unused", "LeftElbow_Supination", "LeftElbow_Flexion",
    "RightElbow_Varus_unused", "RightElbow_Pronation", "RightElbow_Flexion",
    "LeftWrist_RadialDev", "LeftWrist_Supination_unused", "LeftWrist_Flexion",
    "RightWrist_UlnarDev", "RightWrist_Pronation_unused", "RightWrist_Flexion",
]
# qfull/tau_inv column index for a given DOF name (root DOFs occupy 0:3)
ROTDOF2IDX = {name: i + 3 for i, name in enumerate(MP_ROTDOFS)}

VID_NAME_RE = re.compile(r"^s(?P<subject>\d+)_fencing\d+_(?P<action>[A-Za-z]+)(?:_(?P<trial>\d+))?$")


def parse_vid_name(vid_name):
    m = VID_NAME_RE.match(vid_name)
    if not m:
        return None, None, "1"
    return f"S{m.group('subject')}", m.group("action"), m.group("trial") or "1"


def load_pred_dict(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    if "data" in npz and npz["data"].dtype == object:
        # pickled-dict style (matches demo_gif.py / interactive_dashboard.py)
        return npz["data"].item()
    # plain-arrays style
    return {k: npz[k] for k in npz.files}


def detect_foot_segments(grf, fps, bw_threshold_frac=0.05):
    """
    Auto-detect foot lift-off -> contact intervals ("foot in the air") from
    vertical GRF crossing a low threshold. Mirrors the hand-checked
    FENCING_FEET_SEGMENTS already hardcoded in index.html for the 3 demo
    clips -- this is an automated approximation for NEW sessions and should
    be spot-checked, not trusted blindly.
    """
    grf = np.asarray(grf)
    T = grf.shape[0]
    bw = max(np.nanmax(grf[:, :, 1]), 1.0)  # rough scale if bodymass unknown
    thresh = bw_threshold_frac * bw
    segments = {"R": [], "L": []}
    for side_idx, side in [(0, "L"), (1, "R")]:
        in_air = grf[:, side_idx, 1] < thresh
        start = None
        for i in range(T):
            if in_air[i] and start is None:
                start = i
            elif not in_air[i] and start is not None:
                if i - start > 1:  # ignore single-frame noise
                    segments[side].append([start + 1, i + 1])  # 1-based, matches dashboard convention
                start = None
        if start is not None and T - start > 1:
            segments[side].append([start + 1, T])
    return segments


def compute_derivatives(angles_deg, fps):
    dt = 1.0 / fps
    velos = np.gradient(angles_deg, dt, axis=0)
    accels = np.gradient(velos, dt, axis=0)
    return velos, accels


def convert(npz_path, outdir, subject=None, action=None, trial=None):
    os.makedirs(outdir, exist_ok=True)
    vid_name = os.path.basename(npz_path).replace("_pred_mp.npz", "").replace(".npz", "")
    data = load_pred_dict(npz_path)

    fps = int(data["fps"])
    qfull = np.asarray(data["qfull"])          # [T, 57] radians
    tau_inv = np.asarray(data["tau_inv"])      # [T, 57] Nm
    grf = np.asarray(data["GRF"])              # [T, 2, 3]
    cop = np.asarray(data["COP"])              # [T, 2, 3]
    bodymass = float(data.get("bodymass", 0) or 0)
    height = float(data.get("height", 0) or 0)
    T = qfull.shape[0]
    time = (np.arange(T) / fps).tolist()

    angles_deg = np.stack(
        [qfull[:, ROTDOF2IDX[name]] * 180.0 / np.pi for name in MP_ROTDOFS], axis=1
    )  # [T, 54]
    torques = np.stack(
        [tau_inv[:, ROTDOF2IDX[name]] for name in MP_ROTDOFS], axis=1
    )  # [T, 54]
    velos_deg, accels_deg = compute_derivatives(angles_deg, fps)

    def dof_dict(arr2d):
        return {name: arr2d[:, i].tolist() for i, name in enumerate(MP_ROTDOFS)}

    subj, act, tri = parse_vid_name(vid_name)
    subject = subject or subj or "UNKNOWN"
    action = action or act or vid_name
    trial = trial or tri or "1"

    session = {
        "fps": fps,
        "T": T,
        "time": time,
        "vid_name": vid_name,
        "subject": subject,
        "action": action,
        "trial": trial,
        "bodymass": bodymass,
        "height": height,
        "dof_names": MP_ROTDOFS,
        "angles": dof_dict(angles_deg),
        "torques": dof_dict(torques),
        "velos": dof_dict(velos_deg),
        "accels": dof_dict(accels_deg),
        "GRF": grf.tolist(),
        "COP": cop.tolist(),
        "foot_segments": detect_foot_segments(grf, fps),
        "has_mesh": "vertices" in data,
    }

    json_path = os.path.join(outdir, f"{vid_name}.json")
    with open(json_path, "w") as f:
        json.dump(session, f)
    print(f"wrote {json_path}  ({os.path.getsize(json_path)/1e6:.2f} MB)")

    if "vertices" in data and data["vertices"] is not None:
        verts = np.asarray(data["vertices"], dtype=np.float32)  # [T, 6890, 3]
        verts_path = os.path.join(outdir, f"{vid_name}_verts.bin")
        with open(verts_path, "wb") as f:
            f.write(verts.tobytes())
        print(f"wrote {verts_path}  ({os.path.getsize(verts_path)/1e6:.2f} MB, "
              f"shape {verts.shape}, float32 flat)")

    return session


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_path", help="path to <vid_name>_pred_mp.npz")
    ap.add_argument("--outdir", default="./converted")
    ap.add_argument("--subject", default=None, help="override auto-parsed subject label")
    ap.add_argument("--action", default=None, help="override auto-parsed action label")
    ap.add_argument("--trial", default=None, help="override auto-parsed trial label")
    args = ap.parse_args()
    convert(args.npz_path, args.outdir, args.subject, args.action, args.trial)