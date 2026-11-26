"""Video pipeline: frames -> detector -> ByteTrack -> unique counts per class + rescue-priority overlay -> GIF/MP4.

Tracking matters for rescue: counting detections per frame over-counts the same swimmer dozens of times; a
track ID per person lets the system report "N distinct people seen" and where each was first spotted."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from . import ID_CAT
from .detectors import Detections
from .rescue import assess, render


def read_frames(path, fps=4.0, start=0.0, duration=None, max_side=1280):
    tmp = Path(tempfile.mkdtemp())
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)] + (["-t", str(duration)] if duration else []) + \
          ["-i", str(path), "-vf", f"fps={fps},scale='min({max_side},iw)':-2", str(tmp / "%05d.jpg")]
    subprocess.run(cmd, check=True)
    for p in sorted(tmp.glob("*.jpg")):
        yield cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


class Tracker:
    def __init__(self, fps):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
        args = IterableSimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.2, track_low_thresh=0.1,
                                       new_track_thresh=0.25, track_buffer=int(3 * fps), match_thresh=0.9,
                                       fuse_score=True)
        self.t = BYTETracker(args)
        self.hits = {}   # track id -> (class, n_frames)

    def update(self, d: Detections, shape):
        from ultralytics.engine.results import Boxes
        if len(d) == 0:
            arr = np.zeros((0, 6), np.float32)
        else:
            arr = np.concatenate([d.boxes, d.scores[:, None], (d.labels - 1)[:, None]], 1).astype(np.float32)
        out = self.t.update(Boxes(arr, shape[:2]))
        tracks = []
        for row in out:
            x0, y0, x1, y1, tid, score, cls = row[:7]
            tid, cls = int(tid), int(cls) + 1
            c, n = self.hits.get(tid, (cls, 0))
            self.hits[tid] = (c, n + 1)
            tracks.append((tid, cls, float(score), (x0, y0, x1, y1)))
        return tracks

    def unique_counts(self, min_frames=3):
        out = {}
        for tid, (c, n) in self.hits.items():
            if n >= min_frames:
                out[ID_CAT[c]] = out.get(ID_CAT[c], 0) + 1
        return out


def process_video(path, engine, out_dir, stem, fps=4.0, start=0.0, duration=None, water_seg=None,
                  gif_side=480, min_frames=3):
    from .infer import to_rescue_dets
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = Tracker(fps)
    frames, per_frame = [], []
    for k, img in enumerate(read_frames(path, fps, start, duration)):
        d, labels = engine(img)
        d = to_rescue_dets(d, labels)
        tracks = tr.update(d, img.shape)
        water = water_seg.predict(img) if water_seg is not None else None
        # draw every detection of this frame (tracks only add the ID label and drive the unique counts)
        findings = assess(d, water)
        vis = render(img, findings, water=water)
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        for (tid, cls, sc, (x0, y0, x1, y1)) in tracks:
            cv2.putText(bgr, f"#{tid}", (int(x0), int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        uc = tr.unique_counts(min_frames)
        hud = "  ".join(f"{k}: {v}" for k, v in sorted(uc.items())) or "scanning..."
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(bgr, f"RescueEye | unique tracks  {hud}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    1, cv2.LINE_AA)
        vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(vis)
        per_frame.append({"frame": k, "detections": int(len(d)), "tracks": len(tracks)})
    s = gif_side / max(frames[0].shape[:2])
    small = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for f in frames]
    imageio.mimsave(out / f"{stem}.gif", small, duration=1.0 / fps, loop=0)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    summary = {"video": str(path), "fps_sampled": fps, "start_s": start, "frames": len(frames),
               "unique_tracks": tr.unique_counts(min_frames),
               "sum_of_per_frame_detections": int(sum(p["detections"] for p in per_frame))}
    json.dump(summary, open(out / f"{stem}_summary.json", "w"), indent=1)
    return summary
