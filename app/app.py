"""RescueEye demo (Gradio, CPU).

Upload an aerial/drone image or a short video, choose the engine and (for the open-vocabulary teacher) type what
to look for. You get detections, counts, a rescue-priority heat-map and downloadable CSV / GeoJSON reports.

    python app/app.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import os  # noqa: E402
os.chdir(ROOT)

from rescueeye.infer import build_engine  # noqa: E402
from rescueeye.predict import parse_prompt, run_image  # noqa: E402

_ENGINES = {}


def engine(kind):
    if kind not in _ENGINES:
        _ENGINES[kind] = build_engine(kind)
    return _ENGINES[kind]


def analyse_image(img, kind, prompt, water, geo):
    if img is None:
        return None, None, "Upload an image first.", None
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "upload.jpg"
    Image.fromarray(img).convert("RGB").save(p, quality=95)
    prompts = parse_prompt(prompt) if kind == "teacher" and prompt.strip() else None
    s, det_vis, hm, findings = run_image(p, engine(kind), prompts, "auto" if water else "none", geo or None, tmp)
    pri = np.array(Image.open(tmp / "upload_priority.jpg"))
    lines = [f"**{k}**: {v}" for k, v in sorted(s["counts"].items())]
    lines += [f"{k}: {v}" for k, v in s.get("other_free_text_detections", {}).items()]
    lv = s["levels"]
    text = ("### Counts\n" + ("  \n".join(lines) or "nothing detected") +
            f"\n\n### Priority\nCRITICAL {lv['CRITICAL']} · HIGH {lv['HIGH']} · MEDIUM {lv['MEDIUM']} · LOW {lv['LOW']}"
            f"\n\nPeople in or near water: **{s['people_in_or_near_water']}** "
            f"(without a boat nearby: **{s['people_without_boat_nearby_in_water']}**)")
    files = [str(tmp / "upload_detections.csv"), str(tmp / "upload_priority.geojson"), str(tmp / "upload_summary.json")]
    return det_vis, pri, text, files


def analyse_video(path, kind, fps):
    if not path:
        return None, "Upload a video first.", None
    from rescueeye.video import process_video
    tmp = Path(tempfile.mkdtemp())
    s = process_video(path, engine(kind), tmp, "clip", fps=fps, duration=30)
    return str(tmp / "clip.mp4"), "```json\n" + json.dumps(s, indent=1) + "\n```", str(tmp / "clip_summary.json")


_LIVE = {"model": None, "classes": None}


def live_detect(frame, prompt, conf):
    if frame is None:
        return None
    from ultralytics import YOLOWorld
    if _LIVE["model"] is None:
        _LIVE["model"] = YOLOWorld(str(ROOT / "yolov8s-worldv2.pt"))
    classes = [c.strip() for c in (prompt or "person").split(",") if c.strip()] or ["person"]
    if classes != _LIVE["classes"]:
        _LIVE["model"].set_classes(classes)
        _LIVE["classes"] = classes
    r = _LIVE["model"].predict(frame[..., ::-1], imgsz=480, conf=conf, verbose=False)[0]
    return r.plot()[..., ::-1]


examples = [str(p) for p in sorted((ROOT / "app" / "examples").glob("*.jpg"))]

with gr.Blocks(title="RescueEye") as demo:
    gr.Markdown("# RescueEye\nAerial flood-rescue detection on CPU: people, vehicles, boats, flooded buildings, "
                "plus a rescue-priority map. *Research prototype - always verify with a human.*")
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Aerial / drone image (or take a webcam photo)")
                kind = gr.Radio(["student", "teacher"], value="student",
                                label="Engine (student = fast YOLO11n; teacher = open-vocabulary YOLO-World + slicing)")
                prompt = gr.Textbox(label="Teacher prompt (comma-separated; known classes: person, vehicle, boat, "
                                          "flooded_building; anything else is free text)",
                                    value="person, vehicle=car, boat, flooded_building=flooded house")
                water = gr.Checkbox(value=True, label="Segment flood water (in-water priority)")
                geo = gr.Textbox(label="Optional geo: lat,lon,gsd_m of image centre (north-up)", value="")
                btn = gr.Button("Analyse", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp)
            with gr.Column():
                out1 = gr.Image(label="Detections (colour = priority, blue tint = water)")
                out2 = gr.Image(label="Rescue-priority heat-map")
                txt = gr.Markdown()
                files = gr.File(label="Reports (CSV, GeoJSON, JSON)", file_count="multiple")
        btn.click(analyse_image, [inp, kind, prompt, water, geo], [out1, out2, txt, files])
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                vin = gr.Video(label="Aerial video (first 30 s are processed)")
                vkind = gr.Radio(["student", "teacher"], value="student", label="Engine")
                vfps = gr.Slider(1, 8, value=3, step=1, label="Frames per second to analyse")
                vbtn = gr.Button("Track", variant="primary")
            with gr.Column():
                vout = gr.Video(label="Tracked output")
                vtxt = gr.Markdown()
                vfile = gr.File(label="Summary")
        vbtn.click(analyse_video, [vin, vkind, vfps], [vout, vtxt, vfile])
    with gr.Tab("Live camera"):
        gr.Markdown("Live open-vocabulary detection from your webcam (YOLO-World). Type any objects to look for.")
        with gr.Row():
            with gr.Column():
                cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
                lprompt = gr.Textbox(label="Objects to detect (comma-separated)", value="person, phone, bottle, car, boat")
                lconf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            with gr.Column():
                lout = gr.Image(label="Detections")
        cam.stream(live_detect, [cam, lprompt, lconf], [lout], stream_every=0.3, concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv, server_name="0.0.0.0")
