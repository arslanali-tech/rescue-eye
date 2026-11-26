"""Command line interface: python -m rescueeye <command> [options]."""
from __future__ import annotations

import argparse
import importlib

import yaml

COMMANDS = {
    "download": ("rescueeye.data", "cli_download", "download and harmonise the datasets"),
    "zeroshot": ("rescueeye.zeroshot", "cli_zeroshot", "zero-shot benchmark from cached detections"),
    "tune": ("rescueeye.tune", "cli_tune", "SAEA vs GA vs random search for teacher post-processing"),
    "students": ("rescueeye.students", "cli_students", "train/evaluate YOLO11n students (supervised / distilled)"),
    "water": ("rescueeye.water", "cli_water", "train/evaluate the flood-water segmenter on FloodNet"),
    "predict": ("rescueeye.predict", "cli_predict", "detect + rescue-priority report for images"),
    "video": ("rescueeye.predict", "cli_video", "track people/boats/vehicles in a video, count each once"),
    "figures": ("rescueeye.make_figures", "cli_figures", "make the README figures from saved results"),
    "cache": ("rescueeye.cache", "cli_cache", "run a detector and cache raw phrase-level detections"),
}


def load_cfg(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    p = argparse.ArgumentParser(prog="rescueeye", description=__doc__, allow_abbrev=False)
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--config", default="configs/default.yaml")
    args, rest = p.parse_known_args(argv)
    mod, fn, _ = COMMANDS[args.command]
    getattr(importlib.import_module(mod), fn)(load_cfg(args.config), rest)


if __name__ == "__main__":
    main()
