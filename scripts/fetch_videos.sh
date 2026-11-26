#!/usr/bin/env bash
# Freely licensed aerial flood videos from Wikimedia Commons (see README for attribution).
set -e
mkdir -p data/video
UA="RescueEye/0.1 (research prototype)"
curl -sL -A "$UA" -o data/video/aqqala.webm "https://upload.wikimedia.org/wikipedia/commons/2/2e/Aerial_video_of_the_aftermath_of_the_flood_in_Aqqala.webm"
curl -sL -A "$UA" -o data/video/oleshky.webm "https://upload.wikimedia.org/wikipedia/commons/7/72/Flooded_Oleshky%2C_2023-06-07.webm"
