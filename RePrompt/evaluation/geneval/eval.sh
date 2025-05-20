#!/bin/bash

IMAGE_FOLDER="reprompt_flux_on_geneval"
RESULTS_FOLDER="output/results"

python evaluation/evaluate_images.py \
    "$RESULTS_FOLDER/$IMAGE_FOLDER" \
    --outfile "$RESULTS_FOLDER/$IMAGE_FOLDER.jsonl" \
    --model-path "evaluation"

python evaluation/summary_scores.py "$RESULTS_FOLDER/$IMAGE_FOLDER.jsonl"