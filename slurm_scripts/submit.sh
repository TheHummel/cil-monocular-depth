#!/bin/sh

# check for cmd line arg (model name)
if [ -z "$1" ]; then
  echo "Error: No argument provided."
  echo "Usage: $0 <model_name>"
  exit 1
fi

# export env vars
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  . "$ENV_FILE"
  set +o allexport
fi

model_name="$1"
path_predictions="../output/${model_name}/predictions.csv"

echo "Submitting model ${model_name}"

kaggle competitions submit -c ethz-cil-monocular-depth-estimation-2025 -f ${path_predictions} -m "Message"

echo "Successfully submitted"