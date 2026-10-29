#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --job-name=cvri-seg
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_G1_segmentation/logs/%x-%j.out
# usage: sbatch --qos=test --time=00:30:00 seg_job.sh "101,102,104" metrics_tag
set -euo pipefail
PIDS=${1:-101,102,104}; TAG=${2:-gate}
ROOT=/project/home/p201509/work/cvri/simulation
CODE=$ROOT/code/gates/seg
RES=$ROOT/results/2026-10-28_G1_segmentation
OUT=$ROOT/data/processed/acdc_seg
WORK=$RES/work_${SLURM_JOB_ID}
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p $WORK $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv
SEG=/project/home/p201509/envs/cvri-seg/bin/python
TFP=/project/home/p201509/envs/cvri-seg-tf/bin/python
date; $SEG -I $CODE/run_nnunet.py --pids "$PIDS" --out $OUT --work $WORK 2>&1 | grep -v -E "^\s*$" | tail -60 || echo "NNUNET_FAILED"
[[ ${STAGES:-all} == all ]] && date && $SEG -I $CODE/run_cinema.py --pids "$PIDS" --out $OUT || echo "CINEMA_FAILED"
[[ ${STAGES:-all} == all ]] && date && CUDA_VISIBLE_DEVICES="" $TFP -I $CODE/run_ukbb_cardiac.py --pids "$PIDS" --out $OUT 2>&1 | grep -v -i "warning\|cpu_feature\|oneDNN\|rebuild TensorFlow" || echo "UKBB_FAILED"
date; $SEG -I $CODE/evaluate.py --pids "$PIDS" --seg $OUT --metrics $RES/metrics_${TAG}.tsv ${3:+--manifest $OUT/MANIFEST.tsv}
tar -C $RES --exclude="*.npz" -czf $RES/work_${SLURM_JOB_ID}.tar.gz work_${SLURM_JOB_ID} && rm -rf $WORK
date
