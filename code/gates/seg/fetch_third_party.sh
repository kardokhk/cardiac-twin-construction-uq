#!/bin/bash
# Fetch the third-party segmentation sources at the commits used in study I11 into code/gates/seg/src.
# CineMA (MIT licence) and ukbb_cardiac (Apache 2.0) are not redistributed in this repository.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p src; cd src
[ -d CineMA ] || git clone https://github.com/mathpluscode/CineMA.git
git -C CineMA checkout c10daa1d
[ -d ukbb_cardiac ] || git clone https://github.com/baiwenjia/ukbb_cardiac.git
git -C ukbb_cardiac checkout f465832c
