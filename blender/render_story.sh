#!/bin/bash
# GPU box: render every chapter of the story cut with Cycles (OptiX), encode each to a
# near-lossless MP4 for transfer, then the posters and the livery sheet.
#   bash blender/render_story.sh [shot ...]
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
F=results/bdx/film2
mkdir -p $F/shots results/bdx/stills
export BDX_ENGINE=CYCLES BDX_SAMPLES=${BDX_SAMPLES:-128}
SHOTS=${@:-waking listening crossing wind leap arrival nightfall answer excited together}
for s in $SHOTS; do
  [ -d $F/$s ] && find $F/$s -size 0 -delete  # placeholders left by an interrupted render
  $B -b -P blender/film2.py -- $s > $F/log_$s.txt 2>&1
  ffmpeg -v error -y -framerate 24 -i $F/$s/%04d.png -c:v libx264 -preset slow -crf 10 -pix_fmt yuv420p $F/shots/$s.mp4
  echo "done $s $(ls $F/$s | wc -l) frames"
done
if [ $# -eq 0 ] || [ -n "$STILLS" ]; then
  for s in keyart keyart_night sheet; do
    $B -b -P blender/stills.py -- $s > results/bdx/stills/log_$s.txt 2>&1; echo "done still $s"
  done
fi
echo STORY_DONE
