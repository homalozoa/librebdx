#!/bin/bash
# Second pass after the flaw review: title, every shot, and the livery stills.
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
F=results/bdx/film
$B -b -P blender/titles.py -- title > $F/log_title.txt 2>&1; echo "done title"
for s in establish engine walk push transitions jump dance excited tantrum; do
  $B -b -P blender/film.py -- $s > $F/log_$s.txt 2>&1
  echo "done $s $(ls $F/$s | wc -l)"
done
for s in keyart keyart_night sheet; do
  $B -b -P blender/stills.py -- $s > results/bdx/stills/log_$s.txt 2>&1; echo "done still $s"
done
echo RENDER_DONE
