#!/bin/bash
# Render every title and shot at full quality, then cut the film.
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
for t in intro title crawl end; do
  $B -b -P blender/titles.py -- $t > results/bdx/film/log_$t.txt 2>&1
  echo "done $t"
done
for s in establish engine walk push dance excited tantrum jump transitions; do
  $B -b -P blender/film.py -- $s > results/bdx/film/log_$s.txt 2>&1
  echo "done $s"
done
echo RENDER_DONE
