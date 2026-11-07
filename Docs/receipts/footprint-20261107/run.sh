#!/bin/zsh
# AB-A-0176 footprint ladder: one fresh process per row through the real MLXServeEngine (birefnet-smoke, Release),
# process phys_footprint lifetime peak from `/usr/bin/time -l` ("peak memory footprint").
# usage: run.sh <inputs dir with {1,12,24,48,100}mp.jpg> <out dir> <fast.safetensors> <best.safetensors>
# BIREFNET_SMOKE_GPU_CACHE=unmanaged | <bytes> sets the engine's MLX pool policy (default: the engine's automatic cap).
set -u
IN=$1 OUT=$2 F=$3 B=$4
mkdir -p $OUT
for mode in fast best; do for mp in 1 12 24 48 100; do
  /usr/bin/time -l .build/release/birefnet-smoke $IN/${mp}mp.jpg $F $B $OUT/${mode}_${mp}mp.png $mode > $OUT/${mode}_${mp}mp.log 2>&1
  pk=$(awk '/peak memory footprint/{print $1}' $OUT/${mode}_${mp}mp.log)
  echo "$mode ${mp}MP peak_phys=$(echo "scale=2; $pk/1000000000" | bc) GB"
done; done
# Inputs: the Commons lorikeet photo (corpus/commons-sr/images/animals/066534b43e_Wide_angle_rainbow_lorikeet.jpg,
# 5184x3456) resized with `sips -z H W` to 1152x864, 4000x3000, 5656x4242, 8000x6000 and 11548x8660, JPEG q92.
