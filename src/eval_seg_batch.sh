#!/bin/bash

fgL=(2 2 2 2 2)
fgU=(70 70 70 70 70)
iouL=(5 10 20 30 40)
n="${#iouL[@]}"
n=$((n-1))

for i in $(seq 0 "$n")
do
python eval_seg.py -src ../datasets/gt_all.txt \
-target ../datasets/nlcgt_all.txt \
-fgL "${fgL[$i]}" -fgU "${fgU[$i]}" -iouL "${iouL[$i]}" \
-patient \
&>> eval_results.txt
done

for i in $(seq 0 "$n")
do
python eval_seg.py -src ../datasets/gt_all.txt \
-target ../datasets/crfgt_all.txt \
-fgL "${fgL[$i]}" -fgU "${fgU[$i]}" -iouL "${iouL[$i]}" \
-patient \
&>> eval_results.txt
done
