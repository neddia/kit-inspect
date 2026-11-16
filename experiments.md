# Experiments

One row per experiment. Change one thing at a time. Every number comes from a script run, never an estimate.

| Date | Stage | What changed (one thing) | Result | Kept / dropped | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-15 | 2 | Zero-shot YOLOE-26s, text prompts "bolt / nut / screw" (pilot, 38 images) | Count acc 3% at every cutoff 0.05–0.5; 0/20 false passes but 18/18 false rejects (never says PASS). Never finds bolts; labels bolt heads as "nut". | Baseline | `src/eval/zero_shot.py`, reports/stage2/pilot_yoloe-26s-seg_text |
| 2026-11-15 | 2 | Model size: YOLOE-26l instead of 26s (same text prompts) | Count acc 3% (best cutoff 0.02); 18/18 false rejects. No better than 26s. | Dropped (26s) | reports/stage2/pilot_yoloe-26l-seg_text |
| 2026-11-15 | 2 | Prompt wording: "long threaded hex bolt / small hex nut / thin flat head screw" (26s) | Count acc 3%. Nut count error best 0.45 at cutoff 0.15 (was 1.11); bolts and screws still ~never found; 18/18 false rejects. | Kept for nuts only | reports/stage2/pilot_yoloe-26s-seg_text_descriptive |
| 2026-11-15 | 2 | Visual prompt instead of text: one box per part on pilot_0001, drawn by Bob (26s) | Count acc 3%; 0/20 false passes, 18/18 false rejects. Finds some bolts at very low cutoff (bolt error 1.03 at 0.01 vs 2.82 for text). Confidences 0.02–0.4. Reference boxes include neighbouring parts (bolt box holds a ring; screw box holds a bolt head), so matches mix parts. | Baseline | reports/stage2/pilot_yoloe-26s-seg_visual. Could retry with tight boxes on a scattered image |
| 2026-11-15 | 2 | Visual prompt boxes: tight, one part each, on scattered tray pilot_0028 (was loose boxes on crowded pilot_0001) | Best: count acc 11% at cutoff 0.02 (4/38), 2/18 complete kits passed, 0/20 false passes. Unstable: 3% at 0.01 and 0.05+; 1 false pass at 0.01. Nut error 0.21 (was 1.13), bolt 0.95, screw 1.05+. | Best zero-shot result | reports/stage2/visual_tight. Optimistic: cutoff and reference image both from the scored set |
