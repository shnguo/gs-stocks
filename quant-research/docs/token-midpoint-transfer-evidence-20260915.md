# Midpoint calibration and transfer: evidence

Generated 2026-09-16T05:03:43.589938+00:00. Read with the main report and frozen protocol.

Errors and interval widths use percentage-point returns. Coverage is a proportion. Relative changes are fractions, so −0.01 means a 1% reduction. Confidence bounds describe absolute candidate-minus-baseline differences. Lower errors and interval scores are better; coverage targets 0.8. Calibrated CRPS is intentionally unavailable because interval scaling does not define a full calibrated distribution.

## Average high/low metrics

| variant | horizon | mae | coverage80 | width80 | interval_score80 | crps |
| --- | --- | --- | --- | --- | --- | --- |
| ce_calibrated | 2 | 2.750593 | 0.824332 | 9.281937 | 14.946444 | — |
| ce_calibrated | 5 | 4.345011 | 0.828893 | 14.324054 | 22.951075 | — |
| ce_raw | 2 | 2.750593 | 0.717434 | 7.145224 | 14.971285 | 2.086133 |
| ce_raw | 5 | 4.345011 | 0.725167 | 11.415068 | 23.017372 | 3.265837 |
| current_calibrated | 2 | 2.740190 | 0.815562 | 9.089749 | 14.878665 | — |
| current_calibrated | 5 | 4.308029 | 0.828491 | 14.257470 | 22.849953 | — |
| current_raw | 2 | 2.740190 | 0.720126 | 7.164243 | 14.933880 | 2.076553 |
| current_raw | 5 | 4.308029 | 0.730100 | 11.472538 | 22.884654 | 3.238077 |
| midpoint_calibrated | 2 | 2.757670 | 0.823528 | 9.247743 | 14.997298 | — |
| midpoint_calibrated | 5 | 4.357591 | 0.831936 | 14.412853 | 23.219777 | — |
| midpoint_raw | 2 | 2.757670 | 0.713970 | 7.141601 | 15.057343 | 2.091310 |
| midpoint_raw | 5 | 4.357591 | 0.725104 | 11.392949 | 23.286848 | 3.281792 |

## Common cohorts

| seed | horizon | rows | dates |
| --- | --- | --- | --- |
| 17 | 2 | 2381 | 76 |
| 17 | 5 | 2301 | 76 |
| 29 | 2 | 2381 | 76 |
| 29 | 5 | 2300 | 76 |
| 43 | 2 | 2381 | 76 |
| 43 | 5 | 2299 | 76 |

## pipeline_transfer

| horizon | target | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | mae | 3.148458 | 3.147100 | -0.000432 | -0.018341 | 0.015489 |
| 2 | maximum | coverage80 | 0.805981 | 0.814773 | 0.010909 | -0.001412 | 0.016870 |
| 2 | maximum | width80 | 9.944520 | 9.942170 | -0.000236 | -0.112830 | 0.094781 |
| 2 | maximum | interval_score80 | 16.889752 | 16.941584 | 0.003069 | -0.094915 | 0.191912 |
| 2 | minimum | mae | 2.331921 | 2.368239 | 0.015574 | 0.003453 | 0.080517 |
| 2 | minimum | coverage80 | 0.825143 | 0.832283 | 0.008654 | 0.002959 | 0.011826 |
| 2 | minimum | width80 | 8.234979 | 8.553315 | 0.038657 | 0.262228 | 0.370306 |
| 2 | minimum | interval_score80 | 12.867578 | 13.053012 | 0.014411 | 0.021724 | 0.354922 |
| 2 | range | mae | 2.786765 | 2.811393 | 0.008838 | 0.004584 | 0.049653 |
| 2 | range | coverage80 | 0.774217 | 0.780629 | 0.008282 | 0.001978 | 0.010933 |
| 2 | range | width80 | 9.312243 | 9.399153 | 0.009333 | 0.008383 | 0.161581 |
| 2 | range | interval_score80 | 15.318176 | 15.310453 | -0.000504 | -0.175527 | 0.151278 |
| 2 | high_low | mae | 2.740190 | 2.757670 | 0.006379 | -0.003343 | 0.043955 |
| 2 | high_low | coverage80 | 0.815562 | 0.823528 | 0.009768 | 0.001202 | 0.013932 |
| 2 | high_low | width80 | 9.089749 | 9.247743 | 0.017381 | 0.102138 | 0.216237 |
| 2 | high_low | interval_score80 | 14.878665 | 14.997298 | 0.007973 | -0.004685 | 0.238873 |
| 5 | maximum | mae | 5.101260 | 5.099620 | -0.000321 | -0.039536 | 0.029977 |
| 5 | maximum | coverage80 | 0.810581 | 0.813731 | 0.003886 | -0.003442 | 0.009047 |
| 5 | maximum | width80 | 15.109305 | 14.950744 | -0.010494 | -0.421098 | 0.059901 |
| 5 | maximum | interval_score80 | 27.362436 | 27.691458 | 0.012025 | 0.048669 | 0.626585 |
| 5 | minimum | mae | 3.514799 | 3.615561 | 0.028668 | 0.014570 | 0.208657 |
| 5 | minimum | coverage80 | 0.846401 | 0.850142 | 0.004419 | -0.005245 | 0.010813 |
| 5 | minimum | width80 | 13.405636 | 13.874961 | 0.035010 | 0.288668 | 0.644824 |
| 5 | minimum | interval_score80 | 18.337470 | 18.748096 | 0.022393 | 0.096752 | 0.722014 |
| 5 | range | mae | 4.898475 | 4.959687 | 0.012496 | 0.021883 | 0.112478 |
| 5 | range | coverage80 | 0.750074 | 0.759941 | 0.013154 | 0.004509 | 0.015320 |
| 5 | range | width80 | 14.751356 | 14.936480 | 0.012550 | -0.013780 | 0.372387 |
| 5 | range | interval_score80 | 25.938195 | 26.120145 | 0.007015 | -0.069299 | 0.417378 |
| 5 | high_low | mae | 4.308029 | 4.357591 | 0.011504 | -0.002856 | 0.109948 |
| 5 | high_low | coverage80 | 0.828491 | 0.831936 | 0.004159 | -0.001926 | 0.008243 |
| 5 | high_low | width80 | 14.257470 | 14.412853 | 0.010898 | -0.044495 | 0.336145 |
| 5 | high_low | interval_score80 | 22.849953 | 23.219777 | 0.016185 | 0.192217 | 0.551589 |

## loss_transfer

| horizon | target | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | mae | 3.140917 | 3.147100 | 0.001968 | -0.014868 | 0.025076 |
| 2 | maximum | coverage80 | 0.813139 | 0.814773 | 0.002010 | -0.005953 | 0.008976 |
| 2 | maximum | width80 | 9.967656 | 9.942170 | -0.002557 | -0.113510 | 0.067240 |
| 2 | maximum | interval_score80 | 16.906818 | 16.941584 | 0.002056 | -0.114903 | 0.187715 |
| 2 | minimum | mae | 2.360269 | 2.368239 | 0.003377 | -0.012400 | 0.033280 |
| 2 | minimum | coverage80 | 0.835526 | 0.832283 | -0.003881 | -0.007088 | 0.000744 |
| 2 | minimum | width80 | 8.596219 | 8.553315 | -0.004991 | -0.116142 | 0.013576 |
| 2 | minimum | interval_score80 | 12.986070 | 13.053012 | 0.005155 | -0.029653 | 0.189216 |
| 2 | range | mae | 2.799391 | 2.811393 | 0.004288 | -0.009016 | 0.035529 |
| 2 | range | coverage80 | 0.779725 | 0.780629 | 0.001159 | -0.005011 | 0.006472 |
| 2 | range | width80 | 9.370648 | 9.399153 | 0.003042 | -0.037667 | 0.087878 |
| 2 | range | interval_score80 | 15.278861 | 15.310453 | 0.002068 | -0.093689 | 0.161931 |
| 2 | high_low | mae | 2.750593 | 2.757670 | 0.002573 | -0.011585 | 0.026819 |
| 2 | high_low | coverage80 | 0.824332 | 0.823528 | -0.000975 | -0.005778 | 0.004349 |
| 2 | high_low | width80 | 9.281937 | 9.247743 | -0.003684 | -0.071776 | 0.003580 |
| 2 | high_low | interval_score80 | 14.946444 | 14.997298 | 0.003402 | -0.052881 | 0.161121 |
| 5 | maximum | mae | 5.092355 | 5.099620 | 0.001427 | -0.020765 | 0.032608 |
| 5 | maximum | coverage80 | 0.811727 | 0.813731 | 0.002468 | -0.002727 | 0.006653 |
| 5 | maximum | width80 | 14.980204 | 14.950744 | -0.001967 | -0.143299 | 0.111553 |
| 5 | maximum | interval_score80 | 27.388284 | 27.691458 | 0.011069 | 0.050710 | 0.584996 |
| 5 | minimum | mae | 3.597668 | 3.615561 | 0.004974 | -0.031974 | 0.063307 |
| 5 | minimum | coverage80 | 0.846059 | 0.850142 | 0.004826 | -0.000209 | 0.008684 |
| 5 | minimum | width80 | 13.667903 | 13.874961 | 0.015149 | 0.073023 | 0.340003 |
| 5 | minimum | interval_score80 | 18.513865 | 18.748096 | 0.012652 | 0.050090 | 0.442830 |
| 5 | range | mae | 4.908127 | 4.959687 | 0.010505 | 0.022867 | 0.082765 |
| 5 | range | coverage80 | 0.759318 | 0.759941 | 0.000820 | -0.004805 | 0.006427 |
| 5 | range | width80 | 14.990364 | 14.936480 | -0.003595 | -0.134985 | 0.022923 |
| 5 | range | interval_score80 | 25.889529 | 26.120145 | 0.008908 | 0.055210 | 0.417517 |
| 5 | high_low | mae | 4.345011 | 4.357591 | 0.002895 | -0.021935 | 0.044873 |
| 5 | high_low | coverage80 | 0.828893 | 0.831936 | 0.003671 | -0.000517 | 0.006612 |
| 5 | high_low | width80 | 14.324054 | 14.412853 | 0.006199 | 0.030432 | 0.141770 |
| 5 | high_low | interval_score80 | 22.951075 | 23.219777 | 0.011708 | 0.100607 | 0.425160 |

## midpoint_calibration

| horizon | target | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | mae | 3.147100 | 3.147100 | 0.000000 | 0.000000 | 0.000000 |
| 2 | maximum | coverage80 | 0.705664 | 0.814773 | 0.154620 | 0.100270 | 0.118559 |
| 2 | maximum | width80 | 7.668959 | 9.942170 | 0.296417 | 2.070799 | 2.480603 |
| 2 | maximum | interval_score80 | 17.167010 | 16.941584 | -0.013131 | -0.934517 | 0.283022 |
| 2 | minimum | mae | 2.368239 | 2.368239 | 0.000000 | 0.000000 | 0.000000 |
| 2 | minimum | coverage80 | 0.722277 | 0.832283 | 0.152305 | 0.095013 | 0.126481 |
| 2 | minimum | width80 | 6.614244 | 8.553315 | 0.293166 | 1.601572 | 2.319684 |
| 2 | minimum | interval_score80 | 12.947675 | 13.053012 | 0.008136 | -0.870622 | 0.728938 |
| 2 | range | mae | 2.811393 | 2.811393 | 0.000000 | 0.000000 | 0.000000 |
| 2 | range | coverage80 | 0.637591 | 0.780629 | 0.224341 | 0.130196 | 0.155843 |
| 2 | range | width80 | 6.925778 | 9.399153 | 0.357126 | 2.180878 | 2.791876 |
| 2 | range | interval_score80 | 15.900113 | 15.310453 | -0.037085 | -1.579660 | 0.114793 |
| 2 | high_low | mae | 2.757670 | 2.757670 | 0.000000 | 0.000000 | 0.000000 |
| 2 | high_low | coverage80 | 0.713970 | 0.823528 | 0.153449 | 0.102639 | 0.116630 |
| 2 | high_low | width80 | 7.141601 | 9.247743 | 0.294912 | 1.842675 | 2.390034 |
| 2 | high_low | interval_score80 | 15.057343 | 14.997298 | -0.003988 | -0.875408 | 0.467831 |
| 5 | maximum | mae | 5.099620 | 5.099620 | 0.000000 | 0.000000 | 0.000000 |
| 5 | maximum | coverage80 | 0.732882 | 0.813731 | 0.110315 | 0.072463 | 0.089753 |
| 5 | maximum | width80 | 12.223801 | 14.950744 | 0.223085 | 2.505745 | 2.943082 |
| 5 | maximum | interval_score80 | 27.920916 | 27.691458 | -0.008218 | -0.858371 | 0.268340 |
| 5 | minimum | mae | 3.615561 | 3.615561 | 0.000000 | 0.000000 | 0.000000 |
| 5 | minimum | coverage80 | 0.717325 | 0.850142 | 0.185156 | 0.110144 | 0.158601 |
| 5 | minimum | width80 | 10.562098 | 13.874961 | 0.313656 | 2.676822 | 4.044641 |
| 5 | minimum | interval_score80 | 18.652780 | 18.748096 | 0.005110 | -1.636878 | 1.439197 |
| 5 | range | mae | 4.959687 | 4.959687 | 0.000000 | 0.000000 | 0.000000 |
| 5 | range | coverage80 | 0.652225 | 0.759941 | 0.165151 | 0.096231 | 0.118488 |
| 5 | range | width80 | 12.372898 | 14.936480 | 0.207193 | 2.247920 | 2.916981 |
| 5 | range | interval_score80 | 26.838316 | 26.120145 | -0.026759 | -1.406172 | -0.158834 |
| 5 | high_low | mae | 4.357591 | 4.357591 | 0.000000 | 0.000000 | 0.000000 |
| 5 | high_low | coverage80 | 0.725104 | 0.831936 | 0.147334 | 0.095636 | 0.119947 |
| 5 | high_low | width80 | 11.392949 | 14.412853 | 0.265068 | 2.605937 | 3.477625 |
| 5 | high_low | interval_score80 | 23.286848 | 23.219777 | -0.002880 | -1.268595 | 0.797974 |

## ce_calibration

| horizon | target | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | mae | 3.140917 | 3.140917 | 0.000000 | 0.000000 | 0.000000 |
| 2 | maximum | coverage80 | 0.706340 | 0.813139 | 0.151201 | 0.097511 | 0.115751 |
| 2 | maximum | width80 | 7.690895 | 9.967656 | 0.296033 | 2.070954 | 2.476935 |
| 2 | maximum | interval_score80 | 17.106681 | 16.906818 | -0.011683 | -0.899187 | 0.316957 |
| 2 | minimum | mae | 2.360269 | 2.360269 | 0.000000 | 0.000000 | 0.000000 |
| 2 | minimum | coverage80 | 0.728528 | 0.835526 | 0.146868 | 0.093388 | 0.123169 |
| 2 | minimum | width80 | 6.599552 | 8.596219 | 0.302546 | 1.639980 | 2.397015 |
| 2 | minimum | interval_score80 | 12.835889 | 12.986070 | 0.011700 | -0.862533 | 0.799285 |
| 2 | range | mae | 2.799391 | 2.799391 | 0.000000 | 0.000000 | 0.000000 |
| 2 | range | coverage80 | 0.635785 | 0.779725 | 0.226397 | 0.133375 | 0.154265 |
| 2 | range | width80 | 6.959471 | 9.370648 | 0.346460 | 2.126429 | 2.719879 |
| 2 | range | interval_score80 | 15.825279 | 15.278861 | -0.034528 | -1.476918 | 0.128614 |
| 2 | high_low | mae | 2.750593 | 2.750593 | 0.000000 | 0.000000 | 0.000000 |
| 2 | high_low | coverage80 | 0.717434 | 0.824332 | 0.149001 | 0.099833 | 0.115424 |
| 2 | high_low | width80 | 7.145224 | 9.281937 | 0.299041 | 1.867212 | 2.425737 |
| 2 | high_low | interval_score80 | 14.971285 | 14.946444 | -0.001659 | -0.859491 | 0.511740 |
| 5 | maximum | mae | 5.092355 | 5.092355 | 0.000000 | 0.000000 | 0.000000 |
| 5 | maximum | coverage80 | 0.731464 | 0.811727 | 0.109730 | 0.070057 | 0.091480 |
| 5 | maximum | width80 | 12.275459 | 14.980204 | 0.220338 | 2.499869 | 2.904154 |
| 5 | maximum | interval_score80 | 27.650189 | 27.388284 | -0.009472 | -0.909200 | 0.238365 |
| 5 | minimum | mae | 3.597668 | 3.597668 | 0.000000 | 0.000000 | 0.000000 |
| 5 | minimum | coverage80 | 0.718870 | 0.846059 | 0.176929 | 0.105697 | 0.152501 |
| 5 | minimum | width80 | 10.554678 | 13.667903 | 0.294962 | 2.487785 | 3.825155 |
| 5 | minimum | interval_score80 | 18.384556 | 18.513865 | 0.007034 | -1.471207 | 1.330037 |
| 5 | range | mae | 4.908127 | 4.908127 | 0.000000 | 0.000000 | 0.000000 |
| 5 | range | coverage80 | 0.653139 | 0.759318 | 0.162567 | 0.092903 | 0.118883 |
| 5 | range | width80 | 12.425076 | 14.990364 | 0.206461 | 2.245782 | 2.914010 |
| 5 | range | interval_score80 | 26.554520 | 25.889529 | -0.025042 | -1.343696 | -0.132609 |
| 5 | high_low | mae | 4.345011 | 4.345011 | 0.000000 | 0.000000 | 0.000000 |
| 5 | high_low | coverage80 | 0.725167 | 0.828893 | 0.143037 | 0.093792 | 0.115368 |
| 5 | high_low | width80 | 11.415068 | 14.324054 | 0.254837 | 2.509643 | 3.345889 |
| 5 | high_low | interval_score80 | 23.017372 | 22.951075 | -0.002880 | -1.200258 | 0.731029 |

## raw_pipeline

| horizon | target | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | crps | 2.380908 | 2.383129 | 0.000933 | -0.009980 | 0.013882 |
| 2 | minimum | crps | 1.772198 | 1.799491 | 0.015401 | 0.006401 | 0.054273 |
| 2 | range | crps | 2.144902 | 2.159043 | 0.006593 | 0.002797 | 0.027255 |
| 2 | high_low | crps | 2.076553 | 2.091310 | 0.007107 | 0.000157 | 0.032297 |
| 5 | maximum | crps | 3.865196 | 3.871644 | 0.001668 | -0.019779 | 0.029504 |
| 5 | minimum | crps | 2.610958 | 2.691940 | 0.031017 | 0.023212 | 0.156657 |
| 5 | range | crps | 3.712930 | 3.752911 | 0.010768 | 0.012529 | 0.074628 |
| 5 | high_low | crps | 3.238077 | 3.281792 | 0.013500 | 0.008630 | 0.085141 |

## raw_loss

| horizon | target | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | maximum | crps | 2.381611 | 2.383129 | 0.000638 | -0.010808 | 0.014903 |
| 2 | minimum | crps | 1.790656 | 1.799491 | 0.004934 | -0.001597 | 0.019533 |
| 2 | range | crps | 2.152889 | 2.159043 | 0.002859 | -0.004012 | 0.016561 |
| 2 | high_low | crps | 2.086133 | 2.091310 | 0.002482 | -0.004578 | 0.016124 |
| 5 | maximum | crps | 3.864011 | 3.871644 | 0.001975 | -0.018314 | 0.034828 |
| 5 | minimum | crps | 2.667662 | 2.691940 | 0.009101 | -0.004810 | 0.050451 |
| 5 | range | crps | 3.720300 | 3.752911 | 0.008766 | 0.014180 | 0.050470 |
| 5 | high_low | crps | 3.265837 | 3.281792 | 0.004886 | -0.005916 | 0.037220 |

## Raw midpoint distribution

| comparison | horizon | metric | baseline | candidate | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current | 2 | mae | 2.547829 | 2.570716 | 0.008983 | -0.002395 | 0.055885 |
| current | 2 | crps | 1.923535 | 1.940101 | 0.008612 | 0.000979 | 0.035777 |
| current | 5 | mae | 4.003738 | 4.064156 | 0.015090 | 0.004002 | 0.131931 |
| current | 5 | crps | 2.995599 | 3.041989 | 0.015486 | 0.010299 | 0.089950 |
| ce | 2 | mae | 2.565267 | 2.570716 | 0.002124 | -0.015876 | 0.027739 |
| ce | 2 | crps | 1.935032 | 1.940101 | 0.002620 | -0.006205 | 0.017705 |
| ce | 5 | mae | 4.043567 | 4.064156 | 0.005092 | -0.012263 | 0.051740 |
| ce | 5 | crps | 3.027947 | 3.041989 | 0.004638 | -0.007080 | 0.034628 |

## Per-seed high/low changes

| comparison | seed | horizon | metric | relative_change | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline_transfer | 17 | 2 | mae | -0.002696 | -0.039685 | 0.023971 |
| pipeline_transfer | 29 | 2 | mae | -0.018495 | -0.093343 | -0.014734 |
| pipeline_transfer | 43 | 2 | mae | 0.039635 | 0.036835 | 0.204199 |
| pipeline_transfer | 17 | 2 | interval_score80 | -0.003019 | -0.272451 | 0.146068 |
| pipeline_transfer | 29 | 2 | interval_score80 | -0.008088 | -0.378349 | 0.133885 |
| pipeline_transfer | 43 | 2 | interval_score80 | 0.034291 | 0.206463 | 0.874358 |
| pipeline_transfer | 17 | 5 | mae | -0.006494 | -0.073965 | 0.017462 |
| pipeline_transfer | 29 | 5 | mae | -0.032238 | -0.276403 | -0.029255 |
| pipeline_transfer | 43 | 5 | mae | 0.070427 | 0.101725 | 0.596863 |
| pipeline_transfer | 17 | 5 | interval_score80 | 0.008853 | -0.104334 | 0.462211 |
| pipeline_transfer | 29 | 5 | interval_score80 | -0.014931 | -0.950899 | 0.231134 |
| pipeline_transfer | 43 | 5 | interval_score80 | 0.053135 | 0.562188 | 2.052747 |
| loss_transfer | 17 | 2 | mae | -0.003010 | -0.054731 | 0.034181 |
| loss_transfer | 29 | 2 | mae | -0.010313 | -0.061938 | -0.001621 |
| loss_transfer | 43 | 2 | mae | 0.020222 | 0.016500 | 0.110238 |
| loss_transfer | 17 | 2 | interval_score80 | 0.002803 | -0.128090 | 0.201602 |
| loss_transfer | 29 | 2 | interval_score80 | -0.008871 | -0.394190 | 0.132702 |
| loss_transfer | 43 | 2 | interval_score80 | 0.015618 | 0.054054 | 0.439369 |
| loss_transfer | 17 | 5 | mae | -0.004203 | -0.095268 | 0.047590 |
| loss_transfer | 29 | 5 | mae | -0.020207 | -0.172960 | -0.017608 |
| loss_transfer | 43 | 5 | mae | 0.030378 | 0.045814 | 0.251286 |
| loss_transfer | 17 | 5 | interval_score80 | 0.013546 | -0.151529 | 0.690525 |
| loss_transfer | 29 | 5 | interval_score80 | -0.007210 | -0.458891 | 0.139522 |
| loss_transfer | 43 | 5 | interval_score80 | 0.027609 | 0.228972 | 1.185837 |

## Descriptive monthly consistency

These summaries are descriptive and do not select factors, checkpoints or the loss weight. July is a partial month.

| comparison | horizon | metric | month | dates | baseline | candidate | relative_change | improved_date_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline_transfer | 2 | mae | 2025-04 | 21 | 4.483965 | 4.529168 | 0.010081 | 0.333333 |
| pipeline_transfer | 2 | mae | 2025-05 | 19 | 2.254457 | 2.261852 | 0.003280 | 0.473684 |
| pipeline_transfer | 2 | mae | 2025-06 | 20 | 1.927390 | 1.929999 | 0.001353 | 0.400000 |
| pipeline_transfer | 2 | mae | 2025-07 | 16 | 2.044292 | 2.055950 | 0.005703 | 0.500000 |
| pipeline_transfer | 2 | interval_score80 | 2025-04 | 21 | 24.690295 | 24.907681 | 0.008805 | 0.476190 |
| pipeline_transfer | 2 | interval_score80 | 2025-05 | 19 | 11.959074 | 12.128655 | 0.014180 | 0.315789 |
| pipeline_transfer | 2 | interval_score80 | 2025-06 | 20 | 10.234556 | 10.247175 | 0.001233 | 0.450000 |
| pipeline_transfer | 2 | interval_score80 | 2025-07 | 16 | 11.273052 | 11.334086 | 0.005414 | 0.375000 |
| pipeline_transfer | 5 | mae | 2025-04 | 21 | 6.897564 | 7.006359 | 0.015773 | 0.380952 |
| pipeline_transfer | 5 | mae | 2025-05 | 19 | 3.274450 | 3.310662 | 0.011059 | 0.368421 |
| pipeline_transfer | 5 | mae | 2025-06 | 20 | 3.187875 | 3.188873 | 0.000313 | 0.450000 |
| pipeline_transfer | 5 | mae | 2025-07 | 16 | 3.536834 | 3.585208 | 0.013677 | 0.312500 |
| pipeline_transfer | 5 | interval_score80 | 2025-04 | 21 | 34.657303 | 35.086514 | 0.012384 | 0.428571 |
| pipeline_transfer | 5 | interval_score80 | 2025-05 | 19 | 17.314260 | 17.504647 | 0.010996 | 0.263158 |
| pipeline_transfer | 5 | interval_score80 | 2025-06 | 20 | 16.973301 | 17.385962 | 0.024312 | 0.250000 |
| pipeline_transfer | 5 | interval_score80 | 2025-07 | 16 | 21.272255 | 21.723670 | 0.021221 | 0.250000 |
| loss_transfer | 2 | mae | 2025-04 | 21 | 4.511152 | 4.529168 | 0.003994 | 0.380952 |
| loss_transfer | 2 | mae | 2025-05 | 19 | 2.257138 | 2.261852 | 0.002089 | 0.473684 |
| loss_transfer | 2 | mae | 2025-06 | 20 | 1.934496 | 1.929999 | -0.002325 | 0.350000 |
| loss_transfer | 2 | mae | 2025-07 | 16 | 2.045961 | 2.055950 | 0.004882 | 0.312500 |
| loss_transfer | 2 | interval_score80 | 2025-04 | 21 | 24.689892 | 24.907681 | 0.008821 | 0.476190 |
| loss_transfer | 2 | interval_score80 | 2025-05 | 19 | 12.056460 | 12.128655 | 0.005988 | 0.526316 |
| loss_transfer | 2 | interval_score80 | 2025-06 | 20 | 10.367306 | 10.247175 | -0.011587 | 0.750000 |
| loss_transfer | 2 | interval_score80 | 2025-07 | 16 | 11.313948 | 11.334086 | 0.001780 | 0.562500 |
| loss_transfer | 5 | mae | 2025-04 | 21 | 6.996615 | 7.006359 | 0.001393 | 0.476190 |
| loss_transfer | 5 | mae | 2025-05 | 19 | 3.331831 | 3.310662 | -0.006353 | 0.578947 |
| loss_transfer | 5 | mae | 2025-06 | 20 | 3.169532 | 3.188873 | 0.006102 | 0.450000 |
| loss_transfer | 5 | mae | 2025-07 | 16 | 3.537284 | 3.585208 | 0.013548 | 0.250000 |
| loss_transfer | 5 | interval_score80 | 2025-04 | 21 | 34.569165 | 35.086514 | 0.014966 | 0.333333 |
| loss_transfer | 5 | interval_score80 | 2025-05 | 19 | 17.488953 | 17.504647 | 0.000897 | 0.526316 |
| loss_transfer | 5 | interval_score80 | 2025-06 | 20 | 17.096538 | 17.385962 | 0.016929 | 0.300000 |
| loss_transfer | 5 | interval_score80 | 2025-07 | 16 | 21.506772 | 21.723670 | 0.010085 | 0.187500 |

## Descriptive consistency across the two tested periods

Both periods use the same fixed predictor checkpoints and a common three-profile cohort within each seed/horizon. Values are raw point/distribution comparisons. These already examined development windows are not independent untouched tests. This summary does not change the frozen selection or calibration.

| window | baseline | seed | horizon | metric | relative_change | dates |
| --- | --- | --- | --- | --- | --- | --- |
| 2024H2 | current | mean | 2 | center_mae | -0.007319 | 120 |
| 2024H2 | current | mean | 2 | center_crps | -0.003837 | 120 |
| 2024H2 | current | mean | 2 | endpoint_mae | -0.005896 | 120 |
| 2024H2 | current | mean | 2 | range_mae | -0.001382 | 120 |
| 2024H2 | current | mean | 2 | range_crps | 0.000796 | 120 |
| 2024H2 | current | mean | 5 | center_mae | -0.008051 | 120 |
| 2024H2 | current | mean | 5 | center_crps | -0.008381 | 120 |
| 2024H2 | current | mean | 5 | endpoint_mae | -0.006231 | 120 |
| 2024H2 | current | mean | 5 | range_mae | -0.001235 | 120 |
| 2024H2 | current | mean | 5 | range_crps | -0.001059 | 120 |
| 2024H2 | ce | mean | 2 | center_mae | -0.002428 | 120 |
| 2024H2 | ce | mean | 2 | center_crps | 0.000772 | 120 |
| 2024H2 | ce | mean | 2 | endpoint_mae | -0.000883 | 120 |
| 2024H2 | ce | mean | 2 | range_mae | 0.001057 | 120 |
| 2024H2 | ce | mean | 2 | range_crps | 0.001284 | 120 |
| 2024H2 | ce | mean | 5 | center_mae | -0.001756 | 120 |
| 2024H2 | ce | mean | 5 | center_crps | -0.003758 | 120 |
| 2024H2 | ce | mean | 5 | endpoint_mae | -0.001889 | 120 |
| 2024H2 | ce | mean | 5 | range_mae | 0.000385 | 120 |
| 2024H2 | ce | mean | 5 | range_crps | -0.000127 | 120 |
| 2025AprJul | current | mean | 2 | center_mae | 0.008983 | 76 |
| 2025AprJul | current | mean | 2 | center_crps | 0.008612 | 76 |
| 2025AprJul | current | mean | 2 | endpoint_mae | 0.006379 | 76 |
| 2025AprJul | current | mean | 2 | range_mae | 0.008838 | 76 |
| 2025AprJul | current | mean | 2 | range_crps | 0.006593 | 76 |
| 2025AprJul | current | mean | 5 | center_mae | 0.015090 | 76 |
| 2025AprJul | current | mean | 5 | center_crps | 0.015486 | 76 |
| 2025AprJul | current | mean | 5 | endpoint_mae | 0.011504 | 76 |
| 2025AprJul | current | mean | 5 | range_mae | 0.012496 | 76 |
| 2025AprJul | current | mean | 5 | range_crps | 0.010768 | 76 |
| 2025AprJul | ce | mean | 2 | center_mae | 0.002124 | 76 |
| 2025AprJul | ce | mean | 2 | center_crps | 0.002620 | 76 |
| 2025AprJul | ce | mean | 2 | endpoint_mae | 0.002573 | 76 |
| 2025AprJul | ce | mean | 2 | range_mae | 0.004288 | 76 |
| 2025AprJul | ce | mean | 2 | range_crps | 0.002859 | 76 |
| 2025AprJul | ce | mean | 5 | center_mae | 0.005092 | 76 |
| 2025AprJul | ce | mean | 5 | center_crps | 0.004638 | 76 |
| 2025AprJul | ce | mean | 5 | endpoint_mae | 0.002895 | 76 |
| 2025AprJul | ce | mean | 5 | range_mae | 0.010505 | 76 |
| 2025AprJul | ce | mean | 5 | range_crps | 0.008766 | 76 |

## Calibration factors

| owner | seed | horizon | target | scale | rows | dates |
| --- | --- | --- | --- | --- | --- | --- |
| ce | 17 | 2 | maximum | 1.260494 | 1750 | 112 |
| ce | 17 | 2 | minimum | 1.261404 | 1750 | 112 |
| ce | 17 | 2 | range | 1.320089 | 1750 | 112 |
| ce | 17 | 5 | maximum | 1.178458 | 1696 | 112 |
| ce | 17 | 5 | minimum | 1.216434 | 1696 | 112 |
| ce | 17 | 5 | range | 1.181308 | 1696 | 112 |
| ce | 29 | 2 | maximum | 1.328821 | 1751 | 112 |
| ce | 29 | 2 | minimum | 1.321663 | 1751 | 112 |
| ce | 29 | 2 | range | 1.374577 | 1751 | 112 |
| ce | 29 | 5 | maximum | 1.254018 | 1696 | 112 |
| ce | 29 | 5 | minimum | 1.329651 | 1696 | 112 |
| ce | 29 | 5 | range | 1.231199 | 1696 | 112 |
| ce | 43 | 2 | maximum | 1.299691 | 1749 | 112 |
| ce | 43 | 2 | minimum | 1.322647 | 1749 | 112 |
| ce | 43 | 2 | range | 1.346092 | 1749 | 112 |
| ce | 43 | 5 | maximum | 1.230673 | 1695 | 112 |
| ce | 43 | 5 | minimum | 1.333018 | 1695 | 112 |
| ce | 43 | 5 | range | 1.208564 | 1695 | 112 |
| midpoint | 17 | 2 | maximum | 1.242710 | 1749 | 112 |
| midpoint | 17 | 2 | minimum | 1.236153 | 1749 | 112 |
| midpoint | 17 | 2 | range | 1.308278 | 1749 | 112 |
| midpoint | 17 | 5 | maximum | 1.187801 | 1700 | 112 |
| midpoint | 17 | 5 | minimum | 1.251575 | 1700 | 112 |
| midpoint | 17 | 5 | range | 1.162551 | 1700 | 112 |
| midpoint | 29 | 2 | maximum | 1.323395 | 1751 | 112 |
| midpoint | 29 | 2 | minimum | 1.343655 | 1751 | 112 |
| midpoint | 29 | 2 | range | 1.394970 | 1751 | 112 |
| midpoint | 29 | 5 | maximum | 1.270487 | 1698 | 112 |
| midpoint | 29 | 5 | minimum | 1.339284 | 1698 | 112 |
| midpoint | 29 | 5 | range | 1.252059 | 1698 | 112 |
| midpoint | 43 | 2 | maximum | 1.324491 | 1750 | 112 |
| midpoint | 43 | 2 | minimum | 1.302246 | 1750 | 112 |
| midpoint | 43 | 2 | range | 1.369805 | 1750 | 112 |
| midpoint | 43 | 5 | maximum | 1.214769 | 1693 | 112 |
| midpoint | 43 | 5 | minimum | 1.345872 | 1693 | 112 |
| midpoint | 43 | 5 | range | 1.211828 | 1693 | 112 |

## Forecast availability

| owner | seed | horizon | inputs | known | usable | legal_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| current | 17 | 2 | 2432 | 2389 | 2384 | 0.969360 |
| current | 17 | 5 | 2432 | 2324 | 2304 | 0.923867 |
| current | 29 | 2 | 2432 | 2389 | 2384 | 0.968776 |
| current | 29 | 5 | 2432 | 2324 | 2303 | 0.922338 |
| current | 43 | 2 | 2432 | 2389 | 2381 | 0.966874 |
| current | 43 | 5 | 2432 | 2324 | 2302 | 0.919479 |
| ce | 17 | 2 | 2432 | 2389 | 2385 | 0.970330 |
| ce | 17 | 5 | 2432 | 2324 | 2304 | 0.926938 |
| ce | 29 | 2 | 2432 | 2389 | 2384 | 0.972476 |
| ce | 29 | 5 | 2432 | 2324 | 2305 | 0.932521 |
| ce | 43 | 2 | 2432 | 2389 | 2382 | 0.969373 |
| ce | 43 | 5 | 2432 | 2324 | 2306 | 0.921933 |
| midpoint | 17 | 2 | 2432 | 2389 | 2382 | 0.968891 |
| midpoint | 17 | 5 | 2432 | 2324 | 2306 | 0.922447 |
| midpoint | 29 | 2 | 2432 | 2389 | 2384 | 0.973048 |
| midpoint | 29 | 5 | 2432 | 2324 | 2305 | 0.934410 |
| midpoint | 43 | 2 | 2432 | 2389 | 2382 | 0.969476 |
| midpoint | 43 | 5 | 2432 | 2324 | 2305 | 0.921130 |

## Independent verification

| check | value |
| --- | --- |
| passed | True |
| at | 2026-09-16T05:03:05.543951+00:00 |
| factors | 36 |
| replays | 15 |
| replayed_paths | 3840 |
| forecast_chunks | 8160 |
| raw_and_calibrated_rows | 253128 |
| common_rows | 252774 |
| midpoint_rows | 42129 |
| paired_estimates | 608 |
| medians_and_mae_exactly_preserved | True |
| calibrated_crps_not_claimed | True |
| weights_and_sources_unchanged | True |
| calibration_frozen_before_transfer | True |
| sealed_holdout_opened | False |
