# Sampling replication: detailed evidence

Draw 0 is the equal-draw score average; seed 0 is the equal-training-seed score average. No prices or paths are pooled. Negative error changes are better.

## Input support

| cohort | draw | seed | horizon | rows |
| --- | --- | --- | --- | --- |
| all27 | 17 | 17 | 2 | 2376 |
| all27 | 17 | 17 | 5 | 2287 |
| all27 | 17 | 29 | 2 | 2376 |
| all27 | 17 | 29 | 5 | 2287 |
| all27 | 17 | 43 | 2 | 2376 |
| all27 | 17 | 43 | 5 | 2287 |
| all27 | 100017 | 17 | 2 | 2376 |
| all27 | 100017 | 17 | 5 | 2287 |
| all27 | 100017 | 29 | 2 | 2376 |
| all27 | 100017 | 29 | 5 | 2287 |
| all27 | 100017 | 43 | 2 | 2376 |
| all27 | 100017 | 43 | 5 | 2287 |
| all27 | 200017 | 17 | 2 | 2376 |
| all27 | 200017 | 17 | 5 | 2287 |
| all27 | 200017 | 29 | 2 | 2376 |
| all27 | 200017 | 29 | 5 | 2287 |
| all27 | 200017 | 43 | 2 | 2376 |
| all27 | 200017 | 43 | 5 | 2287 |
| within_draw_seed | 17 | 17 | 2 | 2381 |
| within_draw_seed | 17 | 17 | 5 | 2301 |
| within_draw_seed | 17 | 29 | 2 | 2381 |
| within_draw_seed | 17 | 29 | 5 | 2300 |
| within_draw_seed | 17 | 43 | 2 | 2381 |
| within_draw_seed | 17 | 43 | 5 | 2299 |
| within_draw_seed | 100017 | 17 | 2 | 2381 |
| within_draw_seed | 100017 | 17 | 5 | 2297 |
| within_draw_seed | 100017 | 29 | 2 | 2382 |
| within_draw_seed | 100017 | 29 | 5 | 2293 |
| within_draw_seed | 100017 | 43 | 2 | 2380 |
| within_draw_seed | 100017 | 43 | 5 | 2297 |
| within_draw_seed | 200017 | 17 | 2 | 2382 |
| within_draw_seed | 200017 | 17 | 5 | 2300 |
| within_draw_seed | 200017 | 29 | 2 | 2382 |
| within_draw_seed | 200017 | 29 | 5 | 2299 |
| within_draw_seed | 200017 | 43 | 2 | 2380 |
| within_draw_seed | 200017 | 43 | 5 | 2300 |

## All primary comparisons

| cohort | draw | seed | horizon | comparison | metric | baseline | candidate | delta | ci_low | ci_high | relative_change_pct | dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all27 | 0 | 0 | 2 | ce_vs_current | endpoint_mae | 2.7314 | 2.7416 | 0.0102 | -0.0090 | 0.0321 | 0.3734 | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | high_mae | 3.1412 | 3.1339 | -0.0072 | -0.0222 | 0.0071 | -0.2301 | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | low_mae | 2.3216 | 2.3493 | 0.0276 | -0.0020 | 0.0658 | 1.1899 | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | center_bias | -0.6491 | -0.7075 | -0.0584 | -0.1211 | -0.0072 | — | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | half_width_bias | -0.5681 | -0.5458 | 0.0224 | 0.0085 | 0.0396 | — | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | midpoint_mae | 2.5426 | 2.5537 | 0.0112 | -0.0088 | 0.0345 | 0.4386 | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | midpoint_crps | 1.9175 | 1.9251 | 0.0076 | -0.0070 | 0.0246 | 0.3986 | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | range_mae | 2.7881 | 2.7969 | 0.0088 | -0.0083 | 0.0260 | 0.3159 | 76 |
| all27 | 0 | 0 | 2 | ce_vs_current | legal_fraction | 0.9730 | 0.9753 | 0.0022 | 0.0016 | 0.0028 | 0.2288 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | endpoint_mae | 2.7416 | 2.7497 | 0.0081 | -0.0029 | 0.0213 | 0.2954 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | high_mae | 3.1339 | 3.1382 | 0.0042 | -0.0043 | 0.0133 | 0.1347 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | low_mae | 2.3493 | 2.3612 | 0.0120 | -0.0029 | 0.0309 | 0.5098 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | center_bias | -0.7075 | -0.7251 | -0.0176 | -0.0437 | 0.0153 | — | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | half_width_bias | -0.5458 | -0.5470 | -0.0012 | -0.0072 | 0.0055 | — | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | midpoint_mae | 2.5537 | 2.5613 | 0.0076 | -0.0044 | 0.0211 | 0.2963 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | midpoint_crps | 1.9251 | 1.9318 | 0.0066 | -0.0010 | 0.0156 | 0.3448 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | range_mae | 2.7969 | 2.8094 | 0.0125 | -0.0034 | 0.0285 | 0.4472 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_ce | legal_fraction | 0.9753 | 0.9750 | -0.0002 | -0.0005 | 0.0000 | -0.0245 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | endpoint_mae | 2.7314 | 2.7497 | 0.0183 | -0.0018 | 0.0437 | 0.6699 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | high_mae | 3.1412 | 3.1382 | -0.0030 | -0.0147 | 0.0093 | -0.0957 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | low_mae | 2.3216 | 2.3612 | 0.0396 | 0.0048 | 0.0854 | 1.7058 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | center_bias | -0.6491 | -0.7251 | -0.0760 | -0.1234 | -0.0412 | — | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | half_width_bias | -0.5681 | -0.5470 | 0.0212 | 0.0067 | 0.0375 | — | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | midpoint_mae | 2.5426 | 2.5613 | 0.0187 | -0.0049 | 0.0489 | 0.7362 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | midpoint_crps | 1.9175 | 1.9318 | 0.0143 | -0.0003 | 0.0335 | 0.7449 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | range_mae | 2.7881 | 2.8094 | 0.0213 | 0.0026 | 0.0455 | 0.7645 | 76 |
| all27 | 0 | 0 | 2 | midpoint_vs_current | legal_fraction | 0.9730 | 0.9750 | 0.0020 | 0.0013 | 0.0027 | 0.2043 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | endpoint_mae | 4.2731 | 4.3135 | 0.0404 | -0.0116 | 0.0979 | 0.9450 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | high_mae | 5.0499 | 5.0490 | -0.0009 | -0.0247 | 0.0236 | -0.0186 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | low_mae | 3.4963 | 3.5780 | 0.0817 | -0.0066 | 0.1911 | 2.3368 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | center_bias | -1.6792 | -1.8158 | -0.1366 | -0.2700 | -0.0228 | — | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | half_width_bias | -0.8720 | -0.8367 | 0.0353 | 0.0077 | 0.0665 | — | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | midpoint_mae | 3.9752 | 4.0184 | 0.0433 | -0.0062 | 0.1040 | 1.0881 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | midpoint_crps | 2.9730 | 3.0031 | 0.0301 | -0.0064 | 0.0758 | 1.0112 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | range_mae | 4.8669 | 4.8874 | 0.0205 | -0.0069 | 0.0497 | 0.4213 | 76 |
| all27 | 0 | 0 | 5 | ce_vs_current | legal_fraction | 0.9342 | 0.9393 | 0.0051 | 0.0036 | 0.0069 | 0.5472 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | endpoint_mae | 4.3135 | 4.3281 | 0.0146 | -0.0126 | 0.0416 | 0.3376 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | high_mae | 5.0490 | 5.0515 | 0.0025 | -0.0190 | 0.0249 | 0.0495 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | low_mae | 3.5780 | 3.6046 | 0.0266 | -0.0142 | 0.0654 | 0.7440 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | center_bias | -1.8158 | -1.8693 | -0.0536 | -0.1141 | 0.0164 | — | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | half_width_bias | -0.8367 | -0.8388 | -0.0021 | -0.0200 | 0.0149 | — | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | midpoint_mae | 4.0184 | 4.0347 | 0.0163 | -0.0115 | 0.0452 | 0.4048 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | midpoint_crps | 3.0031 | 3.0191 | 0.0160 | -0.0013 | 0.0333 | 0.5325 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | range_mae | 4.8874 | 4.9207 | 0.0333 | 0.0051 | 0.0596 | 0.6806 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_ce | legal_fraction | 0.9393 | 0.9385 | -0.0008 | -0.0013 | -0.0002 | -0.0877 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | endpoint_mae | 4.2731 | 4.3281 | 0.0549 | 0.0019 | 0.1180 | 1.2858 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | high_mae | 5.0499 | 5.0515 | 0.0016 | -0.0292 | 0.0277 | 0.0310 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | low_mae | 3.4963 | 3.6046 | 0.1083 | 0.0217 | 0.2176 | 3.0982 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | center_bias | -1.6792 | -1.8693 | -0.1901 | -0.2787 | -0.1220 | — | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | half_width_bias | -0.8720 | -0.8388 | 0.0332 | 0.0033 | 0.0632 | — | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | midpoint_mae | 3.9752 | 4.0347 | 0.0595 | 0.0037 | 0.1291 | 1.4973 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | midpoint_crps | 2.9730 | 3.0191 | 0.0461 | 0.0093 | 0.0921 | 1.5492 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | range_mae | 4.8669 | 4.9207 | 0.0538 | 0.0220 | 0.0936 | 1.1048 | 76 |
| all27 | 0 | 0 | 5 | midpoint_vs_current | legal_fraction | 0.9342 | 0.9385 | 0.0043 | 0.0028 | 0.0061 | 0.4590 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | endpoint_mae | 2.7046 | 2.6947 | -0.0099 | -0.0405 | 0.0224 | -0.3643 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | high_mae | 3.1360 | 3.1312 | -0.0048 | -0.0328 | 0.0204 | -0.1543 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | low_mae | 2.2731 | 2.2582 | -0.0149 | -0.0528 | 0.0281 | -0.6541 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | center_bias | -0.6431 | -0.5988 | 0.0443 | -0.0116 | 0.0953 | — | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | half_width_bias | -0.5888 | -0.6024 | -0.0136 | -0.0293 | 0.0021 | — | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | midpoint_mae | 2.5177 | 2.5085 | -0.0091 | -0.0411 | 0.0249 | -0.3631 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | midpoint_crps | 1.9032 | 1.8921 | -0.0111 | -0.0341 | 0.0133 | -0.5841 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | range_mae | 2.7601 | 2.7432 | -0.0169 | -0.0308 | -0.0043 | -0.6112 | 76 |
| all27 | 0 | 17 | 2 | ce_vs_current | legal_fraction | 0.9738 | 0.9750 | 0.0013 | 0.0004 | 0.0022 | 0.1289 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | endpoint_mae | 2.6947 | 2.6916 | -0.0031 | -0.0306 | 0.0204 | -0.1146 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | high_mae | 3.1312 | 3.1314 | 0.0002 | -0.0208 | 0.0214 | 0.0060 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | low_mae | 2.2582 | 2.2518 | -0.0064 | -0.0460 | 0.0238 | -0.2817 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | center_bias | -0.5988 | -0.5902 | 0.0086 | -0.0582 | 0.0824 | — | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | half_width_bias | -0.6024 | -0.6071 | -0.0047 | -0.0253 | 0.0152 | — | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | midpoint_mae | 2.5085 | 2.4999 | -0.0086 | -0.0417 | 0.0193 | -0.3429 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | midpoint_crps | 1.8921 | 1.8926 | 0.0005 | -0.0198 | 0.0169 | 0.0270 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | range_mae | 2.7432 | 2.7586 | 0.0154 | 0.0013 | 0.0300 | 0.5602 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_ce | legal_fraction | 0.9750 | 0.9737 | -0.0014 | -0.0018 | -0.0009 | -0.1398 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | endpoint_mae | 2.7046 | 2.6916 | -0.0129 | -0.0418 | 0.0157 | -0.4784 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | high_mae | 3.1360 | 3.1314 | -0.0047 | -0.0335 | 0.0241 | -0.1483 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | low_mae | 2.2731 | 2.2518 | -0.0212 | -0.0565 | 0.0092 | -0.9339 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | center_bias | -0.6431 | -0.5902 | 0.0529 | 0.0082 | 0.1129 | — | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | half_width_bias | -0.5888 | -0.6071 | -0.0183 | -0.0396 | -0.0006 | — | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | midpoint_mae | 2.5177 | 2.4999 | -0.0177 | -0.0557 | 0.0190 | -0.7048 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | midpoint_crps | 1.9032 | 1.8926 | -0.0106 | -0.0308 | 0.0090 | -0.5572 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | range_mae | 2.7601 | 2.7586 | -0.0015 | -0.0165 | 0.0149 | -0.0544 | 76 |
| all27 | 0 | 17 | 2 | midpoint_vs_current | legal_fraction | 0.9738 | 0.9737 | -0.0001 | -0.0010 | 0.0010 | -0.0111 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | endpoint_mae | 4.2015 | 4.1721 | -0.0294 | -0.0815 | 0.0220 | -0.6998 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | high_mae | 5.0592 | 5.0386 | -0.0206 | -0.0608 | 0.0193 | -0.4068 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | low_mae | 3.3438 | 3.3056 | -0.0382 | -0.1089 | 0.0340 | -1.1433 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | center_bias | -1.6164 | -1.5209 | 0.0954 | -0.0159 | 0.1997 | — | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | half_width_bias | -0.9514 | -0.9938 | -0.0424 | -0.0636 | -0.0177 | — | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | midpoint_mae | 3.9075 | 3.8890 | -0.0184 | -0.0730 | 0.0378 | -0.4720 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | midpoint_crps | 2.9259 | 2.9104 | -0.0155 | -0.0502 | 0.0216 | -0.5305 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | range_mae | 4.7958 | 4.7376 | -0.0583 | -0.0938 | -0.0258 | -1.2150 | 76 |
| all27 | 0 | 17 | 5 | ce_vs_current | legal_fraction | 0.9359 | 0.9389 | 0.0030 | 0.0013 | 0.0050 | 0.3204 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | endpoint_mae | 4.1721 | 4.1568 | -0.0153 | -0.0813 | 0.0386 | -0.3670 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | high_mae | 5.0386 | 5.0420 | 0.0034 | -0.0330 | 0.0405 | 0.0679 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | low_mae | 3.3056 | 3.2716 | -0.0340 | -0.1361 | 0.0467 | -1.0299 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | center_bias | -1.5209 | -1.5292 | -0.0083 | -0.1588 | 0.1649 | — | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | half_width_bias | -0.9938 | -1.0057 | -0.0119 | -0.0662 | 0.0367 | — | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | midpoint_mae | 3.8890 | 3.8795 | -0.0095 | -0.0720 | 0.0428 | -0.2446 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | midpoint_crps | 2.9104 | 2.9078 | -0.0026 | -0.0459 | 0.0331 | -0.0885 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | range_mae | 4.7376 | 4.7705 | 0.0330 | -0.0228 | 0.0774 | 0.6956 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_ce | legal_fraction | 0.9389 | 0.9352 | -0.0036 | -0.0048 | -0.0025 | -0.3874 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | endpoint_mae | 4.2015 | 4.1568 | -0.0447 | -0.1032 | 0.0068 | -1.0642 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | high_mae | 5.0592 | 5.0420 | -0.0172 | -0.0584 | 0.0268 | -0.3391 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | low_mae | 3.3438 | 3.2716 | -0.0723 | -0.1588 | -0.0058 | -2.1614 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | center_bias | -1.6164 | -1.5292 | 0.0872 | 0.0044 | 0.1926 | — | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | half_width_bias | -0.9514 | -1.0057 | -0.0543 | -0.1014 | -0.0124 | — | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | midpoint_mae | 3.9075 | 3.8795 | -0.0280 | -0.0869 | 0.0250 | -0.7154 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | midpoint_crps | 2.9259 | 2.9078 | -0.0181 | -0.0502 | 0.0100 | -0.6186 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | range_mae | 4.7958 | 4.7705 | -0.0253 | -0.0922 | 0.0352 | -0.5278 | 76 |
| all27 | 0 | 17 | 5 | midpoint_vs_current | legal_fraction | 0.9359 | 0.9352 | -0.0006 | -0.0026 | 0.0015 | -0.0683 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | endpoint_mae | 2.7208 | 2.6962 | -0.0246 | -0.0428 | -0.0047 | -0.9047 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | high_mae | 3.1558 | 3.1239 | -0.0319 | -0.0505 | -0.0119 | -1.0110 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | low_mae | 2.2858 | 2.2685 | -0.0173 | -0.0469 | 0.0100 | -0.7579 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | center_bias | -0.5010 | -0.6558 | -0.1548 | -0.1834 | -0.1255 | — | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | half_width_bias | -0.5506 | -0.5974 | -0.0468 | -0.0765 | -0.0197 | — | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | midpoint_mae | 2.5265 | 2.5098 | -0.0167 | -0.0333 | 0.0039 | -0.6616 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | midpoint_crps | 1.8983 | 1.8917 | -0.0066 | -0.0212 | 0.0077 | -0.3466 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | range_mae | 2.7917 | 2.7498 | -0.0420 | -0.0901 | -0.0013 | -1.5032 | 76 |
| all27 | 0 | 29 | 2 | ce_vs_current | legal_fraction | 0.9735 | 0.9767 | 0.0033 | 0.0025 | 0.0041 | 0.3345 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | endpoint_mae | 2.6962 | 2.6693 | -0.0269 | -0.0503 | -0.0073 | -0.9988 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | high_mae | 3.1239 | 3.1162 | -0.0077 | -0.0226 | 0.0083 | -0.2457 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | low_mae | 2.2685 | 2.2223 | -0.0462 | -0.0881 | -0.0168 | -2.0359 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | center_bias | -0.6558 | -0.5885 | 0.0673 | 0.0255 | 0.1224 | — | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | half_width_bias | -0.5974 | -0.6321 | -0.0347 | -0.0503 | -0.0198 | — | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | midpoint_mae | 2.5098 | 2.4830 | -0.0268 | -0.0529 | -0.0027 | -1.0670 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | midpoint_crps | 1.8917 | 1.8739 | -0.0178 | -0.0346 | -0.0045 | -0.9429 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | range_mae | 2.7498 | 2.7231 | -0.0267 | -0.0442 | -0.0095 | -0.9715 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_ce | legal_fraction | 0.9767 | 0.9774 | 0.0007 | 0.0001 | 0.0012 | 0.0687 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | endpoint_mae | 2.7208 | 2.6693 | -0.0515 | -0.0868 | -0.0229 | -1.8945 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | high_mae | 3.1558 | 3.1162 | -0.0396 | -0.0564 | -0.0235 | -1.2543 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | low_mae | 2.2858 | 2.2223 | -0.0635 | -0.1247 | -0.0140 | -2.7785 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | center_bias | -0.5010 | -0.5885 | -0.0875 | -0.1269 | -0.0435 | — | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | half_width_bias | -0.5506 | -0.6321 | -0.0815 | -0.1242 | -0.0453 | — | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | midpoint_mae | 2.5265 | 2.4830 | -0.0435 | -0.0736 | -0.0169 | -1.7215 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | midpoint_crps | 1.8983 | 1.8739 | -0.0244 | -0.0472 | -0.0046 | -1.2863 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | range_mae | 2.7917 | 2.7231 | -0.0687 | -0.1242 | -0.0234 | -2.4601 | 76 |
| all27 | 0 | 29 | 2 | midpoint_vs_current | legal_fraction | 0.9735 | 0.9774 | 0.0039 | 0.0030 | 0.0048 | 0.4035 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | endpoint_mae | 4.2230 | 4.1786 | -0.0444 | -0.1166 | 0.0261 | -1.0504 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | high_mae | 5.0478 | 5.0070 | -0.0407 | -0.1016 | 0.0166 | -0.8071 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | low_mae | 3.3981 | 3.3502 | -0.0480 | -0.1720 | 0.0630 | -1.4118 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | center_bias | -1.3519 | -1.6992 | -0.3473 | -0.4027 | -0.2800 | — | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | half_width_bias | -0.8188 | -1.0146 | -0.1959 | -0.2955 | -0.1119 | — | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | midpoint_mae | 3.9119 | 3.8875 | -0.0244 | -0.0963 | 0.0485 | -0.6231 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | midpoint_crps | 2.9222 | 2.9150 | -0.0072 | -0.0471 | 0.0309 | -0.2470 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | range_mae | 4.8592 | 4.7258 | -0.1333 | -0.2850 | 0.0054 | -2.7442 | 76 |
| all27 | 0 | 29 | 5 | ce_vs_current | legal_fraction | 0.9353 | 0.9446 | 0.0093 | 0.0077 | 0.0111 | 0.9970 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | endpoint_mae | 4.1786 | 4.0894 | -0.0892 | -0.1670 | -0.0282 | -2.1353 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | high_mae | 5.0070 | 4.9794 | -0.0276 | -0.0625 | 0.0002 | -0.5520 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | low_mae | 3.3502 | 3.1994 | -0.1508 | -0.2884 | -0.0459 | -4.5017 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | center_bias | -1.6992 | -1.5263 | 0.1729 | 0.0815 | 0.2959 | — | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | half_width_bias | -1.0146 | -1.1098 | -0.0951 | -0.1347 | -0.0569 | — | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | midpoint_mae | 3.8875 | 3.8057 | -0.0819 | -0.1700 | -0.0190 | -2.1059 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | midpoint_crps | 2.9150 | 2.8614 | -0.0536 | -0.1111 | -0.0119 | -1.8391 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | range_mae | 4.7258 | 4.6655 | -0.0603 | -0.1084 | -0.0227 | -1.2752 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_ce | legal_fraction | 0.9446 | 0.9469 | 0.0023 | 0.0011 | 0.0033 | 0.2419 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | endpoint_mae | 4.2230 | 4.0894 | -0.1336 | -0.2685 | -0.0316 | -3.1633 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | high_mae | 5.0478 | 4.9794 | -0.0684 | -0.1289 | -0.0132 | -1.3547 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | low_mae | 3.3981 | 3.1994 | -0.1988 | -0.4356 | -0.0195 | -5.8500 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | center_bias | -1.3519 | -1.5263 | -0.1744 | -0.2791 | -0.0515 | — | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | half_width_bias | -0.8188 | -1.1098 | -0.2910 | -0.4238 | -0.1814 | — | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | midpoint_mae | 3.9119 | 3.8057 | -0.1062 | -0.2384 | -0.0030 | -2.7160 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | midpoint_crps | 2.9222 | 2.8614 | -0.0608 | -0.1451 | 0.0049 | -2.0815 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | range_mae | 4.8592 | 4.6655 | -0.1936 | -0.3828 | -0.0358 | -3.9845 | 76 |
| all27 | 0 | 29 | 5 | midpoint_vs_current | legal_fraction | 0.9353 | 0.9469 | 0.0116 | 0.0102 | 0.0131 | 1.2412 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | endpoint_mae | 2.7688 | 2.8339 | 0.0651 | 0.0218 | 0.1173 | 2.3499 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | high_mae | 3.1316 | 3.1467 | 0.0151 | -0.0087 | 0.0400 | 0.4809 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | low_mae | 2.4060 | 2.5211 | 0.1151 | 0.0355 | 0.2176 | 4.7825 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | center_bias | -0.8031 | -0.8679 | -0.0648 | -0.1952 | 0.0481 | — | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | half_width_bias | -0.5650 | -0.4376 | 0.1275 | 0.0878 | 0.1800 | — | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | midpoint_mae | 2.5835 | 2.6428 | 0.0593 | 0.0121 | 0.1156 | 2.2958 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | midpoint_crps | 1.9510 | 1.9916 | 0.0406 | 0.0057 | 0.0802 | 2.0825 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | range_mae | 2.8125 | 2.8978 | 0.0853 | 0.0130 | 0.1670 | 3.0314 | 76 |
| all27 | 0 | 43 | 2 | ce_vs_current | legal_fraction | 0.9718 | 0.9740 | 0.0022 | 0.0013 | 0.0030 | 0.2231 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | endpoint_mae | 2.8339 | 2.8882 | 0.0543 | 0.0149 | 0.1040 | 1.9166 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | high_mae | 3.1467 | 3.1668 | 0.0202 | -0.0035 | 0.0407 | 0.6404 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | low_mae | 2.5211 | 2.6095 | 0.0885 | 0.0201 | 0.1734 | 3.5096 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | center_bias | -0.8679 | -0.9967 | -0.1287 | -0.1793 | -0.0814 | — | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | half_width_bias | -0.4376 | -0.4018 | 0.0358 | 0.0100 | 0.0643 | — | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | midpoint_mae | 2.6428 | 2.7009 | 0.0581 | 0.0154 | 0.1100 | 2.1976 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | midpoint_crps | 1.9916 | 2.0288 | 0.0372 | 0.0100 | 0.0708 | 1.8699 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | range_mae | 2.8978 | 2.9467 | 0.0489 | 0.0099 | 0.0957 | 1.6864 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_ce | legal_fraction | 0.9740 | 0.9740 | -0.0000 | -0.0007 | 0.0006 | -0.0026 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | endpoint_mae | 2.7688 | 2.8882 | 0.1194 | 0.0445 | 0.2166 | 4.3116 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | high_mae | 3.1316 | 3.1668 | 0.0352 | 0.0107 | 0.0594 | 1.1244 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | low_mae | 2.4060 | 2.6095 | 0.2035 | 0.0670 | 0.3846 | 8.4600 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | center_bias | -0.8031 | -0.9967 | -0.1935 | -0.3560 | -0.0573 | — | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | half_width_bias | -0.5650 | -0.4018 | 0.1632 | 0.1042 | 0.2375 | — | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | midpoint_mae | 2.5835 | 2.7009 | 0.1174 | 0.0374 | 0.2191 | 4.5439 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | midpoint_crps | 1.9510 | 2.0288 | 0.0779 | 0.0250 | 0.1484 | 3.9913 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | range_mae | 2.8125 | 2.9467 | 0.1341 | 0.0456 | 0.2483 | 4.7689 | 76 |
| all27 | 0 | 43 | 2 | midpoint_vs_current | legal_fraction | 0.9718 | 0.9740 | 0.0021 | 0.0013 | 0.0029 | 0.2205 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | endpoint_mae | 4.3949 | 4.5898 | 0.1949 | 0.0545 | 0.3718 | 4.4348 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | high_mae | 5.0429 | 5.1014 | 0.0585 | 0.0150 | 0.0980 | 1.1601 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | low_mae | 3.7470 | 4.0783 | 0.3313 | 0.0575 | 0.6793 | 8.8421 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | center_bias | -2.0694 | -2.2272 | -0.1578 | -0.4557 | 0.1122 | — | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | half_width_bias | -0.8459 | -0.5017 | 0.3442 | 0.2485 | 0.4693 | — | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | midpoint_mae | 4.1062 | 4.2787 | 0.1726 | 0.0394 | 0.3465 | 4.2031 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | midpoint_crps | 3.0711 | 3.1840 | 0.1129 | 0.0150 | 0.2405 | 3.6774 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | range_mae | 4.9457 | 5.1989 | 0.2531 | 0.0590 | 0.4776 | 5.1181 | 76 |
| all27 | 0 | 43 | 5 | ce_vs_current | legal_fraction | 0.9315 | 0.9345 | 0.0030 | 0.0007 | 0.0056 | 0.3234 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | endpoint_mae | 4.5898 | 4.7381 | 0.1482 | 0.0440 | 0.2754 | 3.2294 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | high_mae | 5.1014 | 5.1331 | 0.0317 | -0.0145 | 0.0769 | 0.6218 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | low_mae | 4.0783 | 4.3430 | 0.2647 | 0.0897 | 0.4857 | 6.4910 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | center_bias | -2.2272 | -2.5525 | -0.3253 | -0.4622 | -0.1861 | — | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | half_width_bias | -0.5017 | -0.4009 | 0.1008 | 0.0209 | 0.1947 | — | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | midpoint_mae | 4.2787 | 4.4189 | 0.1402 | 0.0271 | 0.2816 | 3.2761 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | midpoint_crps | 3.1840 | 3.2882 | 0.1042 | 0.0307 | 0.1938 | 3.2714 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | range_mae | 5.1989 | 5.3260 | 0.1271 | 0.0334 | 0.2532 | 2.4448 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_ce | legal_fraction | 0.9345 | 0.9334 | -0.0011 | -0.0028 | 0.0006 | -0.1198 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | endpoint_mae | 4.3949 | 4.7381 | 0.3431 | 0.1291 | 0.6278 | 7.8074 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | high_mae | 5.0429 | 5.1331 | 0.0902 | 0.0396 | 0.1474 | 1.7892 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | low_mae | 3.7470 | 4.3430 | 0.5960 | 0.2001 | 1.1306 | 15.9071 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | center_bias | -2.0694 | -2.5525 | -0.4831 | -0.8638 | -0.1658 | — | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | half_width_bias | -0.8459 | -0.4009 | 0.4449 | 0.2865 | 0.6531 | — | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | midpoint_mae | 4.1062 | 4.4189 | 0.3128 | 0.0912 | 0.6140 | 7.6169 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | midpoint_crps | 3.0711 | 3.2882 | 0.2171 | 0.0671 | 0.4267 | 7.0691 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | range_mae | 4.9457 | 5.3260 | 0.3802 | 0.1336 | 0.7179 | 7.6880 | 76 |
| all27 | 0 | 43 | 5 | midpoint_vs_current | legal_fraction | 0.9315 | 0.9334 | 0.0019 | 0.0000 | 0.0039 | 0.2032 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | endpoint_mae | 2.7367 | 2.7455 | 0.0088 | -0.0111 | 0.0311 | 0.3216 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | high_mae | 3.1441 | 3.1344 | -0.0096 | -0.0264 | 0.0071 | -0.3066 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | low_mae | 2.3293 | 2.3566 | 0.0272 | -0.0022 | 0.0623 | 1.1695 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | center_bias | -0.6491 | -0.7126 | -0.0635 | -0.1202 | -0.0201 | — | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | half_width_bias | -0.5679 | -0.5478 | 0.0201 | 0.0035 | 0.0384 | — | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | midpoint_mae | 2.5439 | 2.5600 | 0.0161 | -0.0062 | 0.0414 | 0.6338 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | midpoint_crps | 1.9211 | 1.9322 | 0.0111 | -0.0059 | 0.0310 | 0.5799 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | range_mae | 2.7843 | 2.7967 | 0.0124 | -0.0089 | 0.0301 | 0.4468 | 76 |
| all27 | 17 | 0 | 2 | ce_vs_current | legal_fraction | 0.9730 | 0.9754 | 0.0025 | 0.0017 | 0.0033 | 0.2534 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | endpoint_mae | 2.7455 | 2.7534 | 0.0079 | -0.0106 | 0.0276 | 0.2887 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | high_mae | 3.1344 | 3.1416 | 0.0071 | -0.0137 | 0.0261 | 0.2279 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | low_mae | 2.3566 | 2.3653 | 0.0087 | -0.0109 | 0.0332 | 0.3695 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | center_bias | -0.7126 | -0.7254 | -0.0128 | -0.0365 | 0.0175 | — | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | half_width_bias | -0.5478 | -0.5512 | -0.0034 | -0.0113 | 0.0038 | — | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | midpoint_mae | 2.5600 | 2.5675 | 0.0075 | -0.0136 | 0.0290 | 0.2921 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | midpoint_crps | 1.9322 | 1.9373 | 0.0050 | -0.0062 | 0.0176 | 0.2601 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | range_mae | 2.7967 | 2.8088 | 0.0121 | -0.0091 | 0.0357 | 0.4329 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_ce | legal_fraction | 0.9754 | 0.9753 | -0.0002 | -0.0006 | 0.0004 | -0.0159 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | endpoint_mae | 2.7367 | 2.7534 | 0.0167 | -0.0048 | 0.0431 | 0.6112 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | high_mae | 3.1441 | 3.1416 | -0.0025 | -0.0190 | 0.0132 | -0.0793 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | low_mae | 2.3293 | 2.3653 | 0.0359 | 0.0032 | 0.0803 | 1.5433 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | center_bias | -0.6491 | -0.7254 | -0.0763 | -0.1225 | -0.0426 | — | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | half_width_bias | -0.5679 | -0.5512 | 0.0167 | 0.0016 | 0.0324 | — | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | midpoint_mae | 2.5439 | 2.5675 | 0.0236 | -0.0016 | 0.0567 | 0.9278 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | midpoint_crps | 1.9211 | 1.9373 | 0.0162 | 0.0003 | 0.0355 | 0.8415 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | range_mae | 2.7843 | 2.8088 | 0.0245 | 0.0044 | 0.0497 | 0.8816 | 76 |
| all27 | 17 | 0 | 2 | midpoint_vs_current | legal_fraction | 0.9730 | 0.9753 | 0.0023 | 0.0016 | 0.0031 | 0.2375 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | endpoint_mae | 4.2844 | 4.3166 | 0.0322 | -0.0244 | 0.0980 | 0.7512 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | high_mae | 5.0668 | 5.0525 | -0.0143 | -0.0419 | 0.0146 | -0.2814 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | low_mae | 3.5020 | 3.5807 | 0.0786 | -0.0169 | 0.1940 | 2.2452 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | center_bias | -1.6767 | -1.8267 | -0.1500 | -0.2734 | -0.0438 | — | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | half_width_bias | -0.8738 | -0.8369 | 0.0368 | 0.0030 | 0.0730 | — | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | midpoint_mae | 3.9867 | 4.0215 | 0.0349 | -0.0225 | 0.1057 | 0.8753 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | midpoint_crps | 2.9797 | 3.0082 | 0.0285 | -0.0101 | 0.0777 | 0.9563 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | range_mae | 4.8724 | 4.8785 | 0.0061 | -0.0316 | 0.0484 | 0.1244 | 76 |
| all27 | 17 | 0 | 5 | ce_vs_current | legal_fraction | 0.9342 | 0.9392 | 0.0050 | 0.0033 | 0.0073 | 0.5398 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | endpoint_mae | 4.3166 | 4.3304 | 0.0138 | -0.0205 | 0.0456 | 0.3196 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | high_mae | 5.0525 | 5.0601 | 0.0076 | -0.0207 | 0.0329 | 0.1503 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | low_mae | 3.5807 | 3.6007 | 0.0200 | -0.0298 | 0.0649 | 0.5585 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | center_bias | -1.8267 | -1.8652 | -0.0385 | -0.0945 | 0.0268 | — | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | half_width_bias | -0.8369 | -0.8396 | -0.0027 | -0.0267 | 0.0209 | — | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | midpoint_mae | 4.0215 | 4.0418 | 0.0203 | -0.0109 | 0.0514 | 0.5038 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | midpoint_crps | 3.0082 | 3.0228 | 0.0146 | -0.0058 | 0.0346 | 0.4852 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | range_mae | 4.8785 | 4.9319 | 0.0534 | 0.0234 | 0.0839 | 1.0955 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_ce | legal_fraction | 0.9392 | 0.9382 | -0.0011 | -0.0021 | -0.0000 | -0.1146 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | endpoint_mae | 4.2844 | 4.3304 | 0.0460 | -0.0072 | 0.1076 | 1.0732 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | high_mae | 5.0668 | 5.0601 | -0.0067 | -0.0422 | 0.0229 | -0.1316 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | low_mae | 3.5020 | 3.6007 | 0.0986 | 0.0114 | 0.2076 | 2.8162 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | center_bias | -1.6767 | -1.8652 | -0.1885 | -0.2759 | -0.1227 | — | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | half_width_bias | -0.8738 | -0.8396 | 0.0342 | 0.0011 | 0.0666 | — | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | midpoint_mae | 3.9867 | 4.0418 | 0.0552 | -0.0027 | 0.1295 | 1.3835 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | midpoint_crps | 2.9797 | 3.0228 | 0.0431 | 0.0070 | 0.0874 | 1.4461 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | range_mae | 4.8724 | 4.9319 | 0.0595 | 0.0190 | 0.1095 | 1.2213 | 76 |
| all27 | 17 | 0 | 5 | midpoint_vs_current | legal_fraction | 0.9342 | 0.9382 | 0.0040 | 0.0022 | 0.0059 | 0.4245 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | endpoint_mae | 2.7104 | 2.7052 | -0.0052 | -0.0440 | 0.0358 | -0.1934 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | high_mae | 3.1445 | 3.1424 | -0.0021 | -0.0423 | 0.0323 | -0.0678 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | low_mae | 2.2764 | 2.2680 | -0.0084 | -0.0518 | 0.0463 | -0.3669 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | center_bias | -0.6427 | -0.6159 | 0.0269 | -0.0376 | 0.0802 | — | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | half_width_bias | -0.5872 | -0.6014 | -0.0142 | -0.0333 | 0.0046 | — | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | midpoint_mae | 2.5207 | 2.5204 | -0.0003 | -0.0415 | 0.0429 | -0.0118 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | midpoint_crps | 1.9078 | 1.9044 | -0.0033 | -0.0336 | 0.0283 | -0.1746 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | range_mae | 2.7599 | 2.7463 | -0.0136 | -0.0339 | 0.0042 | -0.4933 | 76 |
| all27 | 17 | 17 | 2 | ce_vs_current | legal_fraction | 0.9740 | 0.9751 | 0.0011 | 0.0000 | 0.0022 | 0.1095 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | endpoint_mae | 2.7052 | 2.7005 | -0.0047 | -0.0498 | 0.0349 | -0.1735 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | high_mae | 3.1424 | 3.1413 | -0.0011 | -0.0421 | 0.0379 | -0.0345 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | low_mae | 2.2680 | 2.2597 | -0.0083 | -0.0625 | 0.0364 | -0.3661 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | center_bias | -0.6159 | -0.5979 | 0.0179 | -0.0486 | 0.0926 | — | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | half_width_bias | -0.6014 | -0.6097 | -0.0083 | -0.0281 | 0.0090 | — | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | midpoint_mae | 2.5204 | 2.5075 | -0.0129 | -0.0572 | 0.0261 | -0.5117 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | midpoint_crps | 1.9044 | 1.8998 | -0.0046 | -0.0358 | 0.0223 | -0.2425 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | range_mae | 2.7463 | 2.7524 | 0.0061 | -0.0186 | 0.0267 | 0.2217 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_ce | legal_fraction | 0.9751 | 0.9738 | -0.0013 | -0.0019 | -0.0006 | -0.1318 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | endpoint_mae | 2.7104 | 2.7005 | -0.0099 | -0.0419 | 0.0218 | -0.3666 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | high_mae | 3.1445 | 3.1413 | -0.0032 | -0.0373 | 0.0324 | -0.1023 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | low_mae | 2.2764 | 2.2597 | -0.0167 | -0.0562 | 0.0195 | -0.7317 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | center_bias | -0.6427 | -0.5979 | 0.0448 | -0.0118 | 0.1179 | — | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | half_width_bias | -0.5872 | -0.6097 | -0.0224 | -0.0462 | -0.0018 | — | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | midpoint_mae | 2.5207 | 2.5075 | -0.0132 | -0.0597 | 0.0299 | -0.5234 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | midpoint_crps | 1.9078 | 1.8998 | -0.0079 | -0.0279 | 0.0123 | -0.4167 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | range_mae | 2.7599 | 2.7524 | -0.0075 | -0.0296 | 0.0165 | -0.2727 | 76 |
| all27 | 17 | 17 | 2 | midpoint_vs_current | legal_fraction | 0.9740 | 0.9738 | -0.0002 | -0.0014 | 0.0011 | -0.0225 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | endpoint_mae | 4.2065 | 4.1855 | -0.0211 | -0.0754 | 0.0343 | -0.5006 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | high_mae | 5.0726 | 5.0587 | -0.0139 | -0.0654 | 0.0367 | -0.2742 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | low_mae | 3.3405 | 3.3123 | -0.0282 | -0.0932 | 0.0411 | -0.8444 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | center_bias | -1.6116 | -1.5448 | 0.0668 | -0.0422 | 0.1698 | — | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | half_width_bias | -0.9512 | -0.9996 | -0.0484 | -0.0806 | -0.0120 | — | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | midpoint_mae | 3.9237 | 3.9098 | -0.0138 | -0.0688 | 0.0454 | -0.3529 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | midpoint_crps | 2.9339 | 2.9270 | -0.0070 | -0.0464 | 0.0333 | -0.2371 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | range_mae | 4.7905 | 4.7239 | -0.0666 | -0.0980 | -0.0321 | -1.3899 | 76 |
| all27 | 17 | 17 | 5 | ce_vs_current | legal_fraction | 0.9360 | 0.9388 | 0.0027 | 0.0001 | 0.0062 | 0.2916 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | endpoint_mae | 4.1855 | 4.1691 | -0.0163 | -0.0926 | 0.0483 | -0.3905 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | high_mae | 5.0587 | 5.0616 | 0.0029 | -0.0569 | 0.0596 | 0.0582 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | low_mae | 3.3123 | 3.2766 | -0.0356 | -0.1331 | 0.0492 | -1.0759 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | center_bias | -1.5448 | -1.5392 | 0.0056 | -0.1402 | 0.1648 | — | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | half_width_bias | -0.9996 | -1.0024 | -0.0028 | -0.0637 | 0.0551 | — | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | midpoint_mae | 3.9098 | 3.9056 | -0.0042 | -0.0777 | 0.0592 | -0.1066 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | midpoint_crps | 2.9270 | 2.9172 | -0.0098 | -0.0667 | 0.0369 | -0.3343 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | range_mae | 4.7239 | 4.7720 | 0.0481 | -0.0248 | 0.1105 | 1.0186 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_ce | legal_fraction | 0.9388 | 0.9344 | -0.0043 | -0.0056 | -0.0031 | -0.4632 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | endpoint_mae | 4.2065 | 4.1691 | -0.0374 | -0.0948 | 0.0150 | -0.8892 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | high_mae | 5.0726 | 5.0616 | -0.0110 | -0.0590 | 0.0370 | -0.2161 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | low_mae | 3.3405 | 3.2766 | -0.0638 | -0.1457 | 0.0038 | -1.9112 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | center_bias | -1.6116 | -1.5392 | 0.0724 | -0.0156 | 0.1828 | — | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | half_width_bias | -0.9512 | -1.0024 | -0.0512 | -0.1040 | -0.0051 | — | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | midpoint_mae | 3.9237 | 3.9056 | -0.0180 | -0.0784 | 0.0338 | -0.4591 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | midpoint_crps | 2.9339 | 2.9172 | -0.0167 | -0.0553 | 0.0152 | -0.5706 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | range_mae | 4.7905 | 4.7720 | -0.0185 | -0.0928 | 0.0449 | -0.3855 | 76 |
| all27 | 17 | 17 | 5 | midpoint_vs_current | legal_fraction | 0.9360 | 0.9344 | -0.0016 | -0.0040 | 0.0012 | -0.1730 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | endpoint_mae | 2.7205 | 2.6989 | -0.0216 | -0.0424 | -0.0024 | -0.7929 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | high_mae | 3.1548 | 3.1200 | -0.0348 | -0.0521 | -0.0172 | -1.1019 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | low_mae | 2.2862 | 2.2778 | -0.0084 | -0.0470 | 0.0244 | -0.3665 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | center_bias | -0.4951 | -0.6415 | -0.1464 | -0.1698 | -0.1213 | — | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | half_width_bias | -0.5539 | -0.6017 | -0.0478 | -0.0836 | -0.0163 | — | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | midpoint_mae | 2.5181 | 2.5145 | -0.0036 | -0.0248 | 0.0150 | -0.1439 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | midpoint_crps | 1.8965 | 1.8990 | 0.0025 | -0.0138 | 0.0167 | 0.1329 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | range_mae | 2.7915 | 2.7528 | -0.0387 | -0.0936 | 0.0079 | -1.3858 | 76 |
| all27 | 17 | 29 | 2 | ce_vs_current | legal_fraction | 0.9734 | 0.9772 | 0.0038 | 0.0025 | 0.0052 | 0.3858 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | endpoint_mae | 2.6989 | 2.6698 | -0.0291 | -0.0627 | -0.0035 | -1.0779 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | high_mae | 3.1200 | 3.1135 | -0.0065 | -0.0337 | 0.0188 | -0.2095 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | low_mae | 2.2778 | 2.2261 | -0.0516 | -0.0957 | -0.0213 | -2.2675 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | center_bias | -0.6415 | -0.5832 | 0.0583 | 0.0098 | 0.1168 | — | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | half_width_bias | -0.6017 | -0.6395 | -0.0379 | -0.0539 | -0.0226 | — | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | midpoint_mae | 2.5145 | 2.4926 | -0.0219 | -0.0577 | 0.0086 | -0.8691 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | midpoint_crps | 1.8990 | 1.8781 | -0.0209 | -0.0436 | -0.0028 | -1.1010 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | range_mae | 2.7528 | 2.7230 | -0.0298 | -0.0498 | -0.0078 | -1.0827 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_ce | legal_fraction | 0.9772 | 0.9777 | 0.0006 | -0.0003 | 0.0013 | 0.0597 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | endpoint_mae | 2.7205 | 2.6698 | -0.0507 | -0.0935 | -0.0152 | -1.8623 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | high_mae | 3.1548 | 3.1135 | -0.0413 | -0.0643 | -0.0168 | -1.3091 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | low_mae | 2.2862 | 2.2261 | -0.0600 | -0.1312 | -0.0051 | -2.6257 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | center_bias | -0.4951 | -0.5832 | -0.0881 | -0.1437 | -0.0238 | — | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | half_width_bias | -0.5539 | -0.6395 | -0.0856 | -0.1322 | -0.0467 | — | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | midpoint_mae | 2.5181 | 2.4926 | -0.0255 | -0.0601 | 0.0046 | -1.0117 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | midpoint_crps | 1.8965 | 1.8781 | -0.0184 | -0.0445 | 0.0041 | -0.9696 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | range_mae | 2.7915 | 2.7230 | -0.0685 | -0.1251 | -0.0218 | -2.4534 | 76 |
| all27 | 17 | 29 | 2 | midpoint_vs_current | legal_fraction | 0.9734 | 0.9777 | 0.0043 | 0.0036 | 0.0051 | 0.4457 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | endpoint_mae | 4.2277 | 4.1695 | -0.0582 | -0.1393 | 0.0172 | -1.3777 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | high_mae | 5.0455 | 4.9940 | -0.0515 | -0.1240 | 0.0155 | -1.0209 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | low_mae | 3.4099 | 3.3450 | -0.0650 | -0.2112 | 0.0616 | -1.9055 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | center_bias | -1.3494 | -1.6798 | -0.3305 | -0.3909 | -0.2610 | — | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | half_width_bias | -0.8192 | -1.0072 | -0.1880 | -0.2790 | -0.1082 | — | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | midpoint_mae | 3.9182 | 3.8775 | -0.0407 | -0.1203 | 0.0365 | -1.0380 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | midpoint_crps | 2.9267 | 2.9090 | -0.0177 | -0.0639 | 0.0238 | -0.6041 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | range_mae | 4.8805 | 4.7237 | -0.1568 | -0.3099 | -0.0216 | -3.2124 | 76 |
| all27 | 17 | 29 | 5 | ce_vs_current | legal_fraction | 0.9345 | 0.9446 | 0.0101 | 0.0086 | 0.0119 | 1.0827 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | endpoint_mae | 4.1695 | 4.0867 | -0.0828 | -0.1682 | -0.0166 | -1.9862 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | high_mae | 4.9940 | 4.9708 | -0.0232 | -0.0716 | 0.0179 | -0.4639 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | low_mae | 3.3450 | 3.2025 | -0.1425 | -0.2784 | -0.0360 | -4.2590 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | center_bias | -1.6798 | -1.5154 | 0.1645 | 0.0739 | 0.2823 | — | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | half_width_bias | -1.0072 | -1.1025 | -0.0953 | -0.1424 | -0.0518 | — | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | midpoint_mae | 3.8775 | 3.8049 | -0.0726 | -0.1638 | -0.0073 | -1.8734 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | midpoint_crps | 2.9090 | 2.8609 | -0.0481 | -0.1011 | -0.0075 | -1.6530 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | range_mae | 4.7237 | 4.6731 | -0.0506 | -0.1042 | -0.0054 | -1.0710 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_ce | legal_fraction | 0.9446 | 0.9469 | 0.0023 | 0.0005 | 0.0038 | 0.2395 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | endpoint_mae | 4.2277 | 4.0867 | -0.1411 | -0.2795 | -0.0329 | -3.3365 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | high_mae | 5.0455 | 4.9708 | -0.0747 | -0.1378 | -0.0211 | -1.4801 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | low_mae | 3.4099 | 3.2025 | -0.2074 | -0.4591 | -0.0133 | -6.0833 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | center_bias | -1.3494 | -1.5154 | -0.1660 | -0.2946 | -0.0208 | — | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | half_width_bias | -0.8192 | -1.1025 | -0.2833 | -0.4123 | -0.1706 | — | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | midpoint_mae | 3.9182 | 3.8049 | -0.1133 | -0.2504 | 0.0048 | -2.8919 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | midpoint_crps | 2.9267 | 2.8609 | -0.0658 | -0.1557 | 0.0037 | -2.2471 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | range_mae | 4.8805 | 4.6731 | -0.2074 | -0.3990 | -0.0466 | -4.2490 | 76 |
| all27 | 17 | 29 | 5 | midpoint_vs_current | legal_fraction | 0.9345 | 0.9469 | 0.0124 | 0.0109 | 0.0137 | 1.3248 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | endpoint_mae | 2.7792 | 2.8324 | 0.0532 | 0.0144 | 0.0999 | 1.9148 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | high_mae | 3.1329 | 3.1409 | 0.0080 | -0.0163 | 0.0334 | 0.2547 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | low_mae | 2.4254 | 2.5238 | 0.0985 | 0.0242 | 0.1858 | 4.0592 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | center_bias | -0.8094 | -0.8804 | -0.0710 | -0.1905 | 0.0364 | — | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | half_width_bias | -0.5625 | -0.4402 | 0.1223 | 0.0837 | 0.1700 | — | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | midpoint_mae | 2.5929 | 2.6452 | 0.0523 | 0.0033 | 0.1107 | 2.0168 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | midpoint_crps | 1.9590 | 1.9933 | 0.0342 | 0.0010 | 0.0743 | 1.7473 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | range_mae | 2.8015 | 2.8911 | 0.0896 | 0.0149 | 0.1705 | 3.1988 | 76 |
| all27 | 17 | 43 | 2 | ce_vs_current | legal_fraction | 0.9715 | 0.9741 | 0.0026 | 0.0016 | 0.0036 | 0.2649 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | endpoint_mae | 2.8324 | 2.8899 | 0.0576 | 0.0167 | 0.1106 | 2.0324 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | high_mae | 3.1409 | 3.1700 | 0.0291 | 0.0031 | 0.0525 | 0.9250 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | low_mae | 2.5238 | 2.6099 | 0.0861 | 0.0117 | 0.1811 | 3.4105 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | center_bias | -0.8804 | -0.9951 | -0.1146 | -0.1708 | -0.0596 | — | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | half_width_bias | -0.4402 | -0.4043 | 0.0359 | 0.0046 | 0.0714 | — | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | midpoint_mae | 2.6452 | 2.7024 | 0.0572 | 0.0148 | 0.1071 | 2.1617 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | midpoint_crps | 1.9933 | 2.0339 | 0.0406 | 0.0098 | 0.0778 | 2.0371 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | range_mae | 2.8911 | 2.9511 | 0.0600 | 0.0102 | 0.1242 | 2.0766 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_ce | legal_fraction | 0.9741 | 0.9743 | 0.0002 | -0.0006 | 0.0010 | 0.0245 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | endpoint_mae | 2.7792 | 2.8899 | 0.1108 | 0.0373 | 0.2045 | 3.9861 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | high_mae | 3.1329 | 3.1700 | 0.0370 | 0.0079 | 0.0666 | 1.1821 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | low_mae | 2.4254 | 2.6099 | 0.1845 | 0.0535 | 0.3565 | 7.6082 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | center_bias | -0.8094 | -0.9951 | -0.1857 | -0.3452 | -0.0515 | — | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | half_width_bias | -0.5625 | -0.4043 | 0.1582 | 0.1004 | 0.2297 | — | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | midpoint_mae | 2.5929 | 2.7024 | 0.1095 | 0.0306 | 0.2087 | 4.2220 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | midpoint_crps | 1.9590 | 2.0339 | 0.0748 | 0.0200 | 0.1453 | 3.8200 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | range_mae | 2.8015 | 2.9511 | 0.1496 | 0.0541 | 0.2788 | 5.3418 | 76 |
| all27 | 17 | 43 | 2 | midpoint_vs_current | legal_fraction | 0.9715 | 0.9743 | 0.0028 | 0.0017 | 0.0039 | 0.2894 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | endpoint_mae | 4.4189 | 4.5948 | 0.1759 | 0.0196 | 0.3681 | 3.9796 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | high_mae | 5.0822 | 5.1048 | 0.0226 | -0.0256 | 0.0673 | 0.4455 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | low_mae | 3.7557 | 4.0848 | 0.3291 | 0.0302 | 0.7038 | 8.7618 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | center_bias | -2.0692 | -2.2555 | -0.1863 | -0.4826 | 0.0817 | — | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | half_width_bias | -0.8509 | -0.5040 | 0.3469 | 0.2415 | 0.4783 | — | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | midpoint_mae | 4.1181 | 4.2773 | 0.1592 | 0.0000 | 0.3687 | 3.8658 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | midpoint_crps | 3.0785 | 3.1887 | 0.1101 | 0.0006 | 0.2534 | 3.5770 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | range_mae | 4.9462 | 5.1877 | 0.2415 | 0.0266 | 0.4884 | 4.8835 | 76 |
| all27 | 17 | 43 | 5 | ce_vs_current | legal_fraction | 0.9320 | 0.9343 | 0.0023 | -0.0005 | 0.0053 | 0.2447 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | endpoint_mae | 4.5948 | 4.7353 | 0.1405 | 0.0442 | 0.2521 | 3.0588 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | high_mae | 5.1048 | 5.1478 | 0.0430 | 0.0033 | 0.0828 | 0.8424 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | low_mae | 4.0848 | 4.3228 | 0.2381 | 0.0641 | 0.4362 | 5.8287 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | center_bias | -2.2555 | -2.5410 | -0.2855 | -0.4207 | -0.1454 | — | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | half_width_bias | -0.5040 | -0.4139 | 0.0901 | 0.0191 | 0.1778 | — | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | midpoint_mae | 4.2773 | 4.4149 | 0.1376 | 0.0337 | 0.2567 | 3.2168 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | midpoint_crps | 3.1887 | 3.2903 | 0.1017 | 0.0345 | 0.1810 | 3.1880 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | range_mae | 5.1877 | 5.3505 | 0.1628 | 0.0543 | 0.3055 | 3.1383 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_ce | legal_fraction | 0.9343 | 0.9332 | -0.0011 | -0.0028 | 0.0006 | -0.1224 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | endpoint_mae | 4.4189 | 4.7353 | 0.3164 | 0.1069 | 0.5987 | 7.1601 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | high_mae | 5.0822 | 5.1478 | 0.0656 | 0.0204 | 0.1157 | 1.2917 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | low_mae | 3.7557 | 4.3228 | 0.5672 | 0.1697 | 1.0953 | 15.1012 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | center_bias | -2.0692 | -2.5410 | -0.4719 | -0.8426 | -0.1614 | — | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | half_width_bias | -0.8509 | -0.4139 | 0.4370 | 0.2791 | 0.6458 | — | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | midpoint_mae | 4.1181 | 4.4149 | 0.2968 | 0.0668 | 0.6143 | 7.2069 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | midpoint_crps | 3.0785 | 3.2903 | 0.2118 | 0.0580 | 0.4227 | 6.8791 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | range_mae | 4.9462 | 5.3505 | 0.4044 | 0.1340 | 0.7806 | 8.1751 | 76 |
| all27 | 17 | 43 | 5 | midpoint_vs_current | legal_fraction | 0.9320 | 0.9332 | 0.0011 | -0.0016 | 0.0036 | 0.1220 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | endpoint_mae | 2.7258 | 2.7375 | 0.0117 | -0.0060 | 0.0322 | 0.4274 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | high_mae | 3.1359 | 3.1332 | -0.0027 | -0.0203 | 0.0134 | -0.0862 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | low_mae | 2.3158 | 2.3418 | 0.0260 | -0.0029 | 0.0627 | 1.1230 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | center_bias | -0.6466 | -0.7001 | -0.0535 | -0.1166 | 0.0022 | — | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | half_width_bias | -0.5626 | -0.5406 | 0.0221 | 0.0084 | 0.0379 | — | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | midpoint_mae | 2.5415 | 2.5508 | 0.0092 | -0.0096 | 0.0314 | 0.3638 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | midpoint_crps | 1.9135 | 1.9201 | 0.0065 | -0.0080 | 0.0224 | 0.3406 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | range_mae | 2.7875 | 2.7971 | 0.0095 | -0.0076 | 0.0283 | 0.3423 | 76 |
| all27 | 100017 | 0 | 2 | ce_vs_current | legal_fraction | 0.9729 | 0.9751 | 0.0022 | 0.0014 | 0.0031 | 0.2271 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | endpoint_mae | 2.7375 | 2.7444 | 0.0069 | -0.0068 | 0.0205 | 0.2537 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | high_mae | 3.1332 | 3.1330 | -0.0001 | -0.0139 | 0.0143 | -0.0046 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | low_mae | 2.3418 | 2.3559 | 0.0140 | -0.0024 | 0.0312 | 0.5992 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | center_bias | -0.7001 | -0.7239 | -0.0238 | -0.0505 | 0.0089 | — | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | half_width_bias | -0.5406 | -0.5451 | -0.0045 | -0.0157 | 0.0055 | — | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | midpoint_mae | 2.5508 | 2.5577 | 0.0069 | -0.0079 | 0.0240 | 0.2698 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | midpoint_crps | 1.9201 | 1.9297 | 0.0097 | -0.0006 | 0.0215 | 0.5046 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | range_mae | 2.7971 | 2.8057 | 0.0086 | -0.0094 | 0.0260 | 0.3070 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_ce | legal_fraction | 0.9751 | 0.9749 | -0.0002 | -0.0008 | 0.0004 | -0.0173 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | endpoint_mae | 2.7258 | 2.7444 | 0.0186 | -0.0036 | 0.0461 | 0.6822 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | high_mae | 3.1359 | 3.1330 | -0.0028 | -0.0195 | 0.0163 | -0.0909 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | low_mae | 2.3158 | 2.3559 | 0.0400 | 0.0044 | 0.0859 | 1.7290 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | center_bias | -0.6466 | -0.7239 | -0.0773 | -0.1246 | -0.0418 | — | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | half_width_bias | -0.5626 | -0.5451 | 0.0175 | 0.0033 | 0.0345 | — | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | midpoint_mae | 2.5415 | 2.5577 | 0.0161 | -0.0099 | 0.0499 | 0.6346 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | midpoint_crps | 1.9135 | 1.9297 | 0.0162 | -0.0002 | 0.0385 | 0.8469 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | range_mae | 2.7875 | 2.8057 | 0.0181 | -0.0009 | 0.0425 | 0.6503 | 76 |
| all27 | 100017 | 0 | 2 | midpoint_vs_current | legal_fraction | 0.9729 | 0.9749 | 0.0020 | 0.0013 | 0.0027 | 0.2097 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | endpoint_mae | 4.2727 | 4.3066 | 0.0339 | -0.0151 | 0.0854 | 0.7935 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | high_mae | 5.0428 | 5.0462 | 0.0034 | -0.0247 | 0.0290 | 0.0671 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | low_mae | 3.5025 | 3.5670 | 0.0644 | -0.0139 | 0.1539 | 1.8394 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | center_bias | -1.6805 | -1.8024 | -0.1218 | -0.2459 | -0.0155 | — | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | half_width_bias | -0.8634 | -0.8359 | 0.0275 | -0.0013 | 0.0597 | — | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | midpoint_mae | 3.9724 | 4.0043 | 0.0320 | -0.0156 | 0.0822 | 0.8048 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | midpoint_crps | 2.9698 | 2.9920 | 0.0222 | -0.0146 | 0.0627 | 0.7475 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | range_mae | 4.8566 | 4.8920 | 0.0355 | 0.0053 | 0.0631 | 0.7301 | 76 |
| all27 | 100017 | 0 | 5 | ce_vs_current | legal_fraction | 0.9340 | 0.9395 | 0.0055 | 0.0036 | 0.0075 | 0.5866 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | endpoint_mae | 4.3066 | 4.3320 | 0.0255 | -0.0095 | 0.0589 | 0.5912 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | high_mae | 5.0462 | 5.0513 | 0.0051 | -0.0199 | 0.0287 | 0.1013 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | low_mae | 3.5670 | 3.6128 | 0.0458 | -0.0083 | 0.1000 | 1.2842 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | center_bias | -1.8024 | -1.8736 | -0.0712 | -0.1273 | -0.0098 | — | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | half_width_bias | -0.8359 | -0.8357 | 0.0002 | -0.0206 | 0.0222 | — | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | midpoint_mae | 4.0043 | 4.0327 | 0.0284 | -0.0118 | 0.0678 | 0.7084 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | midpoint_crps | 2.9920 | 3.0149 | 0.0230 | -0.0000 | 0.0467 | 0.7675 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | range_mae | 4.8920 | 4.9061 | 0.0140 | -0.0248 | 0.0524 | 0.2865 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_ce | legal_fraction | 0.9395 | 0.9387 | -0.0008 | -0.0017 | 0.0001 | -0.0839 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | endpoint_mae | 4.2727 | 4.3320 | 0.0594 | 0.0052 | 0.1285 | 1.3894 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | high_mae | 5.0428 | 5.0513 | 0.0085 | -0.0178 | 0.0373 | 0.1684 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | low_mae | 3.5025 | 3.6128 | 0.1102 | 0.0212 | 0.2330 | 3.1472 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | center_bias | -1.6805 | -1.8736 | -0.1930 | -0.2781 | -0.1204 | — | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | half_width_bias | -0.8634 | -0.8357 | 0.0276 | -0.0050 | 0.0585 | — | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | midpoint_mae | 3.9724 | 4.0327 | 0.0603 | -0.0021 | 0.1356 | 1.5189 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | midpoint_crps | 2.9698 | 3.0149 | 0.0452 | 0.0065 | 0.0943 | 1.5208 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | range_mae | 4.8566 | 4.9061 | 0.0495 | 0.0204 | 0.0859 | 1.0187 | 76 |
| all27 | 100017 | 0 | 5 | midpoint_vs_current | legal_fraction | 0.9340 | 0.9387 | 0.0047 | 0.0033 | 0.0062 | 0.5022 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | endpoint_mae | 2.6988 | 2.6884 | -0.0104 | -0.0393 | 0.0159 | -0.3843 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | high_mae | 3.1214 | 3.1238 | 0.0024 | -0.0220 | 0.0236 | 0.0759 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | low_mae | 2.2762 | 2.2531 | -0.0231 | -0.0626 | 0.0135 | -1.0154 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | center_bias | -0.6305 | -0.5795 | 0.0510 | 0.0041 | 0.0965 | — | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | half_width_bias | -0.5839 | -0.5984 | -0.0145 | -0.0309 | 0.0013 | — | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | midpoint_mae | 2.5179 | 2.5016 | -0.0163 | -0.0446 | 0.0124 | -0.6481 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | midpoint_crps | 1.8975 | 1.8817 | -0.0158 | -0.0375 | 0.0062 | -0.8326 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | range_mae | 2.7514 | 2.7331 | -0.0184 | -0.0407 | 0.0021 | -0.6672 | 76 |
| all27 | 100017 | 17 | 2 | ce_vs_current | legal_fraction | 0.9737 | 0.9750 | 0.0014 | 0.0000 | 0.0027 | 0.1399 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | endpoint_mae | 2.6884 | 2.6819 | -0.0066 | -0.0243 | 0.0106 | -0.2446 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | high_mae | 3.1238 | 3.1211 | -0.0027 | -0.0245 | 0.0171 | -0.0851 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | low_mae | 2.2531 | 2.2426 | -0.0105 | -0.0440 | 0.0152 | -0.4658 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | center_bias | -0.5795 | -0.5786 | 0.0009 | -0.0661 | 0.0717 | — | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | half_width_bias | -0.5984 | -0.6110 | -0.0126 | -0.0399 | 0.0138 | — | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | midpoint_mae | 2.5016 | 2.4926 | -0.0090 | -0.0330 | 0.0147 | -0.3589 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | midpoint_crps | 1.8817 | 1.8874 | 0.0057 | -0.0077 | 0.0184 | 0.3030 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | range_mae | 2.7331 | 2.7555 | 0.0225 | 0.0038 | 0.0440 | 0.8222 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_ce | legal_fraction | 0.9750 | 0.9739 | -0.0011 | -0.0018 | -0.0005 | -0.1167 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | endpoint_mae | 2.6988 | 2.6819 | -0.0169 | -0.0434 | 0.0079 | -0.6280 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | high_mae | 3.1214 | 3.1211 | -0.0003 | -0.0283 | 0.0279 | -0.0093 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | low_mae | 2.2762 | 2.2426 | -0.0336 | -0.0700 | 0.0006 | -1.4765 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | center_bias | -0.6305 | -0.5786 | 0.0519 | 0.0126 | 0.0990 | — | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | half_width_bias | -0.5839 | -0.6110 | -0.0271 | -0.0528 | -0.0040 | — | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | midpoint_mae | 2.5179 | 2.4926 | -0.0253 | -0.0595 | 0.0075 | -1.0047 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | midpoint_crps | 1.8975 | 1.8874 | -0.0101 | -0.0308 | 0.0089 | -0.5321 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | range_mae | 2.7514 | 2.7555 | 0.0041 | -0.0233 | 0.0316 | 0.1496 | 76 |
| all27 | 100017 | 17 | 2 | midpoint_vs_current | legal_fraction | 0.9737 | 0.9739 | 0.0002 | -0.0008 | 0.0013 | 0.0230 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | endpoint_mae | 4.1985 | 4.1674 | -0.0311 | -0.0838 | 0.0229 | -0.7407 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | high_mae | 5.0473 | 5.0348 | -0.0124 | -0.0592 | 0.0339 | -0.2459 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | low_mae | 3.3498 | 3.3000 | -0.0498 | -0.1261 | 0.0233 | -1.4863 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | center_bias | -1.6126 | -1.4988 | 0.1138 | 0.0050 | 0.2243 | — | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | half_width_bias | -0.9386 | -0.9829 | -0.0443 | -0.0663 | -0.0194 | — | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | midpoint_mae | 3.9073 | 3.8725 | -0.0348 | -0.0919 | 0.0230 | -0.8917 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | midpoint_crps | 2.9188 | 2.8922 | -0.0266 | -0.0608 | 0.0101 | -0.9113 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | range_mae | 4.7925 | 4.7313 | -0.0612 | -0.1112 | -0.0108 | -1.2771 | 76 |
| all27 | 100017 | 17 | 5 | ce_vs_current | legal_fraction | 0.9357 | 0.9387 | 0.0030 | 0.0014 | 0.0048 | 0.3195 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | endpoint_mae | 4.1674 | 4.1583 | -0.0091 | -0.0777 | 0.0489 | -0.2195 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | high_mae | 5.0348 | 5.0400 | 0.0051 | -0.0380 | 0.0503 | 0.1023 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | low_mae | 3.3000 | 3.2766 | -0.0234 | -0.1281 | 0.0591 | -0.7105 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | center_bias | -1.4988 | -1.5192 | -0.0204 | -0.1743 | 0.1496 | — | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | half_width_bias | -0.9829 | -1.0096 | -0.0268 | -0.0812 | 0.0225 | — | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | midpoint_mae | 3.8725 | 3.8631 | -0.0094 | -0.0724 | 0.0449 | -0.2418 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | midpoint_crps | 2.8922 | 2.8994 | 0.0072 | -0.0289 | 0.0364 | 0.2488 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | range_mae | 4.7313 | 4.7762 | 0.0449 | -0.0140 | 0.1011 | 0.9484 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_ce | legal_fraction | 0.9387 | 0.9364 | -0.0024 | -0.0036 | -0.0010 | -0.2511 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | endpoint_mae | 4.1985 | 4.1583 | -0.0402 | -0.0987 | 0.0092 | -0.9586 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | high_mae | 5.0473 | 5.0400 | -0.0073 | -0.0419 | 0.0305 | -0.1439 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | low_mae | 3.3498 | 3.2766 | -0.0732 | -0.1740 | 0.0054 | -2.1862 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | center_bias | -1.6126 | -1.5192 | 0.0934 | 0.0089 | 0.1894 | — | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | half_width_bias | -0.9386 | -1.0096 | -0.0710 | -0.1234 | -0.0271 | — | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | midpoint_mae | 3.9073 | 3.8631 | -0.0442 | -0.1067 | 0.0148 | -1.1313 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | midpoint_crps | 2.9188 | 2.8994 | -0.0194 | -0.0471 | 0.0045 | -0.6648 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | range_mae | 4.7925 | 4.7762 | -0.0163 | -0.0916 | 0.0663 | -0.3408 | 76 |
| all27 | 100017 | 17 | 5 | midpoint_vs_current | legal_fraction | 0.9357 | 0.9364 | 0.0006 | -0.0015 | 0.0027 | 0.0676 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | endpoint_mae | 2.7201 | 2.6961 | -0.0240 | -0.0453 | -0.0007 | -0.8829 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | high_mae | 3.1586 | 3.1266 | -0.0320 | -0.0596 | -0.0047 | -1.0124 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | low_mae | 2.2817 | 2.2656 | -0.0161 | -0.0440 | 0.0103 | -0.7038 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | center_bias | -0.5087 | -0.6653 | -0.1566 | -0.1972 | -0.1192 | — | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | half_width_bias | -0.5418 | -0.5877 | -0.0460 | -0.0746 | -0.0200 | — | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | midpoint_mae | 2.5325 | 2.5122 | -0.0204 | -0.0437 | 0.0056 | -0.8039 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | midpoint_crps | 1.9002 | 1.8923 | -0.0079 | -0.0208 | 0.0055 | -0.4171 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | range_mae | 2.7952 | 2.7503 | -0.0449 | -0.0874 | -0.0097 | -1.6056 | 76 |
| all27 | 100017 | 29 | 2 | ce_vs_current | legal_fraction | 0.9731 | 0.9766 | 0.0035 | 0.0027 | 0.0043 | 0.3595 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | endpoint_mae | 2.6961 | 2.6693 | -0.0268 | -0.0506 | -0.0026 | -0.9927 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | high_mae | 3.1266 | 3.1117 | -0.0149 | -0.0386 | 0.0087 | -0.4750 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | low_mae | 2.2656 | 2.2270 | -0.0387 | -0.0806 | -0.0010 | -1.7072 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | center_bias | -0.6653 | -0.5961 | 0.0692 | 0.0288 | 0.1239 | — | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | half_width_bias | -0.5877 | -0.6206 | -0.0328 | -0.0489 | -0.0159 | — | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | midpoint_mae | 2.5122 | 2.4873 | -0.0249 | -0.0517 | 0.0014 | -0.9902 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | midpoint_crps | 1.8923 | 1.8759 | -0.0163 | -0.0306 | -0.0027 | -0.8635 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | range_mae | 2.7503 | 2.7235 | -0.0269 | -0.0547 | -0.0015 | -0.9767 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_ce | legal_fraction | 0.9766 | 0.9770 | 0.0003 | -0.0003 | 0.0009 | 0.0331 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | endpoint_mae | 2.7201 | 2.6693 | -0.0508 | -0.0810 | -0.0223 | -1.8669 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | high_mae | 3.1586 | 3.1117 | -0.0468 | -0.0689 | -0.0239 | -1.4825 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | low_mae | 2.2817 | 2.2270 | -0.0547 | -0.1127 | -0.0055 | -2.3989 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | center_bias | -0.5087 | -0.5961 | -0.0874 | -0.1228 | -0.0517 | — | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | half_width_bias | -0.5418 | -0.6206 | -0.0788 | -0.1180 | -0.0421 | — | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | midpoint_mae | 2.5325 | 2.4873 | -0.0452 | -0.0703 | -0.0197 | -1.7862 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | midpoint_crps | 1.9002 | 1.8759 | -0.0243 | -0.0442 | -0.0029 | -1.2770 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | range_mae | 2.7952 | 2.7235 | -0.0717 | -0.1173 | -0.0286 | -2.5665 | 76 |
| all27 | 100017 | 29 | 2 | midpoint_vs_current | legal_fraction | 0.9731 | 0.9770 | 0.0038 | 0.0029 | 0.0047 | 0.3928 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | endpoint_mae | 4.2363 | 4.1847 | -0.0515 | -0.1424 | 0.0275 | -1.2166 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | high_mae | 5.0605 | 5.0196 | -0.0409 | -0.0962 | 0.0143 | -0.8085 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | low_mae | 3.4120 | 3.3498 | -0.0622 | -0.2115 | 0.0631 | -1.8219 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | center_bias | -1.3644 | -1.7049 | -0.3405 | -0.4058 | -0.2628 | — | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | half_width_bias | -0.8122 | -1.0188 | -0.2065 | -0.3140 | -0.1169 | — | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | midpoint_mae | 3.9104 | 3.8848 | -0.0256 | -0.1085 | 0.0497 | -0.6542 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | midpoint_crps | 2.9251 | 2.9177 | -0.0075 | -0.0495 | 0.0298 | -0.2554 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | range_mae | 4.8456 | 4.7300 | -0.1155 | -0.2553 | 0.0222 | -2.3846 | 76 |
| all27 | 100017 | 29 | 5 | ce_vs_current | legal_fraction | 0.9351 | 0.9448 | 0.0097 | 0.0078 | 0.0120 | 1.0395 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | endpoint_mae | 4.1847 | 4.0974 | -0.0874 | -0.1484 | -0.0375 | -2.0874 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | high_mae | 5.0196 | 4.9920 | -0.0277 | -0.0653 | 0.0117 | -0.5510 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | low_mae | 3.3498 | 3.2028 | -0.1470 | -0.2712 | -0.0457 | -4.3896 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | center_bias | -1.7049 | -1.5317 | 0.1733 | 0.0736 | 0.3022 | — | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | half_width_bias | -1.0188 | -1.1043 | -0.0855 | -0.1216 | -0.0482 | — | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | midpoint_mae | 3.8848 | 3.8252 | -0.0596 | -0.1228 | -0.0086 | -1.5332 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | midpoint_crps | 2.9177 | 2.8655 | -0.0521 | -0.1050 | -0.0124 | -1.7869 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | range_mae | 4.7300 | 4.6541 | -0.0759 | -0.1399 | -0.0239 | -1.6053 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_ce | legal_fraction | 0.9448 | 0.9465 | 0.0016 | 0.0004 | 0.0029 | 0.1722 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | endpoint_mae | 4.2363 | 4.0974 | -0.1389 | -0.2734 | -0.0355 | -3.2786 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | high_mae | 5.0605 | 4.9920 | -0.0686 | -0.1320 | 0.0005 | -1.3550 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | low_mae | 3.4120 | 3.2028 | -0.2092 | -0.4455 | -0.0231 | -6.1315 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | center_bias | -1.3644 | -1.5317 | -0.1672 | -0.2799 | -0.0375 | — | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | half_width_bias | -0.8122 | -1.1043 | -0.2921 | -0.4286 | -0.1803 | — | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | midpoint_mae | 3.9104 | 3.8252 | -0.0851 | -0.2090 | 0.0109 | -2.1774 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | midpoint_crps | 2.9251 | 2.8655 | -0.0596 | -0.1421 | 0.0072 | -2.0378 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | range_mae | 4.8456 | 4.6541 | -0.1915 | -0.3707 | -0.0400 | -3.9517 | 76 |
| all27 | 100017 | 29 | 5 | midpoint_vs_current | legal_fraction | 0.9351 | 0.9465 | 0.0113 | 0.0097 | 0.0131 | 1.2135 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | endpoint_mae | 2.7586 | 2.8280 | 0.0693 | 0.0232 | 0.1279 | 2.5137 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | high_mae | 3.1277 | 3.1492 | 0.0215 | -0.0240 | 0.0597 | 0.6872 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | low_mae | 2.3896 | 2.5068 | 0.1172 | 0.0348 | 0.2301 | 4.9045 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | center_bias | -0.8007 | -0.8555 | -0.0548 | -0.1891 | 0.0617 | — | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | half_width_bias | -0.5623 | -0.4356 | 0.1267 | 0.0817 | 0.1807 | — | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | midpoint_mae | 2.5742 | 2.6386 | 0.0644 | 0.0144 | 0.1252 | 2.5024 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | midpoint_crps | 1.9429 | 1.9861 | 0.0433 | 0.0062 | 0.0855 | 2.2273 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | range_mae | 2.8160 | 2.9078 | 0.0919 | 0.0190 | 0.1724 | 3.2621 | 76 |
| all27 | 100017 | 43 | 2 | ce_vs_current | legal_fraction | 0.9718 | 0.9735 | 0.0018 | 0.0007 | 0.0029 | 0.1817 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | endpoint_mae | 2.8280 | 2.8821 | 0.0542 | 0.0117 | 0.1050 | 1.9157 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | high_mae | 3.1492 | 3.1662 | 0.0171 | -0.0070 | 0.0428 | 0.5422 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | low_mae | 2.5068 | 2.5980 | 0.0913 | 0.0221 | 0.1754 | 3.6411 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | center_bias | -0.8555 | -0.9970 | -0.1414 | -0.1947 | -0.0928 | — | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | half_width_bias | -0.4356 | -0.4037 | 0.0319 | 0.0113 | 0.0542 | — | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | midpoint_mae | 2.6386 | 2.6931 | 0.0545 | 0.0104 | 0.1136 | 2.0654 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | midpoint_crps | 1.9861 | 2.0258 | 0.0397 | 0.0105 | 0.0774 | 1.9991 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | range_mae | 2.9078 | 2.9380 | 0.0302 | -0.0008 | 0.0632 | 1.0369 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_ce | legal_fraction | 0.9735 | 0.9739 | 0.0003 | -0.0010 | 0.0016 | 0.0317 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | endpoint_mae | 2.7586 | 2.8821 | 0.1235 | 0.0484 | 0.2289 | 4.4775 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | high_mae | 3.1277 | 3.1662 | 0.0386 | 0.0044 | 0.0739 | 1.2331 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | low_mae | 2.3896 | 2.5980 | 0.2085 | 0.0669 | 0.3992 | 8.7241 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | center_bias | -0.8007 | -0.9970 | -0.1962 | -0.3662 | -0.0549 | — | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | half_width_bias | -0.5623 | -0.4037 | 0.1586 | 0.0976 | 0.2320 | — | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | midpoint_mae | 2.5742 | 2.6931 | 0.1189 | 0.0341 | 0.2332 | 4.6195 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | midpoint_crps | 1.9429 | 2.0258 | 0.0830 | 0.0286 | 0.1573 | 4.2710 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | range_mae | 2.8160 | 2.9380 | 0.1220 | 0.0358 | 0.2246 | 4.3328 | 76 |
| all27 | 100017 | 43 | 2 | midpoint_vs_current | legal_fraction | 0.9718 | 0.9739 | 0.0021 | 0.0010 | 0.0031 | 0.2135 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | endpoint_mae | 4.3832 | 4.5675 | 0.1843 | 0.0572 | 0.3469 | 4.2058 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | high_mae | 5.0205 | 5.0840 | 0.0635 | 0.0035 | 0.1180 | 1.2642 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | low_mae | 3.7458 | 4.0511 | 0.3052 | 0.0646 | 0.6094 | 8.1484 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | center_bias | -2.0646 | -2.2034 | -0.1388 | -0.4230 | 0.1174 | — | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | half_width_bias | -0.8392 | -0.5060 | 0.3332 | 0.2454 | 0.4452 | — | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | midpoint_mae | 4.0994 | 4.2557 | 0.1563 | 0.0349 | 0.3126 | 3.8135 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | midpoint_crps | 3.0654 | 3.1661 | 0.1007 | 0.0050 | 0.2169 | 3.2840 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | range_mae | 4.9317 | 5.2148 | 0.2831 | 0.0959 | 0.4760 | 5.7410 | 76 |
| all27 | 100017 | 43 | 5 | ce_vs_current | legal_fraction | 0.9311 | 0.9348 | 0.0037 | 0.0010 | 0.0064 | 0.4003 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | endpoint_mae | 4.5675 | 4.7404 | 0.1729 | 0.0509 | 0.3352 | 3.7849 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | high_mae | 5.0840 | 5.1218 | 0.0378 | -0.0138 | 0.0865 | 0.7444 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | low_mae | 4.0511 | 4.3590 | 0.3079 | 0.0968 | 0.5985 | 7.6008 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | center_bias | -2.2034 | -2.5698 | -0.3664 | -0.5311 | -0.2084 | — | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | half_width_bias | -0.5060 | -0.3932 | 0.1128 | 0.0215 | 0.2229 | — | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | midpoint_mae | 4.2557 | 4.4097 | 0.1540 | 0.0216 | 0.3307 | 3.6193 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | midpoint_crps | 3.1661 | 3.2799 | 0.1138 | 0.0244 | 0.2283 | 3.5954 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | range_mae | 5.2148 | 5.2879 | 0.0731 | -0.0168 | 0.1991 | 1.4020 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_ce | legal_fraction | 0.9348 | 0.9331 | -0.0016 | -0.0038 | 0.0004 | -0.1749 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | endpoint_mae | 4.3832 | 4.7404 | 0.3572 | 0.1335 | 0.6695 | 8.1499 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | high_mae | 5.0205 | 5.1218 | 0.1013 | 0.0239 | 0.1842 | 2.0180 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | low_mae | 3.7458 | 4.3590 | 0.6131 | 0.2065 | 1.1777 | 16.3685 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | center_bias | -2.0646 | -2.5698 | -0.5052 | -0.8972 | -0.1717 | — | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | half_width_bias | -0.8392 | -0.3932 | 0.4460 | 0.2929 | 0.6505 | — | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | midpoint_mae | 4.0994 | 4.4097 | 0.3104 | 0.0820 | 0.6259 | 7.5708 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | midpoint_crps | 3.0654 | 3.2799 | 0.2145 | 0.0579 | 0.4319 | 6.9975 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | range_mae | 4.9317 | 5.2879 | 0.3562 | 0.1185 | 0.6570 | 7.2235 | 76 |
| all27 | 100017 | 43 | 5 | midpoint_vs_current | legal_fraction | 0.9311 | 0.9331 | 0.0021 | -0.0000 | 0.0038 | 0.2246 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | endpoint_mae | 2.7317 | 2.7418 | 0.0101 | -0.0140 | 0.0369 | 0.3713 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | high_mae | 3.1435 | 3.1342 | -0.0093 | -0.0291 | 0.0116 | -0.2972 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | low_mae | 2.3198 | 2.3494 | 0.0296 | -0.0060 | 0.0755 | 1.2771 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | center_bias | -0.6515 | -0.7098 | -0.0582 | -0.1311 | 0.0001 | — | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | half_width_bias | -0.5739 | -0.5491 | 0.0249 | 0.0099 | 0.0444 | — | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | midpoint_mae | 2.5422 | 2.5503 | 0.0081 | -0.0166 | 0.0365 | 0.3180 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | midpoint_crps | 1.9178 | 1.9231 | 0.0053 | -0.0116 | 0.0236 | 0.2750 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | range_mae | 2.7925 | 2.7970 | 0.0044 | -0.0149 | 0.0269 | 0.1591 | 76 |
| all27 | 200017 | 0 | 2 | ce_vs_current | legal_fraction | 0.9733 | 0.9753 | 0.0020 | 0.0014 | 0.0026 | 0.2061 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | endpoint_mae | 2.7418 | 2.7512 | 0.0094 | -0.0032 | 0.0221 | 0.3438 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | high_mae | 3.1342 | 3.1399 | 0.0057 | -0.0064 | 0.0162 | 0.1807 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | low_mae | 2.3494 | 2.3626 | 0.0132 | -0.0043 | 0.0333 | 0.5615 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | center_bias | -0.7098 | -0.7260 | -0.0163 | -0.0482 | 0.0213 | — | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | half_width_bias | -0.5491 | -0.5447 | 0.0043 | -0.0041 | 0.0128 | — | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | midpoint_mae | 2.5503 | 2.5587 | 0.0083 | -0.0041 | 0.0203 | 0.3269 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | midpoint_crps | 1.9231 | 1.9283 | 0.0052 | -0.0069 | 0.0161 | 0.2704 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | range_mae | 2.7970 | 2.8138 | 0.0168 | 0.0039 | 0.0295 | 0.6016 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_ce | legal_fraction | 0.9753 | 0.9749 | -0.0004 | -0.0011 | 0.0004 | -0.0404 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | endpoint_mae | 2.7317 | 2.7512 | 0.0196 | -0.0033 | 0.0459 | 0.7164 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | high_mae | 3.1435 | 3.1399 | -0.0037 | -0.0215 | 0.0151 | -0.1170 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | low_mae | 2.3198 | 2.3626 | 0.0428 | 0.0048 | 0.0932 | 1.8458 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | center_bias | -0.6515 | -0.7260 | -0.0745 | -0.1276 | -0.0357 | — | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | half_width_bias | -0.5739 | -0.5447 | 0.0292 | 0.0122 | 0.0491 | — | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | midpoint_mae | 2.5422 | 2.5587 | 0.0164 | -0.0099 | 0.0459 | 0.6460 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | midpoint_crps | 1.9178 | 1.9283 | 0.0105 | -0.0047 | 0.0286 | 0.5462 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | range_mae | 2.7925 | 2.8138 | 0.0213 | -0.0008 | 0.0465 | 0.7617 | 76 |
| all27 | 200017 | 0 | 2 | midpoint_vs_current | legal_fraction | 0.9733 | 0.9749 | 0.0016 | 0.0006 | 0.0027 | 0.1656 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | endpoint_mae | 4.2623 | 4.3174 | 0.0551 | 0.0019 | 0.1159 | 1.2917 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | high_mae | 5.0403 | 5.0484 | 0.0081 | -0.0266 | 0.0432 | 0.1599 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | low_mae | 3.4844 | 3.5864 | 0.1021 | 0.0030 | 0.2251 | 2.9289 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | center_bias | -1.6804 | -1.8182 | -0.1378 | -0.2913 | -0.0072 | — | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | half_width_bias | -0.8789 | -0.8373 | 0.0416 | 0.0160 | 0.0719 | — | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | midpoint_mae | 3.9665 | 4.0294 | 0.0629 | 0.0079 | 0.1284 | 1.5859 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | midpoint_crps | 2.9696 | 3.0091 | 0.0395 | -0.0014 | 0.0882 | 1.3301 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | range_mae | 4.8717 | 4.8917 | 0.0200 | -0.0114 | 0.0537 | 0.4104 | 76 |
| all27 | 200017 | 0 | 5 | ce_vs_current | legal_fraction | 0.9345 | 0.9393 | 0.0048 | 0.0036 | 0.0062 | 0.5151 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | endpoint_mae | 4.3174 | 4.3218 | 0.0044 | -0.0219 | 0.0324 | 0.1026 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | high_mae | 5.0484 | 5.0432 | -0.0052 | -0.0324 | 0.0246 | -0.1030 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | low_mae | 3.5864 | 3.6005 | 0.0141 | -0.0240 | 0.0547 | 0.3920 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | center_bias | -1.8182 | -1.8692 | -0.0510 | -0.1264 | 0.0368 | — | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | half_width_bias | -0.8373 | -0.8410 | -0.0038 | -0.0203 | 0.0131 | — | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | midpoint_mae | 4.0294 | 4.0296 | 0.0002 | -0.0295 | 0.0333 | 0.0041 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | midpoint_crps | 3.0091 | 3.0196 | 0.0104 | -0.0119 | 0.0304 | 0.3463 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | range_mae | 4.8917 | 4.9241 | 0.0323 | 0.0043 | 0.0575 | 0.6608 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_ce | legal_fraction | 0.9393 | 0.9387 | -0.0006 | -0.0014 | 0.0004 | -0.0646 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | endpoint_mae | 4.2623 | 4.3218 | 0.0595 | 0.0035 | 0.1217 | 1.3956 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | high_mae | 5.0403 | 5.0432 | 0.0029 | -0.0345 | 0.0407 | 0.0568 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | low_mae | 3.4844 | 3.6005 | 0.1161 | 0.0298 | 0.2243 | 3.3324 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | center_bias | -1.6804 | -1.8692 | -0.1889 | -0.2820 | -0.1181 | — | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | half_width_bias | -0.8789 | -0.8410 | 0.0378 | 0.0100 | 0.0681 | — | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | midpoint_mae | 3.9665 | 4.0296 | 0.0631 | 0.0042 | 0.1285 | 1.5901 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | midpoint_crps | 2.9696 | 3.0196 | 0.0499 | 0.0103 | 0.0961 | 1.6810 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | range_mae | 4.8717 | 4.9241 | 0.0523 | 0.0152 | 0.0930 | 1.0740 | 76 |
| all27 | 200017 | 0 | 5 | midpoint_vs_current | legal_fraction | 0.9345 | 0.9387 | 0.0042 | 0.0027 | 0.0062 | 0.4502 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | endpoint_mae | 2.7044 | 2.6905 | -0.0139 | -0.0469 | 0.0215 | -0.5156 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | high_mae | 3.1422 | 3.1275 | -0.0148 | -0.0457 | 0.0155 | -0.4695 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | low_mae | 2.2666 | 2.2535 | -0.0131 | -0.0566 | 0.0342 | -0.5795 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | center_bias | -0.6560 | -0.6009 | 0.0551 | -0.0106 | 0.1123 | — | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | half_width_bias | -0.5953 | -0.6075 | -0.0122 | -0.0303 | 0.0064 | — | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | midpoint_mae | 2.5144 | 2.5036 | -0.0108 | -0.0467 | 0.0257 | -0.4298 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | midpoint_crps | 1.9042 | 1.8900 | -0.0142 | -0.0393 | 0.0104 | -0.7468 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | range_mae | 2.7689 | 2.7503 | -0.0186 | -0.0356 | -0.0013 | -0.6731 | 76 |
| all27 | 200017 | 17 | 2 | ce_vs_current | legal_fraction | 0.9737 | 0.9750 | 0.0013 | 0.0006 | 0.0022 | 0.1373 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | endpoint_mae | 2.6905 | 2.6925 | 0.0020 | -0.0264 | 0.0293 | 0.0747 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | high_mae | 3.1275 | 3.1318 | 0.0043 | -0.0255 | 0.0387 | 0.1376 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | low_mae | 2.2535 | 2.2532 | -0.0003 | -0.0347 | 0.0283 | -0.0126 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | center_bias | -0.6009 | -0.5939 | 0.0070 | -0.0668 | 0.0897 | — | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | half_width_bias | -0.6075 | -0.6007 | 0.0069 | -0.0122 | 0.0261 | — | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | midpoint_mae | 2.5036 | 2.4997 | -0.0039 | -0.0426 | 0.0335 | -0.1570 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | midpoint_crps | 1.8900 | 1.8905 | 0.0005 | -0.0240 | 0.0223 | 0.0239 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | range_mae | 2.7503 | 2.7678 | 0.0175 | -0.0033 | 0.0376 | 0.6377 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_ce | legal_fraction | 0.9750 | 0.9733 | -0.0017 | -0.0030 | -0.0004 | -0.1709 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | endpoint_mae | 2.7044 | 2.6925 | -0.0119 | -0.0503 | 0.0257 | -0.4413 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | high_mae | 3.1422 | 3.1318 | -0.0104 | -0.0520 | 0.0283 | -0.3325 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | low_mae | 2.2666 | 2.2532 | -0.0134 | -0.0564 | 0.0273 | -0.5920 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | center_bias | -0.6560 | -0.5939 | 0.0621 | 0.0119 | 0.1282 | — | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | half_width_bias | -0.5953 | -0.6007 | -0.0053 | -0.0248 | 0.0124 | — | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | midpoint_mae | 2.5144 | 2.4997 | -0.0147 | -0.0567 | 0.0300 | -0.5862 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | midpoint_crps | 1.9042 | 1.8905 | -0.0138 | -0.0398 | 0.0112 | -0.7230 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | range_mae | 2.7689 | 2.7678 | -0.0011 | -0.0147 | 0.0152 | -0.0396 | 76 |
| all27 | 200017 | 17 | 2 | midpoint_vs_current | legal_fraction | 0.9737 | 0.9733 | -0.0003 | -0.0015 | 0.0010 | -0.0338 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | endpoint_mae | 4.1995 | 4.1634 | -0.0361 | -0.1007 | 0.0302 | -0.8586 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | high_mae | 5.0577 | 5.0223 | -0.0354 | -0.0971 | 0.0338 | -0.7003 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | low_mae | 3.3412 | 3.3045 | -0.0367 | -0.1198 | 0.0467 | -1.0982 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | center_bias | -1.6249 | -1.5191 | 0.1057 | -0.0268 | 0.2197 | — | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | half_width_bias | -0.9644 | -0.9989 | -0.0345 | -0.0547 | -0.0135 | — | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | midpoint_mae | 3.8914 | 3.8848 | -0.0066 | -0.0763 | 0.0593 | -0.1707 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | midpoint_crps | 2.9249 | 2.9119 | -0.0130 | -0.0529 | 0.0283 | -0.4449 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | range_mae | 4.8044 | 4.7574 | -0.0470 | -0.0982 | 0.0025 | -0.9785 | 76 |
| all27 | 200017 | 17 | 5 | ce_vs_current | legal_fraction | 0.9359 | 0.9391 | 0.0033 | 0.0016 | 0.0048 | 0.3500 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | endpoint_mae | 4.1634 | 4.1430 | -0.0204 | -0.0803 | 0.0323 | -0.4909 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | high_mae | 5.0223 | 5.0244 | 0.0022 | -0.0336 | 0.0437 | 0.0433 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | low_mae | 3.3045 | 3.2615 | -0.0431 | -0.1518 | 0.0478 | -1.3028 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | center_bias | -1.5191 | -1.5292 | -0.0101 | -0.1801 | 0.1855 | — | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | half_width_bias | -0.9989 | -1.0050 | -0.0061 | -0.0582 | 0.0414 | — | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | midpoint_mae | 3.8848 | 3.8698 | -0.0150 | -0.0835 | 0.0445 | -0.3862 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | midpoint_crps | 2.9119 | 2.9068 | -0.0051 | -0.0514 | 0.0341 | -0.1764 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | range_mae | 4.7574 | 4.7633 | 0.0059 | -0.0540 | 0.0530 | 0.1235 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_ce | legal_fraction | 0.9391 | 0.9349 | -0.0042 | -0.0063 | -0.0022 | -0.4478 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | endpoint_mae | 4.1995 | 4.1430 | -0.0565 | -0.1193 | 0.0026 | -1.3452 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | high_mae | 5.0577 | 5.0244 | -0.0332 | -0.0959 | 0.0327 | -0.6572 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | low_mae | 3.3412 | 3.2615 | -0.0797 | -0.1643 | -0.0166 | -2.3867 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | center_bias | -1.6249 | -1.5292 | 0.0957 | 0.0038 | 0.2238 | — | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | half_width_bias | -0.9644 | -1.0050 | -0.0406 | -0.0836 | 0.0011 | — | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | midpoint_mae | 3.8914 | 3.8698 | -0.0216 | -0.0925 | 0.0460 | -0.5562 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | midpoint_crps | 2.9249 | 2.9068 | -0.0182 | -0.0537 | 0.0163 | -0.6205 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | range_mae | 4.8044 | 4.7633 | -0.0411 | -0.1119 | 0.0269 | -0.8562 | 76 |
| all27 | 200017 | 17 | 5 | midpoint_vs_current | legal_fraction | 0.9359 | 0.9349 | -0.0009 | -0.0030 | 0.0014 | -0.0994 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | endpoint_mae | 2.7219 | 2.6936 | -0.0283 | -0.0527 | -0.0013 | -1.0382 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | high_mae | 3.1541 | 3.1251 | -0.0290 | -0.0613 | 0.0046 | -0.9188 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | low_mae | 2.2897 | 2.2621 | -0.0275 | -0.0571 | 0.0011 | -1.2027 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | center_bias | -0.4994 | -0.6606 | -0.1613 | -0.2009 | -0.1240 | — | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | half_width_bias | -0.5562 | -0.6027 | -0.0466 | -0.0744 | -0.0209 | — | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | midpoint_mae | 2.5289 | 2.5028 | -0.0262 | -0.0533 | 0.0061 | -1.0346 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | midpoint_crps | 1.8982 | 1.8839 | -0.0143 | -0.0329 | 0.0061 | -0.7552 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | range_mae | 2.7885 | 2.7462 | -0.0423 | -0.0953 | 0.0060 | -1.5181 | 76 |
| all27 | 200017 | 29 | 2 | ce_vs_current | legal_fraction | 0.9739 | 0.9764 | 0.0025 | 0.0016 | 0.0035 | 0.2583 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | endpoint_mae | 2.6936 | 2.6687 | -0.0249 | -0.0506 | -0.0004 | -0.9257 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | high_mae | 3.1251 | 3.1235 | -0.0016 | -0.0231 | 0.0193 | -0.0525 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | low_mae | 2.2621 | 2.2139 | -0.0482 | -0.0943 | -0.0104 | -2.1321 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | center_bias | -0.6606 | -0.5862 | 0.0744 | 0.0259 | 0.1316 | — | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | half_width_bias | -0.6027 | -0.6362 | -0.0334 | -0.0534 | -0.0169 | — | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | midpoint_mae | 2.5028 | 2.4692 | -0.0336 | -0.0687 | -0.0008 | -1.3428 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | midpoint_crps | 1.8839 | 1.8676 | -0.0163 | -0.0387 | 0.0024 | -0.8634 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | range_mae | 2.7462 | 2.7227 | -0.0235 | -0.0467 | -0.0019 | -0.8549 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_ce | legal_fraction | 0.9764 | 0.9775 | 0.0011 | -0.0000 | 0.0024 | 0.1134 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | endpoint_mae | 2.7219 | 2.6687 | -0.0532 | -0.0899 | -0.0219 | -1.9543 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | high_mae | 3.1541 | 3.1235 | -0.0306 | -0.0565 | -0.0035 | -0.9708 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | low_mae | 2.2897 | 2.2139 | -0.0758 | -0.1355 | -0.0259 | -3.3092 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | center_bias | -0.4994 | -0.5862 | -0.0869 | -0.1218 | -0.0439 | — | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | half_width_bias | -0.5562 | -0.6362 | -0.0800 | -0.1211 | -0.0446 | — | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | midpoint_mae | 2.5289 | 2.4692 | -0.0598 | -0.1016 | -0.0197 | -2.3635 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | midpoint_crps | 1.8982 | 1.8676 | -0.0306 | -0.0588 | -0.0046 | -1.6121 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | range_mae | 2.7885 | 2.7227 | -0.0658 | -0.1303 | -0.0164 | -2.3601 | 76 |
| all27 | 200017 | 29 | 2 | midpoint_vs_current | legal_fraction | 0.9739 | 0.9775 | 0.0036 | 0.0022 | 0.0050 | 0.3720 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | endpoint_mae | 4.2049 | 4.1816 | -0.0233 | -0.0889 | 0.0426 | -0.5540 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | high_mae | 5.0373 | 5.0075 | -0.0298 | -0.1074 | 0.0420 | -0.5916 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | low_mae | 3.3725 | 3.3557 | -0.0168 | -0.1102 | 0.0752 | -0.4978 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | center_bias | -1.3418 | -1.7128 | -0.3710 | -0.4357 | -0.2992 | — | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | half_width_bias | -0.8248 | -1.0178 | -0.1930 | -0.2972 | -0.1066 | — | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | midpoint_mae | 3.9071 | 3.9003 | -0.0069 | -0.0845 | 0.0730 | -0.1760 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | midpoint_crps | 2.9147 | 2.9183 | 0.0035 | -0.0404 | 0.0489 | 0.1201 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | range_mae | 4.8514 | 4.7237 | -0.1277 | -0.2960 | 0.0203 | -2.6325 | 76 |
| all27 | 200017 | 29 | 5 | ce_vs_current | legal_fraction | 0.9363 | 0.9444 | 0.0081 | 0.0059 | 0.0106 | 0.8689 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | endpoint_mae | 4.1816 | 4.0841 | -0.0975 | -0.1962 | -0.0202 | -2.3321 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | high_mae | 5.0075 | 4.9754 | -0.0321 | -0.0905 | 0.0147 | -0.6410 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | low_mae | 3.3557 | 3.1928 | -0.1629 | -0.3227 | -0.0383 | -4.8555 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | center_bias | -1.7128 | -1.5318 | 0.1810 | 0.0746 | 0.3165 | — | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | half_width_bias | -1.0178 | -1.1225 | -0.1046 | -0.1450 | -0.0639 | — | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | midpoint_mae | 3.9003 | 3.7869 | -0.1134 | -0.2340 | -0.0203 | -2.9077 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | midpoint_crps | 2.9183 | 2.8576 | -0.0606 | -0.1344 | -0.0060 | -2.0767 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | range_mae | 4.7237 | 4.6694 | -0.0543 | -0.0979 | -0.0155 | -1.1489 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_ce | legal_fraction | 0.9444 | 0.9474 | 0.0030 | 0.0008 | 0.0050 | 0.3140 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | endpoint_mae | 4.2049 | 4.0841 | -0.1208 | -0.2540 | -0.0209 | -2.8731 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | high_mae | 5.0373 | 4.9754 | -0.0619 | -0.1298 | 0.0013 | -1.2288 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | low_mae | 3.3725 | 3.1928 | -0.1797 | -0.4014 | -0.0170 | -5.3291 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | center_bias | -1.3418 | -1.5318 | -0.1900 | -0.2763 | -0.0885 | — | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | half_width_bias | -0.8248 | -1.1225 | -0.2977 | -0.4322 | -0.1865 | — | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | midpoint_mae | 3.9071 | 3.7869 | -0.1203 | -0.2559 | -0.0127 | -3.0785 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | midpoint_crps | 2.9147 | 2.8576 | -0.0571 | -0.1388 | 0.0084 | -1.9591 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | range_mae | 4.8514 | 4.6694 | -0.1820 | -0.3785 | -0.0191 | -3.7512 | 76 |
| all27 | 200017 | 29 | 5 | midpoint_vs_current | legal_fraction | 0.9363 | 0.9474 | 0.0111 | 0.0089 | 0.0136 | 1.1856 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | endpoint_mae | 2.7687 | 2.8413 | 0.0726 | 0.0208 | 0.1306 | 2.6233 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | high_mae | 3.1343 | 3.1500 | 0.0157 | -0.0167 | 0.0514 | 0.5012 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | low_mae | 2.4030 | 2.5326 | 0.1296 | 0.0409 | 0.2430 | 5.3913 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | center_bias | -0.7993 | -0.8678 | -0.0686 | -0.2052 | 0.0475 | — | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | half_width_bias | -0.5703 | -0.4369 | 0.1334 | 0.0931 | 0.1907 | — | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | midpoint_mae | 2.5833 | 2.6446 | 0.0612 | 0.0081 | 0.1159 | 2.3701 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | midpoint_crps | 1.9510 | 1.9954 | 0.0444 | 0.0057 | 0.0860 | 2.2747 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | range_mae | 2.8202 | 2.8945 | 0.0743 | -0.0010 | 0.1585 | 2.6346 | 76 |
| all27 | 200017 | 43 | 2 | ce_vs_current | legal_fraction | 0.9722 | 0.9744 | 0.0022 | 0.0011 | 0.0033 | 0.2225 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | endpoint_mae | 2.8413 | 2.8925 | 0.0512 | 0.0099 | 0.0975 | 1.8022 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | high_mae | 3.1500 | 3.1643 | 0.0143 | -0.0161 | 0.0383 | 0.4547 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | low_mae | 2.5326 | 2.6207 | 0.0881 | 0.0213 | 0.1683 | 3.4782 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | center_bias | -0.8678 | -0.9980 | -0.1302 | -0.1734 | -0.0864 | — | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | half_width_bias | -0.4369 | -0.3973 | 0.0396 | 0.0113 | 0.0705 | — | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | midpoint_mae | 2.6446 | 2.7071 | 0.0626 | 0.0163 | 0.1154 | 2.3653 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | midpoint_crps | 1.9954 | 2.0268 | 0.0314 | 0.0070 | 0.0605 | 1.5743 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | range_mae | 2.8945 | 2.9509 | 0.0564 | 0.0159 | 0.1062 | 1.9493 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_ce | legal_fraction | 0.9744 | 0.9737 | -0.0006 | -0.0016 | 0.0003 | -0.0638 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | endpoint_mae | 2.7687 | 2.8925 | 0.1238 | 0.0451 | 0.2186 | 4.4728 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | high_mae | 3.1343 | 3.1643 | 0.0300 | -0.0005 | 0.0589 | 0.9582 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | low_mae | 2.4030 | 2.6207 | 0.2176 | 0.0755 | 0.4011 | 9.0570 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | center_bias | -0.7993 | -0.9980 | -0.1987 | -0.3554 | -0.0657 | — | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | half_width_bias | -0.5703 | -0.3973 | 0.1730 | 0.1107 | 0.2495 | — | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | midpoint_mae | 2.5833 | 2.7071 | 0.1238 | 0.0431 | 0.2206 | 4.7915 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | midpoint_crps | 1.9510 | 2.0268 | 0.0758 | 0.0233 | 0.1431 | 3.8849 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | range_mae | 2.8202 | 2.9509 | 0.1307 | 0.0426 | 0.2444 | 4.6353 | 76 |
| all27 | 200017 | 43 | 2 | midpoint_vs_current | legal_fraction | 0.9722 | 0.9737 | 0.0015 | 0.0004 | 0.0028 | 0.1585 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | endpoint_mae | 4.3827 | 4.6072 | 0.2245 | 0.0810 | 0.4020 | 5.1230 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | high_mae | 5.0260 | 5.1154 | 0.0894 | 0.0375 | 0.1394 | 1.7788 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | low_mae | 3.7394 | 4.0990 | 0.3596 | 0.0758 | 0.7281 | 9.6177 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | center_bias | -2.0745 | -2.2227 | -0.1482 | -0.4688 | 0.1372 | — | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | half_width_bias | -0.8474 | -0.4951 | 0.3523 | 0.2538 | 0.4865 | — | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | midpoint_mae | 4.1010 | 4.3032 | 0.2022 | 0.0721 | 0.3737 | 4.9313 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | midpoint_crps | 3.0693 | 3.1973 | 0.1280 | 0.0315 | 0.2537 | 4.1708 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | range_mae | 4.9593 | 5.1940 | 0.2347 | 0.0430 | 0.4771 | 4.7327 | 76 |
| all27 | 200017 | 43 | 5 | ce_vs_current | legal_fraction | 0.9314 | 0.9344 | 0.0030 | 0.0004 | 0.0058 | 0.3253 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | endpoint_mae | 4.6072 | 4.7384 | 0.1312 | 0.0291 | 0.2468 | 2.8487 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | high_mae | 5.1154 | 5.1297 | 0.0143 | -0.0530 | 0.0833 | 0.2800 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | low_mae | 4.0990 | 4.3472 | 0.2482 | 0.0941 | 0.4347 | 6.0543 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | center_bias | -2.2227 | -2.5467 | -0.3240 | -0.4513 | -0.2032 | — | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | half_width_bias | -0.4951 | -0.3956 | 0.0994 | 0.0222 | 0.1963 | — | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | midpoint_mae | 4.3032 | 4.4321 | 0.1289 | 0.0189 | 0.2604 | 2.9956 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | midpoint_crps | 3.1973 | 3.2943 | 0.0970 | 0.0325 | 0.1792 | 3.0338 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | range_mae | 5.1940 | 5.3394 | 0.1454 | 0.0504 | 0.2697 | 2.7989 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_ce | legal_fraction | 0.9344 | 0.9339 | -0.0006 | -0.0023 | 0.0013 | -0.0620 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | endpoint_mae | 4.3827 | 4.7384 | 0.3558 | 0.1452 | 0.6314 | 8.1176 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | high_mae | 5.0260 | 5.1297 | 0.1037 | 0.0359 | 0.1705 | 2.0637 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | low_mae | 3.7394 | 4.3472 | 0.6078 | 0.2153 | 1.1189 | 16.2543 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | center_bias | -2.0745 | -2.5467 | -0.4722 | -0.8427 | -0.1639 | — | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | half_width_bias | -0.8474 | -0.3956 | 0.4518 | 0.2871 | 0.6679 | — | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | midpoint_mae | 4.1010 | 4.4321 | 0.3311 | 0.1206 | 0.6097 | 8.0746 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | midpoint_crps | 3.0693 | 3.2943 | 0.2250 | 0.0789 | 0.4266 | 7.3312 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | range_mae | 4.9593 | 5.3394 | 0.3801 | 0.1444 | 0.7175 | 7.6640 | 76 |
| all27 | 200017 | 43 | 5 | midpoint_vs_current | legal_fraction | 0.9314 | 0.9339 | 0.0025 | 0.0005 | 0.0048 | 0.2631 | 76 |

## Score means

| cohort | draw | seed | owner | horizon | endpoint_mae | high_mae | low_mae | center_bias | half_width_bias | midpoint_mae | midpoint_crps | range_mae | legal_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all27 | 0 | 0 | ce | 2 | 2.7416 | 3.1339 | 2.3493 | -0.7075 | -0.5458 | 2.5537 | 1.9251 | 2.7969 | 0.9753 |
| all27 | 0 | 0 | ce | 5 | 4.3135 | 5.0490 | 3.5780 | -1.8158 | -0.8367 | 4.0184 | 3.0031 | 4.8874 | 0.9393 |
| all27 | 0 | 0 | current | 2 | 2.7314 | 3.1412 | 2.3216 | -0.6491 | -0.5681 | 2.5426 | 1.9175 | 2.7881 | 0.9730 |
| all27 | 0 | 0 | current | 5 | 4.2731 | 5.0499 | 3.4963 | -1.6792 | -0.8720 | 3.9752 | 2.9730 | 4.8669 | 0.9342 |
| all27 | 0 | 0 | midpoint | 2 | 2.7497 | 3.1382 | 2.3612 | -0.7251 | -0.5470 | 2.5613 | 1.9318 | 2.8094 | 0.9750 |
| all27 | 0 | 0 | midpoint | 5 | 4.3281 | 5.0515 | 3.6046 | -1.8693 | -0.8388 | 4.0347 | 3.0191 | 4.9207 | 0.9385 |
| all27 | 0 | 17 | ce | 2 | 2.6947 | 3.1312 | 2.2582 | -0.5988 | -0.6024 | 2.5085 | 1.8921 | 2.7432 | 0.9750 |
| all27 | 0 | 17 | ce | 5 | 4.1721 | 5.0386 | 3.3056 | -1.5209 | -0.9938 | 3.8890 | 2.9104 | 4.7376 | 0.9389 |
| all27 | 0 | 17 | current | 2 | 2.7046 | 3.1360 | 2.2731 | -0.6431 | -0.5888 | 2.5177 | 1.9032 | 2.7601 | 0.9738 |
| all27 | 0 | 17 | current | 5 | 4.2015 | 5.0592 | 3.3438 | -1.6164 | -0.9514 | 3.9075 | 2.9259 | 4.7958 | 0.9359 |
| all27 | 0 | 17 | midpoint | 2 | 2.6916 | 3.1314 | 2.2518 | -0.5902 | -0.6071 | 2.4999 | 1.8926 | 2.7586 | 0.9737 |
| all27 | 0 | 17 | midpoint | 5 | 4.1568 | 5.0420 | 3.2716 | -1.5292 | -1.0057 | 3.8795 | 2.9078 | 4.7705 | 0.9352 |
| all27 | 0 | 29 | ce | 2 | 2.6962 | 3.1239 | 2.2685 | -0.6558 | -0.5974 | 2.5098 | 1.8917 | 2.7498 | 0.9767 |
| all27 | 0 | 29 | ce | 5 | 4.1786 | 5.0070 | 3.3502 | -1.6992 | -1.0146 | 3.8875 | 2.9150 | 4.7258 | 0.9446 |
| all27 | 0 | 29 | current | 2 | 2.7208 | 3.1558 | 2.2858 | -0.5010 | -0.5506 | 2.5265 | 1.8983 | 2.7917 | 0.9735 |
| all27 | 0 | 29 | current | 5 | 4.2230 | 5.0478 | 3.3981 | -1.3519 | -0.8188 | 3.9119 | 2.9222 | 4.8592 | 0.9353 |
| all27 | 0 | 29 | midpoint | 2 | 2.6693 | 3.1162 | 2.2223 | -0.5885 | -0.6321 | 2.4830 | 1.8739 | 2.7231 | 0.9774 |
| all27 | 0 | 29 | midpoint | 5 | 4.0894 | 4.9794 | 3.1994 | -1.5263 | -1.1098 | 3.8057 | 2.8614 | 4.6655 | 0.9469 |
| all27 | 0 | 43 | ce | 2 | 2.8339 | 3.1467 | 2.5211 | -0.8679 | -0.4376 | 2.6428 | 1.9916 | 2.8978 | 0.9740 |
| all27 | 0 | 43 | ce | 5 | 4.5898 | 5.1014 | 4.0783 | -2.2272 | -0.5017 | 4.2787 | 3.1840 | 5.1989 | 0.9345 |
| all27 | 0 | 43 | current | 2 | 2.7688 | 3.1316 | 2.4060 | -0.8031 | -0.5650 | 2.5835 | 1.9510 | 2.8125 | 0.9718 |
| all27 | 0 | 43 | current | 5 | 4.3949 | 5.0429 | 3.7470 | -2.0694 | -0.8459 | 4.1062 | 3.0711 | 4.9457 | 0.9315 |
| all27 | 0 | 43 | midpoint | 2 | 2.8882 | 3.1668 | 2.6095 | -0.9967 | -0.4018 | 2.7009 | 2.0288 | 2.9467 | 0.9740 |
| all27 | 0 | 43 | midpoint | 5 | 4.7381 | 5.1331 | 4.3430 | -2.5525 | -0.4009 | 4.4189 | 3.2882 | 5.3260 | 0.9334 |
| all27 | 17 | 0 | ce | 2 | 2.7455 | 3.1344 | 2.3566 | -0.7126 | -0.5478 | 2.5600 | 1.9322 | 2.7967 | 0.9754 |
| all27 | 17 | 0 | ce | 5 | 4.3166 | 5.0525 | 3.5807 | -1.8267 | -0.8369 | 4.0215 | 3.0082 | 4.8785 | 0.9392 |
| all27 | 17 | 0 | current | 2 | 2.7367 | 3.1441 | 2.3293 | -0.6491 | -0.5679 | 2.5439 | 1.9211 | 2.7843 | 0.9730 |
| all27 | 17 | 0 | current | 5 | 4.2844 | 5.0668 | 3.5020 | -1.6767 | -0.8738 | 3.9867 | 2.9797 | 4.8724 | 0.9342 |
| all27 | 17 | 0 | midpoint | 2 | 2.7534 | 3.1416 | 2.3653 | -0.7254 | -0.5512 | 2.5675 | 1.9373 | 2.8088 | 0.9753 |
| all27 | 17 | 0 | midpoint | 5 | 4.3304 | 5.0601 | 3.6007 | -1.8652 | -0.8396 | 4.0418 | 3.0228 | 4.9319 | 0.9382 |
| all27 | 17 | 17 | ce | 2 | 2.7052 | 3.1424 | 2.2680 | -0.6159 | -0.6014 | 2.5204 | 1.9044 | 2.7463 | 0.9751 |
| all27 | 17 | 17 | ce | 5 | 4.1855 | 5.0587 | 3.3123 | -1.5448 | -0.9996 | 3.9098 | 2.9270 | 4.7239 | 0.9388 |
| all27 | 17 | 17 | current | 2 | 2.7104 | 3.1445 | 2.2764 | -0.6427 | -0.5872 | 2.5207 | 1.9078 | 2.7599 | 0.9740 |
| all27 | 17 | 17 | current | 5 | 4.2065 | 5.0726 | 3.3405 | -1.6116 | -0.9512 | 3.9237 | 2.9339 | 4.7905 | 0.9360 |
| all27 | 17 | 17 | midpoint | 2 | 2.7005 | 3.1413 | 2.2597 | -0.5979 | -0.6097 | 2.5075 | 1.8998 | 2.7524 | 0.9738 |
| all27 | 17 | 17 | midpoint | 5 | 4.1691 | 5.0616 | 3.2766 | -1.5392 | -1.0024 | 3.9056 | 2.9172 | 4.7720 | 0.9344 |
| all27 | 17 | 29 | ce | 2 | 2.6989 | 3.1200 | 2.2778 | -0.6415 | -0.6017 | 2.5145 | 1.8990 | 2.7528 | 0.9772 |
| all27 | 17 | 29 | ce | 5 | 4.1695 | 4.9940 | 3.3450 | -1.6798 | -1.0072 | 3.8775 | 2.9090 | 4.7237 | 0.9446 |
| all27 | 17 | 29 | current | 2 | 2.7205 | 3.1548 | 2.2862 | -0.4951 | -0.5539 | 2.5181 | 1.8965 | 2.7915 | 0.9734 |
| all27 | 17 | 29 | current | 5 | 4.2277 | 5.0455 | 3.4099 | -1.3494 | -0.8192 | 3.9182 | 2.9267 | 4.8805 | 0.9345 |
| all27 | 17 | 29 | midpoint | 2 | 2.6698 | 3.1135 | 2.2261 | -0.5832 | -0.6395 | 2.4926 | 1.8781 | 2.7230 | 0.9777 |
| all27 | 17 | 29 | midpoint | 5 | 4.0867 | 4.9708 | 3.2025 | -1.5154 | -1.1025 | 3.8049 | 2.8609 | 4.6731 | 0.9469 |
| all27 | 17 | 43 | ce | 2 | 2.8324 | 3.1409 | 2.5238 | -0.8804 | -0.4402 | 2.6452 | 1.9933 | 2.8911 | 0.9741 |
| all27 | 17 | 43 | ce | 5 | 4.5948 | 5.1048 | 4.0848 | -2.2555 | -0.5040 | 4.2773 | 3.1887 | 5.1877 | 0.9343 |
| all27 | 17 | 43 | current | 2 | 2.7792 | 3.1329 | 2.4254 | -0.8094 | -0.5625 | 2.5929 | 1.9590 | 2.8015 | 0.9715 |
| all27 | 17 | 43 | current | 5 | 4.4189 | 5.0822 | 3.7557 | -2.0692 | -0.8509 | 4.1181 | 3.0785 | 4.9462 | 0.9320 |
| all27 | 17 | 43 | midpoint | 2 | 2.8899 | 3.1700 | 2.6099 | -0.9951 | -0.4043 | 2.7024 | 2.0339 | 2.9511 | 0.9743 |
| all27 | 17 | 43 | midpoint | 5 | 4.7353 | 5.1478 | 4.3228 | -2.5410 | -0.4139 | 4.4149 | 3.2903 | 5.3505 | 0.9332 |
| all27 | 100017 | 0 | ce | 2 | 2.7375 | 3.1332 | 2.3418 | -0.7001 | -0.5406 | 2.5508 | 1.9201 | 2.7971 | 0.9751 |
| all27 | 100017 | 0 | ce | 5 | 4.3066 | 5.0462 | 3.5670 | -1.8024 | -0.8359 | 4.0043 | 2.9920 | 4.8920 | 0.9395 |
| all27 | 100017 | 0 | current | 2 | 2.7258 | 3.1359 | 2.3158 | -0.6466 | -0.5626 | 2.5415 | 1.9135 | 2.7875 | 0.9729 |
| all27 | 100017 | 0 | current | 5 | 4.2727 | 5.0428 | 3.5025 | -1.6805 | -0.8634 | 3.9724 | 2.9698 | 4.8566 | 0.9340 |
| all27 | 100017 | 0 | midpoint | 2 | 2.7444 | 3.1330 | 2.3559 | -0.7239 | -0.5451 | 2.5577 | 1.9297 | 2.8057 | 0.9749 |
| all27 | 100017 | 0 | midpoint | 5 | 4.3320 | 5.0513 | 3.6128 | -1.8736 | -0.8357 | 4.0327 | 3.0149 | 4.9061 | 0.9387 |
| all27 | 100017 | 17 | ce | 2 | 2.6884 | 3.1238 | 2.2531 | -0.5795 | -0.5984 | 2.5016 | 1.8817 | 2.7331 | 0.9750 |
| all27 | 100017 | 17 | ce | 5 | 4.1674 | 5.0348 | 3.3000 | -1.4988 | -0.9829 | 3.8725 | 2.8922 | 4.7313 | 0.9387 |
| all27 | 100017 | 17 | current | 2 | 2.6988 | 3.1214 | 2.2762 | -0.6305 | -0.5839 | 2.5179 | 1.8975 | 2.7514 | 0.9737 |
| all27 | 100017 | 17 | current | 5 | 4.1985 | 5.0473 | 3.3498 | -1.6126 | -0.9386 | 3.9073 | 2.9188 | 4.7925 | 0.9357 |
| all27 | 100017 | 17 | midpoint | 2 | 2.6819 | 3.1211 | 2.2426 | -0.5786 | -0.6110 | 2.4926 | 1.8874 | 2.7555 | 0.9739 |
| all27 | 100017 | 17 | midpoint | 5 | 4.1583 | 5.0400 | 3.2766 | -1.5192 | -1.0096 | 3.8631 | 2.8994 | 4.7762 | 0.9364 |
| all27 | 100017 | 29 | ce | 2 | 2.6961 | 3.1266 | 2.2656 | -0.6653 | -0.5877 | 2.5122 | 1.8923 | 2.7503 | 0.9766 |
| all27 | 100017 | 29 | ce | 5 | 4.1847 | 5.0196 | 3.3498 | -1.7049 | -1.0188 | 3.8848 | 2.9177 | 4.7300 | 0.9448 |
| all27 | 100017 | 29 | current | 2 | 2.7201 | 3.1586 | 2.2817 | -0.5087 | -0.5418 | 2.5325 | 1.9002 | 2.7952 | 0.9731 |
| all27 | 100017 | 29 | current | 5 | 4.2363 | 5.0605 | 3.4120 | -1.3644 | -0.8122 | 3.9104 | 2.9251 | 4.8456 | 0.9351 |
| all27 | 100017 | 29 | midpoint | 2 | 2.6693 | 3.1117 | 2.2270 | -0.5961 | -0.6206 | 2.4873 | 1.8759 | 2.7235 | 0.9770 |
| all27 | 100017 | 29 | midpoint | 5 | 4.0974 | 4.9920 | 3.2028 | -1.5317 | -1.1043 | 3.8252 | 2.8655 | 4.6541 | 0.9465 |
| all27 | 100017 | 43 | ce | 2 | 2.8280 | 3.1492 | 2.5068 | -0.8555 | -0.4356 | 2.6386 | 1.9861 | 2.9078 | 0.9735 |
| all27 | 100017 | 43 | ce | 5 | 4.5675 | 5.0840 | 4.0511 | -2.2034 | -0.5060 | 4.2557 | 3.1661 | 5.2148 | 0.9348 |
| all27 | 100017 | 43 | current | 2 | 2.7586 | 3.1277 | 2.3896 | -0.8007 | -0.5623 | 2.5742 | 1.9429 | 2.8160 | 0.9718 |
| all27 | 100017 | 43 | current | 5 | 4.3832 | 5.0205 | 3.7458 | -2.0646 | -0.8392 | 4.0994 | 3.0654 | 4.9317 | 0.9311 |
| all27 | 100017 | 43 | midpoint | 2 | 2.8821 | 3.1662 | 2.5980 | -0.9970 | -0.4037 | 2.6931 | 2.0258 | 2.9380 | 0.9739 |
| all27 | 100017 | 43 | midpoint | 5 | 4.7404 | 5.1218 | 4.3590 | -2.5698 | -0.3932 | 4.4097 | 3.2799 | 5.2879 | 0.9331 |
| all27 | 200017 | 0 | ce | 2 | 2.7418 | 3.1342 | 2.3494 | -0.7098 | -0.5491 | 2.5503 | 1.9231 | 2.7970 | 0.9753 |
| all27 | 200017 | 0 | ce | 5 | 4.3174 | 5.0484 | 3.5864 | -1.8182 | -0.8373 | 4.0294 | 3.0091 | 4.8917 | 0.9393 |
| all27 | 200017 | 0 | current | 2 | 2.7317 | 3.1435 | 2.3198 | -0.6515 | -0.5739 | 2.5422 | 1.9178 | 2.7925 | 0.9733 |
| all27 | 200017 | 0 | current | 5 | 4.2623 | 5.0403 | 3.4844 | -1.6804 | -0.8789 | 3.9665 | 2.9696 | 4.8717 | 0.9345 |
| all27 | 200017 | 0 | midpoint | 2 | 2.7512 | 3.1399 | 2.3626 | -0.7260 | -0.5447 | 2.5587 | 1.9283 | 2.8138 | 0.9749 |
| all27 | 200017 | 0 | midpoint | 5 | 4.3218 | 5.0432 | 3.6005 | -1.8692 | -0.8410 | 4.0296 | 3.0196 | 4.9241 | 0.9387 |
| all27 | 200017 | 17 | ce | 2 | 2.6905 | 3.1275 | 2.2535 | -0.6009 | -0.6075 | 2.5036 | 1.8900 | 2.7503 | 0.9750 |
| all27 | 200017 | 17 | ce | 5 | 4.1634 | 5.0223 | 3.3045 | -1.5191 | -0.9989 | 3.8848 | 2.9119 | 4.7574 | 0.9391 |
| all27 | 200017 | 17 | current | 2 | 2.7044 | 3.1422 | 2.2666 | -0.6560 | -0.5953 | 2.5144 | 1.9042 | 2.7689 | 0.9737 |
| all27 | 200017 | 17 | current | 5 | 4.1995 | 5.0577 | 3.3412 | -1.6249 | -0.9644 | 3.8914 | 2.9249 | 4.8044 | 0.9359 |
| all27 | 200017 | 17 | midpoint | 2 | 2.6925 | 3.1318 | 2.2532 | -0.5939 | -0.6007 | 2.4997 | 1.8905 | 2.7678 | 0.9733 |
| all27 | 200017 | 17 | midpoint | 5 | 4.1430 | 5.0244 | 3.2615 | -1.5292 | -1.0050 | 3.8698 | 2.9068 | 4.7633 | 0.9349 |
| all27 | 200017 | 29 | ce | 2 | 2.6936 | 3.1251 | 2.2621 | -0.6606 | -0.6027 | 2.5028 | 1.8839 | 2.7462 | 0.9764 |
| all27 | 200017 | 29 | ce | 5 | 4.1816 | 5.0075 | 3.3557 | -1.7128 | -1.0178 | 3.9003 | 2.9183 | 4.7237 | 0.9444 |
| all27 | 200017 | 29 | current | 2 | 2.7219 | 3.1541 | 2.2897 | -0.4994 | -0.5562 | 2.5289 | 1.8982 | 2.7885 | 0.9739 |
| all27 | 200017 | 29 | current | 5 | 4.2049 | 5.0373 | 3.3725 | -1.3418 | -0.8248 | 3.9071 | 2.9147 | 4.8514 | 0.9363 |
| all27 | 200017 | 29 | midpoint | 2 | 2.6687 | 3.1235 | 2.2139 | -0.5862 | -0.6362 | 2.4692 | 1.8676 | 2.7227 | 0.9775 |
| all27 | 200017 | 29 | midpoint | 5 | 4.0841 | 4.9754 | 3.1928 | -1.5318 | -1.1225 | 3.7869 | 2.8576 | 4.6694 | 0.9474 |
| all27 | 200017 | 43 | ce | 2 | 2.8413 | 3.1500 | 2.5326 | -0.8678 | -0.4369 | 2.6446 | 1.9954 | 2.8945 | 0.9744 |
| all27 | 200017 | 43 | ce | 5 | 4.6072 | 5.1154 | 4.0990 | -2.2227 | -0.4951 | 4.3032 | 3.1973 | 5.1940 | 0.9344 |
| all27 | 200017 | 43 | current | 2 | 2.7687 | 3.1343 | 2.4030 | -0.7993 | -0.5703 | 2.5833 | 1.9510 | 2.8202 | 0.9722 |
| all27 | 200017 | 43 | current | 5 | 4.3827 | 5.0260 | 3.7394 | -2.0745 | -0.8474 | 4.1010 | 3.0693 | 4.9593 | 0.9314 |
| all27 | 200017 | 43 | midpoint | 2 | 2.8925 | 3.1643 | 2.6207 | -0.9980 | -0.3973 | 2.7071 | 2.0268 | 2.9509 | 0.9737 |
| all27 | 200017 | 43 | midpoint | 5 | 4.7384 | 5.1297 | 4.3472 | -2.5467 | -0.3956 | 4.4321 | 3.2943 | 5.3394 | 0.9339 |
| within_draw_seed | 0 | 0 | ce | 2 | 2.7452 | 3.1389 | 2.3515 | -0.7005 | -0.5426 | 2.5570 | 1.9274 | 2.7995 | 0.9739 |
| within_draw_seed | 0 | 0 | ce | 5 | 4.3365 | 5.0844 | 3.5886 | -1.8268 | -0.8445 | 4.0358 | 3.0169 | 4.9121 | 0.9365 |
| within_draw_seed | 0 | 0 | current | 2 | 2.7343 | 3.1447 | 2.3240 | -0.6432 | -0.5654 | 2.5457 | 1.9196 | 2.7908 | 0.9717 |
| within_draw_seed | 0 | 0 | current | 5 | 4.2934 | 5.0800 | 3.5068 | -1.6872 | -0.8752 | 3.9880 | 2.9848 | 4.8880 | 0.9313 |
| within_draw_seed | 0 | 0 | midpoint | 2 | 2.7537 | 3.1437 | 2.3638 | -0.7180 | -0.5433 | 2.5649 | 1.9345 | 2.8120 | 0.9737 |
| within_draw_seed | 0 | 0 | midpoint | 5 | 4.3507 | 5.0868 | 3.6146 | -1.8807 | -0.8477 | 4.0514 | 3.0328 | 4.9459 | 0.9357 |
| within_draw_seed | 0 | 17 | ce | 2 | 2.7003 | 3.1378 | 2.2627 | -0.5910 | -0.5996 | 2.5141 | 1.8955 | 2.7466 | 0.9737 |
| within_draw_seed | 0 | 17 | ce | 5 | 4.1922 | 5.0707 | 3.3137 | -1.5282 | -0.9985 | 3.9033 | 2.9192 | 4.7566 | 0.9359 |
| within_draw_seed | 0 | 17 | current | 2 | 2.7081 | 3.1405 | 2.2757 | -0.6391 | -0.5847 | 2.5203 | 1.9061 | 2.7648 | 0.9724 |
| within_draw_seed | 0 | 17 | current | 5 | 4.2136 | 5.0746 | 3.3526 | -1.6135 | -0.9428 | 3.9093 | 2.9295 | 4.8075 | 0.9328 |
| within_draw_seed | 0 | 17 | midpoint | 2 | 2.6973 | 3.1394 | 2.2552 | -0.5839 | -0.6017 | 2.5041 | 1.8960 | 2.7627 | 0.9723 |
| within_draw_seed | 0 | 17 | midpoint | 5 | 4.1761 | 5.0733 | 3.2789 | -1.5373 | -1.0105 | 3.8930 | 2.9152 | 4.7873 | 0.9323 |
| within_draw_seed | 0 | 29 | ce | 2 | 2.7015 | 3.1329 | 2.2702 | -0.6462 | -0.5913 | 2.5141 | 1.8950 | 2.7563 | 0.9753 |
| within_draw_seed | 0 | 29 | ce | 5 | 4.2019 | 5.0406 | 3.3632 | -1.7102 | -1.0215 | 3.9034 | 2.9267 | 4.7514 | 0.9420 |
| within_draw_seed | 0 | 29 | current | 2 | 2.7259 | 3.1624 | 2.2894 | -0.4915 | -0.5472 | 2.5328 | 1.9016 | 2.7969 | 0.9720 |
| within_draw_seed | 0 | 29 | current | 5 | 4.2437 | 5.0801 | 3.4072 | -1.3630 | -0.8276 | 3.9257 | 2.9334 | 4.8827 | 0.9326 |
| within_draw_seed | 0 | 29 | midpoint | 2 | 2.6755 | 3.1248 | 2.2261 | -0.5778 | -0.6275 | 2.4894 | 1.8783 | 2.7285 | 0.9760 |
| within_draw_seed | 0 | 29 | midpoint | 5 | 4.1106 | 5.0108 | 3.2103 | -1.5375 | -1.1185 | 3.8193 | 2.8728 | 4.6915 | 0.9443 |
| within_draw_seed | 0 | 43 | ce | 2 | 2.8337 | 3.1458 | 2.5215 | -0.8643 | -0.4369 | 2.6428 | 1.9916 | 2.8958 | 0.9729 |
| within_draw_seed | 0 | 43 | ce | 5 | 4.6155 | 5.1419 | 4.0890 | -2.2420 | -0.5135 | 4.3006 | 3.2049 | 5.2281 | 0.9316 |
| within_draw_seed | 0 | 43 | current | 2 | 2.7690 | 3.1311 | 2.4068 | -0.7991 | -0.5644 | 2.5841 | 1.9510 | 2.8107 | 0.9707 |
| within_draw_seed | 0 | 43 | current | 5 | 4.4230 | 5.0854 | 3.7606 | -2.0851 | -0.8551 | 4.1292 | 3.0916 | 4.9738 | 0.9286 |
| within_draw_seed | 0 | 43 | midpoint | 2 | 2.8884 | 3.1668 | 2.6101 | -0.9924 | -0.4008 | 2.7011 | 2.0291 | 2.9447 | 0.9729 |
| within_draw_seed | 0 | 43 | midpoint | 5 | 4.7654 | 5.1762 | 4.3546 | -2.5672 | -0.4140 | 4.4418 | 3.3104 | 5.3590 | 0.9305 |
| within_draw_seed | 17 | 0 | ce | 2 | 2.7506 | 3.1409 | 2.3603 | -0.7031 | -0.5445 | 2.5653 | 1.9350 | 2.7994 | 0.9741 |
| within_draw_seed | 17 | 0 | ce | 5 | 4.3450 | 5.0924 | 3.5977 | -1.8459 | -0.8469 | 4.0436 | 3.0279 | 4.9081 | 0.9360 |
| within_draw_seed | 17 | 0 | current | 2 | 2.7402 | 3.1485 | 2.3319 | -0.6412 | -0.5650 | 2.5478 | 1.9235 | 2.7868 | 0.9717 |
| within_draw_seed | 17 | 0 | current | 5 | 4.3080 | 5.1013 | 3.5148 | -1.6886 | -0.8786 | 4.0037 | 2.9956 | 4.8985 | 0.9309 |
| within_draw_seed | 17 | 0 | midpoint | 2 | 2.7577 | 3.1471 | 2.3682 | -0.7177 | -0.5471 | 2.5707 | 1.9401 | 2.8114 | 0.9740 |
| within_draw_seed | 17 | 0 | midpoint | 5 | 4.3576 | 5.0996 | 3.6156 | -1.8838 | -0.8507 | 4.0642 | 3.0420 | 4.9597 | 0.9349 |
| within_draw_seed | 17 | 17 | ce | 2 | 2.7141 | 3.1532 | 2.2750 | -0.6028 | -0.5974 | 2.5296 | 1.9089 | 2.7512 | 0.9738 |
| within_draw_seed | 17 | 17 | ce | 5 | 4.2071 | 5.0886 | 3.3256 | -1.5588 | -1.0039 | 3.9247 | 2.9392 | 4.7413 | 0.9353 |
| within_draw_seed | 17 | 17 | current | 2 | 2.7133 | 3.1499 | 2.2766 | -0.6355 | -0.5826 | 2.5234 | 1.9106 | 2.7635 | 0.9727 |
| within_draw_seed | 17 | 17 | current | 5 | 4.2168 | 5.0818 | 3.3518 | -1.6081 | -0.9406 | 3.9250 | 2.9381 | 4.8003 | 0.9326 |
| within_draw_seed | 17 | 17 | midpoint | 2 | 2.7059 | 3.1500 | 2.2619 | -0.5905 | -0.6020 | 2.5094 | 1.9026 | 2.7583 | 0.9725 |
| within_draw_seed | 17 | 17 | midpoint | 5 | 4.1894 | 5.0911 | 3.2877 | -1.5531 | -1.0075 | 3.9203 | 2.9251 | 4.7830 | 0.9309 |
| within_draw_seed | 17 | 29 | ce | 2 | 2.7046 | 3.1286 | 2.2806 | -0.6314 | -0.5968 | 2.5202 | 1.9024 | 2.7577 | 0.9758 |
| within_draw_seed | 17 | 29 | ce | 5 | 4.2058 | 5.0426 | 3.3691 | -1.7089 | -1.0216 | 3.9042 | 2.9319 | 4.7668 | 0.9413 |
| within_draw_seed | 17 | 29 | current | 2 | 2.7271 | 3.1628 | 2.2915 | -0.4842 | -0.5507 | 2.5257 | 1.9004 | 2.7971 | 0.9722 |
| within_draw_seed | 17 | 29 | current | 5 | 4.2581 | 5.0928 | 3.4235 | -1.3754 | -0.8367 | 3.9383 | 2.9443 | 4.9242 | 0.9312 |
| within_draw_seed | 17 | 29 | midpoint | 2 | 2.6767 | 3.1221 | 2.2313 | -0.5721 | -0.6357 | 2.4998 | 1.8835 | 2.7273 | 0.9765 |
| within_draw_seed | 17 | 29 | midpoint | 5 | 4.1208 | 5.0191 | 3.2226 | -1.5423 | -1.1176 | 3.8315 | 2.8842 | 4.7183 | 0.9436 |
| within_draw_seed | 17 | 43 | ce | 2 | 2.8331 | 3.1409 | 2.5253 | -0.8751 | -0.4392 | 2.6460 | 1.9938 | 2.8892 | 0.9728 |
| within_draw_seed | 17 | 43 | ce | 5 | 4.6221 | 5.1459 | 4.0983 | -2.2700 | -0.5153 | 4.3018 | 3.2128 | 5.2163 | 0.9312 |
| within_draw_seed | 17 | 43 | current | 2 | 2.7802 | 3.1327 | 2.4276 | -0.8038 | -0.5619 | 2.5944 | 1.9596 | 2.7996 | 0.9703 |
| within_draw_seed | 17 | 43 | current | 5 | 4.4492 | 5.1293 | 3.7691 | -2.0822 | -0.8584 | 4.1479 | 3.1044 | 4.9709 | 0.9291 |
| within_draw_seed | 17 | 43 | midpoint | 2 | 2.8904 | 3.1692 | 2.6115 | -0.9904 | -0.4036 | 2.7029 | 2.0342 | 2.9485 | 0.9730 |
| within_draw_seed | 17 | 43 | midpoint | 5 | 4.7625 | 5.1887 | 4.3364 | -2.5561 | -0.4270 | 4.4407 | 3.3167 | 5.3777 | 0.9302 |
| within_draw_seed | 100017 | 0 | ce | 2 | 2.7395 | 3.1369 | 2.3420 | -0.6955 | -0.5371 | 2.5520 | 1.9213 | 2.7985 | 0.9738 |
| within_draw_seed | 100017 | 0 | ce | 5 | 4.3213 | 5.0716 | 3.5711 | -1.8070 | -0.8414 | 4.0143 | 2.9992 | 4.9111 | 0.9373 |
| within_draw_seed | 100017 | 0 | current | 2 | 2.7271 | 3.1379 | 2.3163 | -0.6433 | -0.5602 | 2.5436 | 1.9145 | 2.7886 | 0.9715 |
| within_draw_seed | 100017 | 0 | current | 5 | 4.2859 | 5.0626 | 3.5093 | -1.6865 | -0.8660 | 3.9794 | 2.9761 | 4.8716 | 0.9318 |
| within_draw_seed | 100017 | 0 | midpoint | 2 | 2.7471 | 3.1359 | 2.3583 | -0.7196 | -0.5426 | 2.5602 | 1.9312 | 2.8081 | 0.9736 |
| within_draw_seed | 100017 | 0 | midpoint | 5 | 4.3469 | 5.0761 | 3.6177 | -1.8781 | -0.8412 | 4.0417 | 3.0227 | 4.9284 | 0.9366 |
| within_draw_seed | 100017 | 17 | ce | 2 | 2.6888 | 3.1231 | 2.2545 | -0.5804 | -0.5971 | 2.5017 | 1.8822 | 2.7321 | 0.9737 |
| within_draw_seed | 100017 | 17 | ce | 5 | 4.1834 | 5.0643 | 3.3024 | -1.5017 | -0.9885 | 3.8822 | 2.8964 | 4.7505 | 0.9364 |
| within_draw_seed | 100017 | 17 | current | 2 | 2.7004 | 3.1220 | 2.2787 | -0.6317 | -0.5826 | 2.5194 | 1.8983 | 2.7506 | 0.9724 |
| within_draw_seed | 100017 | 17 | current | 5 | 4.2100 | 5.0643 | 3.3557 | -1.6184 | -0.9380 | 3.9100 | 2.9211 | 4.8016 | 0.9333 |
| within_draw_seed | 100017 | 17 | midpoint | 2 | 2.6843 | 3.1222 | 2.2464 | -0.5800 | -0.6088 | 2.4943 | 1.8885 | 2.7579 | 0.9726 |
| within_draw_seed | 100017 | 17 | midpoint | 5 | 4.1752 | 5.0700 | 3.2804 | -1.5243 | -1.0128 | 3.8736 | 2.9056 | 4.7961 | 0.9340 |
| within_draw_seed | 100017 | 29 | ce | 2 | 2.7034 | 3.1405 | 2.2664 | -0.6526 | -0.5795 | 2.5174 | 1.8966 | 2.7576 | 0.9751 |
| within_draw_seed | 100017 | 29 | ce | 5 | 4.1879 | 5.0254 | 3.3505 | -1.7010 | -1.0154 | 3.8818 | 2.9150 | 4.7372 | 0.9434 |
| within_draw_seed | 100017 | 29 | current | 2 | 2.7238 | 3.1654 | 2.2821 | -0.5000 | -0.5367 | 2.5384 | 1.9034 | 2.8010 | 0.9715 |
| within_draw_seed | 100017 | 29 | current | 5 | 4.2382 | 5.0641 | 3.4123 | -1.3610 | -0.8092 | 3.9096 | 2.9226 | 4.8531 | 0.9335 |
| within_draw_seed | 100017 | 29 | midpoint | 2 | 2.6764 | 3.1206 | 2.2322 | -0.5845 | -0.6164 | 2.4948 | 1.8804 | 2.7300 | 0.9754 |
| within_draw_seed | 100017 | 29 | midpoint | 5 | 4.0975 | 4.9936 | 3.2014 | -1.5246 | -1.1027 | 3.8199 | 2.8616 | 4.6630 | 0.9450 |
| within_draw_seed | 100017 | 43 | ce | 2 | 2.8262 | 3.1473 | 2.5052 | -0.8533 | -0.4348 | 2.6369 | 1.9852 | 2.9059 | 0.9725 |
| within_draw_seed | 100017 | 43 | ce | 5 | 4.5928 | 5.1250 | 4.0605 | -2.2184 | -0.5204 | 4.2787 | 3.1863 | 5.2457 | 0.9323 |
| within_draw_seed | 100017 | 43 | current | 2 | 2.7571 | 3.1261 | 2.3881 | -0.7982 | -0.5612 | 2.5730 | 1.9418 | 2.8141 | 0.9706 |
| within_draw_seed | 100017 | 43 | current | 5 | 4.4095 | 5.0592 | 3.7598 | -2.0802 | -0.8508 | 4.1186 | 3.0844 | 4.9602 | 0.9286 |
| within_draw_seed | 100017 | 43 | midpoint | 2 | 2.8806 | 3.1648 | 2.5964 | -0.9943 | -0.4027 | 2.6914 | 2.0248 | 2.9364 | 0.9728 |
| within_draw_seed | 100017 | 43 | midpoint | 5 | 4.7680 | 5.1647 | 4.3713 | -2.5854 | -0.4081 | 4.4317 | 3.3010 | 5.3260 | 0.9306 |
| within_draw_seed | 200017 | 0 | ce | 2 | 2.7454 | 3.1387 | 2.3521 | -0.7030 | -0.5462 | 2.5537 | 1.9258 | 2.8007 | 0.9739 |
| within_draw_seed | 200017 | 0 | ce | 5 | 4.3432 | 5.0893 | 3.5971 | -1.8275 | -0.8452 | 4.0495 | 3.0236 | 4.9169 | 0.9362 |
| within_draw_seed | 200017 | 0 | current | 2 | 2.7357 | 3.1477 | 2.3238 | -0.6452 | -0.5710 | 2.5458 | 1.9206 | 2.7971 | 0.9719 |
| within_draw_seed | 200017 | 0 | current | 5 | 4.2863 | 5.0762 | 3.4964 | -1.6865 | -0.8809 | 3.9810 | 2.9829 | 4.8939 | 0.9313 |
| within_draw_seed | 200017 | 0 | midpoint | 2 | 2.7564 | 3.1480 | 2.3648 | -0.7168 | -0.5403 | 2.5638 | 1.9320 | 2.8164 | 0.9736 |
| within_draw_seed | 200017 | 0 | midpoint | 5 | 4.3475 | 5.0846 | 3.6105 | -1.8801 | -0.8511 | 4.0482 | 3.0337 | 4.9497 | 0.9355 |
| within_draw_seed | 200017 | 17 | ce | 2 | 2.6979 | 3.1372 | 2.2587 | -0.5898 | -0.6042 | 2.5110 | 1.8955 | 2.7564 | 0.9735 |
| within_draw_seed | 200017 | 17 | ce | 5 | 4.1862 | 5.0593 | 3.3132 | -1.5242 | -1.0032 | 3.9029 | 2.9219 | 4.7781 | 0.9361 |
| within_draw_seed | 200017 | 17 | current | 2 | 2.7107 | 3.1495 | 2.2719 | -0.6502 | -0.5888 | 2.5181 | 1.9093 | 2.7804 | 0.9722 |
| within_draw_seed | 200017 | 17 | current | 5 | 4.2140 | 5.0776 | 3.3504 | -1.6142 | -0.9498 | 3.8928 | 2.9295 | 4.8206 | 0.9326 |
| within_draw_seed | 200017 | 17 | midpoint | 2 | 2.7016 | 3.1460 | 2.2572 | -0.5810 | -0.5944 | 2.5085 | 1.8968 | 2.7718 | 0.9718 |
| within_draw_seed | 200017 | 17 | midpoint | 5 | 4.1636 | 5.0588 | 3.2685 | -1.5346 | -1.0113 | 3.8851 | 2.9149 | 4.7828 | 0.9318 |
| within_draw_seed | 200017 | 29 | ce | 2 | 2.6966 | 3.1295 | 2.2637 | -0.6547 | -0.5975 | 2.5048 | 1.8861 | 2.7535 | 0.9749 |
| within_draw_seed | 200017 | 29 | ce | 5 | 4.2119 | 5.0538 | 3.3700 | -1.7206 | -1.0275 | 3.9242 | 2.9331 | 4.7503 | 0.9413 |
| within_draw_seed | 200017 | 29 | current | 2 | 2.7269 | 3.1591 | 2.2946 | -0.4901 | -0.5541 | 2.5342 | 1.9011 | 2.7925 | 0.9724 |
| within_draw_seed | 200017 | 29 | current | 5 | 4.2347 | 5.0834 | 3.3860 | -1.3525 | -0.8367 | 3.9290 | 2.9333 | 4.8709 | 0.9331 |
| within_draw_seed | 200017 | 29 | midpoint | 2 | 2.6733 | 3.1316 | 2.2149 | -0.5768 | -0.6303 | 2.4737 | 1.8712 | 2.7282 | 0.9761 |
| within_draw_seed | 200017 | 29 | midpoint | 5 | 4.1134 | 5.0198 | 3.2070 | -1.5456 | -1.1352 | 3.8064 | 2.8726 | 4.6932 | 0.9442 |
| within_draw_seed | 200017 | 43 | ce | 2 | 2.8417 | 3.1494 | 2.5341 | -0.8644 | -0.4368 | 2.6454 | 1.9959 | 2.8922 | 0.9734 |
| within_draw_seed | 200017 | 43 | ce | 5 | 4.6315 | 5.1549 | 4.1081 | -2.2376 | -0.5049 | 4.3213 | 3.2157 | 5.2223 | 0.9311 |
| within_draw_seed | 200017 | 43 | current | 2 | 2.7696 | 3.1345 | 2.4048 | -0.7954 | -0.5699 | 2.5850 | 1.9516 | 2.8184 | 0.9712 |
| within_draw_seed | 200017 | 43 | current | 5 | 4.4102 | 5.0677 | 3.7528 | -2.0928 | -0.8561 | 4.1211 | 3.0858 | 4.9901 | 0.9281 |
| within_draw_seed | 200017 | 43 | midpoint | 2 | 2.8944 | 3.1665 | 2.6223 | -0.9926 | -0.3962 | 2.7091 | 2.0282 | 2.9492 | 0.9728 |
| within_draw_seed | 200017 | 43 | midpoint | 5 | 4.7656 | 5.1752 | 4.3560 | -2.5601 | -0.4069 | 4.4531 | 3.3135 | 5.3732 | 0.9306 |

## Legal-path coverage

| draw | seed | owner | horizon | inputs | known | usable | legal_paths | total_paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 17 | current | 2 | 2432 | 2389 | 2384 | 150879 | 155648 |
| 17 | 17 | current | 5 | 2432 | 2324 | 2304 | 143798 | 155648 |
| 17 | 17 | ce | 2 | 2432 | 2389 | 2385 | 151030 | 155648 |
| 17 | 17 | ce | 5 | 2432 | 2324 | 2304 | 144276 | 155648 |
| 17 | 17 | midpoint | 2 | 2432 | 2389 | 2382 | 150806 | 155648 |
| 17 | 17 | midpoint | 5 | 2432 | 2324 | 2306 | 143577 | 155648 |
| 17 | 29 | current | 2 | 2432 | 2389 | 2384 | 150788 | 155648 |
| 17 | 29 | current | 5 | 2432 | 2324 | 2303 | 143560 | 155648 |
| 17 | 29 | ce | 2 | 2432 | 2389 | 2384 | 151364 | 155648 |
| 17 | 29 | ce | 5 | 2432 | 2324 | 2305 | 145145 | 155648 |
| 17 | 29 | midpoint | 2 | 2432 | 2389 | 2384 | 151453 | 155648 |
| 17 | 29 | midpoint | 5 | 2432 | 2324 | 2305 | 145439 | 155648 |
| 17 | 43 | current | 2 | 2432 | 2389 | 2381 | 150492 | 155648 |
| 17 | 43 | current | 5 | 2432 | 2324 | 2302 | 143115 | 155648 |
| 17 | 43 | ce | 2 | 2432 | 2389 | 2382 | 150881 | 155648 |
| 17 | 43 | ce | 5 | 2432 | 2324 | 2306 | 143497 | 155648 |
| 17 | 43 | midpoint | 2 | 2432 | 2389 | 2382 | 150897 | 155648 |
| 17 | 43 | midpoint | 5 | 2432 | 2324 | 2305 | 143372 | 155648 |
| 100017 | 17 | current | 2 | 2432 | 2389 | 2385 | 150840 | 155648 |
| 100017 | 17 | current | 5 | 2432 | 2324 | 2302 | 143802 | 155648 |
| 100017 | 17 | ce | 2 | 2432 | 2389 | 2385 | 151030 | 155648 |
| 100017 | 17 | ce | 5 | 2432 | 2324 | 2303 | 144204 | 155648 |
| 100017 | 17 | midpoint | 2 | 2432 | 2389 | 2384 | 150845 | 155648 |
| 100017 | 17 | midpoint | 5 | 2432 | 2324 | 2305 | 143890 | 155648 |
| 100017 | 29 | current | 2 | 2432 | 2389 | 2384 | 150737 | 155648 |
| 100017 | 29 | current | 5 | 2432 | 2324 | 2298 | 143589 | 155648 |
| 100017 | 29 | ce | 2 | 2432 | 2389 | 2384 | 151303 | 155648 |
| 100017 | 29 | ce | 5 | 2432 | 2324 | 2303 | 145145 | 155648 |
| 100017 | 29 | midpoint | 2 | 2432 | 2389 | 2383 | 151339 | 155648 |
| 100017 | 29 | midpoint | 5 | 2432 | 2324 | 2303 | 145372 | 155648 |
| 100017 | 43 | current | 2 | 2432 | 2389 | 2381 | 150500 | 155648 |
| 100017 | 43 | current | 5 | 2432 | 2324 | 2300 | 143021 | 155648 |
| 100017 | 43 | ce | 2 | 2432 | 2389 | 2384 | 150804 | 155648 |
| 100017 | 43 | ce | 5 | 2432 | 2324 | 2301 | 143591 | 155648 |
| 100017 | 43 | midpoint | 2 | 2432 | 2389 | 2384 | 150860 | 155648 |
| 100017 | 43 | midpoint | 5 | 2432 | 2324 | 2303 | 143327 | 155648 |
| 200017 | 17 | current | 2 | 2432 | 2389 | 2386 | 150856 | 155648 |
| 200017 | 17 | current | 5 | 2432 | 2324 | 2306 | 143788 | 155648 |
| 200017 | 17 | ce | 2 | 2432 | 2389 | 2385 | 151053 | 155648 |
| 200017 | 17 | ce | 5 | 2432 | 2324 | 2301 | 144293 | 155648 |
| 200017 | 17 | midpoint | 2 | 2432 | 2389 | 2383 | 150768 | 155648 |
| 200017 | 17 | midpoint | 5 | 2432 | 2324 | 2307 | 143677 | 155648 |
| 200017 | 29 | current | 2 | 2432 | 2389 | 2386 | 150873 | 155648 |
| 200017 | 29 | current | 5 | 2432 | 2324 | 2302 | 143783 | 155648 |
| 200017 | 29 | ce | 2 | 2432 | 2389 | 2383 | 151257 | 155648 |
| 200017 | 29 | ce | 5 | 2432 | 2324 | 2302 | 145094 | 155648 |
| 200017 | 29 | midpoint | 2 | 2432 | 2389 | 2384 | 151441 | 155648 |
| 200017 | 29 | midpoint | 5 | 2432 | 2324 | 2304 | 145510 | 155648 |
| 200017 | 43 | current | 2 | 2432 | 2389 | 2383 | 150590 | 155648 |
| 200017 | 43 | current | 5 | 2432 | 2324 | 2304 | 143046 | 155648 |
| 200017 | 43 | ce | 2 | 2432 | 2389 | 2382 | 150934 | 155648 |
| 200017 | 43 | ce | 5 | 2432 | 2324 | 2306 | 143519 | 155648 |
| 200017 | 43 | midpoint | 2 | 2432 | 2389 | 2382 | 150828 | 155648 |
| 200017 | 43 | midpoint | 5 | 2432 | 2324 | 2304 | 143436 | 155648 |

