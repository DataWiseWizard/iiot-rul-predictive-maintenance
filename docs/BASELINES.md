# Baseline results (validation, train engines only)

scripts/run_baselines.py: 5-fold group CV, seeds 0-4, cuts raw RUL 10-150, Ridge alpha=1.0 (untuned).
Code commit: d73bba2

```
                   nasa    mae   rmse  late_fraction  mean_late_error
truth   model                                                        
rul     age    16216.07  24.84  34.94           0.46            25.59
        mean   17958.41  34.21  39.59           0.54            40.08
        ridge   1116.27  18.16  22.34           0.56            19.56
rul_raw age    21750.16  27.07  35.91           0.46            25.59
        mean   18639.02  36.44  42.23           0.54            40.08
        ridge   1412.07  20.38  24.71           0.56            19.56

std over seeds:
                   nasa   mae  rmse  late_fraction  mean_late_error
truth   model                                                      
rul     age     6640.50  1.71  2.00           0.06             2.08
        mean    7920.49  4.46  4.18           0.11             6.54
        ridge    228.47  1.81  1.64           0.06             1.83
rul_raw age    13005.69  1.74  2.28           0.06             2.08
        mean    7493.75  3.11  2.29           0.11             6.54
        ridge    284.50  2.52  2.45           0.06             1.83
```
