# Explainability Benchmark Report
**Random Seed**: 42

## Summary

| Metric | Value |
|--------|-------|
| Total Compounds | 60 |
| Toxic | 30 |
| Non-toxic | 30 |
| Datasets | 3 (Tox21, ClinTox, hERG_Karim) |
| Visualizations | 90 |

## Per-Dataset Statistics

### Tox21

- Toxic: 10  |  Non-toxic: 10
- Model: `tox21_gatv2_model`
- Method: `gradient_saliency`

### ClinTox

- Toxic: 10  |  Non-toxic: 10
- Model: `RandomForest_ClinTox_ECFP4`
- Method: `rf_fingerprint`

### hERG_Karim

- Toxic: 10  |  Non-toxic: 10
- Model: `RandomForest_hERG_ECFP4`
- Method: `rf_fingerprint`

## All 60 Compounds

| # | ID | Dataset | SMILES | Label | Pred | P(toxic) | Model |
|---|-----|---------|--------|-------|------|----------|-------|
| 1 | compound_001 | Tox21 | `O=C1c2ccccc2Cc2ccccc21` | Toxic | Safe | 0.321 | tox21_gatv2_model |
| 2 | compound_002 | Tox21 | `CCCCCC[C@@H](O)C/C=C\CCCCCCCC(=O)[O-]` | Toxic | Safe | 0.164 | tox21_gatv2_model |
| 3 | compound_003 | Tox21 | `O=C(O)c1ccccc1Nc1cccc(C(F)(F)F)c1` | Toxic | Safe | 0.322 | tox21_gatv2_model |
| 4 | compound_004 | Tox21 | `C=CC[C@]1(O)CC[C@H]2[C@@H]3CCC4=CCCC[C@@…` | Toxic | Safe | 0.309 | tox21_gatv2_model |
| 5 | compound_005 | Tox21 | `C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H…` | Toxic | Safe | 0.293 | tox21_gatv2_model |
| 6 | compound_006 | Tox21 | `O=C(O)Cc1sc(-c2ccccc2)nc1-c1ccc(Cl)cc1` | Toxic | Safe | 0.298 | tox21_gatv2_model |
| 7 | compound_007 | Tox21 | `O=C(c1ccccc1)c1cccc(O)c1` | Toxic | Safe | 0.402 | tox21_gatv2_model |
| 8 | compound_008 | Tox21 | `CC(C)C(C(=O)OC(C#N)c1cccc(Oc2ccccc2)c1)c…` | Toxic | Safe | 0.376 | tox21_gatv2_model |
| 9 | compound_009 | Tox21 | `Oc1cc(Cl)ccc1Cl` | Toxic | Safe | 0.478 | tox21_gatv2_model |
| 10 | compound_010 | Tox21 | `COc1ccc(C(Cl)=C(c2ccc(OC)cc2)c2ccc(OC)cc…` | Toxic | Safe | 0.473 | tox21_gatv2_model |
| 11 | compound_011 | Tox21 | `CC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3-c2cc…` | Safe | Safe | 0.171 | tox21_gatv2_model |
| 12 | compound_012 | Tox21 | `O=C1CC2(CCCC2)CC(=O)N1CCNCC1COc2ccccc2O1` | Safe | Safe | 0.134 | tox21_gatv2_model |
| 13 | compound_013 | Tox21 | `O=S(=O)([O-])c1ccc(Nc2nc(Nc3ccc(/C=C/c4c…` | Safe | Safe | 0.200 | tox21_gatv2_model |
| 14 | compound_014 | Tox21 | `O=C(CCCN1CCC(n2c(=O)[nH]c3ccccc32)CC1)c1…` | Safe | Safe | 0.149 | tox21_gatv2_model |
| 15 | compound_015 | Tox21 | `CCCCCCCCCCCCCCCC(=O)OC(C)C` | Safe | Safe | 0.120 | tox21_gatv2_model |
| 16 | compound_016 | Tox21 | `CC(C)(C#N)/N=N/C(C)(C)C#N` | Safe | Safe | 0.079 | tox21_gatv2_model |
| 17 | compound_017 | Tox21 | `CC1CS(=O)(=O)CCN1N=Cc1ccc([N+](=O)[O-])o…` | Safe | Safe | 0.125 | tox21_gatv2_model |
| 18 | compound_018 | Tox21 | `O=C1CCCC(=O)O1` | Safe | Safe | 0.077 | tox21_gatv2_model |
| 19 | compound_019 | Tox21 | `O=Cc1ccccc1` | Safe | Safe | 0.098 | tox21_gatv2_model |
| 20 | compound_020 | Tox21 | `CC(C)(C)N` | Safe | Safe | 0.126 | tox21_gatv2_model |
| 21 | compound_021 | ClinTox | `CN1CCC(CC1)CNC2=NN3C(=NC=C3C4=CC(=CC=C4)…` | Toxic | Toxic | 0.675 | RandomForest_ClinTox_ECFP4 |
| 22 | compound_022 | ClinTox | `CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(…` | Toxic | Toxic | 0.805 | RandomForest_ClinTox_ECFP4 |
| 23 | compound_023 | ClinTox | `CN1C2=C(C=C(C=C2)N(CCCl)CCCl)N=C1CCCC(=O…` | Toxic | Toxic | 0.802 | RandomForest_ClinTox_ECFP4 |
| 24 | compound_024 | ClinTox | `CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)…` | Toxic | Toxic | 0.526 | RandomForest_ClinTox_ECFP4 |
| 25 | compound_025 | ClinTox | `[Se]` | Toxic | Toxic | 0.585 | RandomForest_ClinTox_ECFP4 |
| 26 | compound_026 | ClinTox | `C1=NC2=C(N1[C@H]3[C@H]([C@@H]([C@H](O3)C…` | Toxic | Toxic | 0.652 | RandomForest_ClinTox_ECFP4 |
| 27 | compound_027 | ClinTox | `C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO` | Toxic | Toxic | 0.635 | RandomForest_ClinTox_ECFP4 |
| 28 | compound_028 | ClinTox | `C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N` | Toxic | Toxic | 0.508 | RandomForest_ClinTox_ECFP4 |
| 29 | compound_029 | ClinTox | `CCOC1=CC=C(C=C1)C2=CC(=CN2C3=CC=C(C=C3)S…` | Toxic | Toxic | 0.690 | RandomForest_ClinTox_ECFP4 |
| 30 | compound_030 | ClinTox | `C1=NC2=C(N1)C(=S)N=C(N2)N` | Toxic | Safe | 0.428 | RandomForest_ClinTox_ECFP4 |
| 31 | compound_031 | ClinTox | `CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)…` | Safe | Safe | 0.000 | RandomForest_ClinTox_ECFP4 |
| 32 | compound_032 | ClinTox | `c1cc(c(cc1[C@H](C[NH3+])O)O)O` | Safe | Safe | 0.000 | RandomForest_ClinTox_ECFP4 |
| 33 | compound_033 | ClinTox | `CC(C(=O)[O-])O` | Safe | Safe | 0.000 | RandomForest_ClinTox_ECFP4 |
| 34 | compound_034 | ClinTox | `CC(C)(C)[NH2+]CC(COc1cccc2c1CCC(=O)N2)O` | Safe | Safe | 0.088 | RandomForest_ClinTox_ECFP4 |
| 35 | compound_035 | ClinTox | `CC(C12CC3CC(C1)CC(C3)C2)[NH3+]` | Safe | Safe | 0.010 | RandomForest_ClinTox_ECFP4 |
| 36 | compound_036 | ClinTox | `C[NH+](C)CCOC(c1ccc(cc1)Cl)c2ccccn2` | Safe | Safe | 0.080 | RandomForest_ClinTox_ECFP4 |
| 37 | compound_037 | ClinTox | `c1cc(ccc1NCS(=O)[O-])S(=O)(=O)c2ccc(cc2)…` | Safe | Safe | 0.010 | RandomForest_ClinTox_ECFP4 |
| 38 | compound_038 | ClinTox | `C[NH+](C)CCN(Cc1ccc(cc1)OC)c2ccccn2` | Safe | Safe | 0.015 | RandomForest_ClinTox_ECFP4 |
| 39 | compound_039 | ClinTox | `C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2OC(…` | Safe | Safe | 0.020 | RandomForest_ClinTox_ECFP4 |
| 40 | compound_040 | ClinTox | `c1nc2c(n1COCCO)[nH]c(nc2=O)N` | Safe | Safe | 0.111 | RandomForest_ClinTox_ECFP4 |
| 41 | compound_041 | hERG_Karim | `N#Cc1ccc(Cn2cncc2C[NH2+][C@H]2CCN(c3cccc…` | Toxic | Toxic | 0.965 | RandomForest_hERG_ECFP4 |
| 42 | compound_042 | hERG_Karim | `CC(C)[NH2+]C[C@@H](O)COc1cccc2ccccc12` | Toxic | Toxic | 0.930 | RandomForest_hERG_ECFP4 |
| 43 | compound_043 | hERG_Karim | `CN(C(=O)N1CC(c2cc(F)ccc2F)=C[C@H]1c1cccc…` | Toxic | Toxic | 0.990 | RandomForest_hERG_ECFP4 |
| 44 | compound_044 | hERG_Karim | `Fc1ccc(Cn2c(NC3CC[NH2+]CC3)nc3ccccc32)cc…` | Toxic | Toxic | 0.940 | RandomForest_hERG_ECFP4 |
| 45 | compound_045 | hERG_Karim | `F[C@@H]1CC[NH2+]C[C@H]1c1c(-c2ccccc2)[nH…` | Toxic | Toxic | 0.940 | RandomForest_hERG_ECFP4 |
| 46 | compound_046 | hERG_Karim | `CCCCNCCC(O)c1cc2c(Cl)cc(Cl)cc2c2cc(C(F)(…` | Toxic | Toxic | 0.750 | RandomForest_hERG_ECFP4 |
| 47 | compound_047 | hERG_Karim | `CCCN1CCCC[C@@H]1C(=O)Nc1c(C)cccc1C` | Toxic | Toxic | 0.915 | RandomForest_hERG_ECFP4 |
| 48 | compound_048 | hERG_Karim | `CC(C)Cn1c(=O)n(C)c(=O)c2[nH]cnc21` | Toxic | Toxic | 0.782 | RandomForest_hERG_ECFP4 |
| 49 | compound_049 | hERG_Karim | `COc1ccc(CCN(C)CCC[C@@](C#N)(c2cc(OC)c(OC…` | Toxic | Toxic | 0.923 | RandomForest_hERG_ECFP4 |
| 50 | compound_050 | hERG_Karim | `O=C(NC1CCN(Cc2ccc3c(c2)OCO3)CC1)c1cc(=O)…` | Toxic | Toxic | 0.950 | RandomForest_hERG_ECFP4 |
| 51 | compound_051 | hERG_Karim | `CCN(C)C(=O)Oc1cccc([C@H](C)[NH+](C)C)c1` | Safe | Safe | 0.245 | RandomForest_hERG_ECFP4 |
| 52 | compound_052 | hERG_Karim | `Nc1ccncc1` | Safe | Safe | 0.200 | RandomForest_hERG_ECFP4 |
| 53 | compound_053 | hERG_Karim | `CC[C@H](O)c1cn(-c2ccc(F)cc2)c2ccc(Cl)cc1…` | Safe | Toxic | 0.755 | RandomForest_hERG_ECFP4 |
| 54 | compound_054 | hERG_Karim | `CC(=O)Nc1nnc(S(N)(=O)=O)s1` | Safe | Safe | 0.060 | RandomForest_hERG_ECFP4 |
| 55 | compound_055 | hERG_Karim | `C[NH+]1CCC[C@@H]1c1cccnc1` | Safe | Safe | 0.270 | RandomForest_hERG_ECFP4 |
| 56 | compound_056 | hERG_Karim | `COCCc1ccc(OC[C@H](O)C[NH2+]C(C)C)cc1` | Safe | Safe | 0.200 | RandomForest_hERG_ECFP4 |
| 57 | compound_057 | hERG_Karim | `CN(C)CC(c1ccc(O)cc1)C1(O)CCCCC1` | Safe | Toxic | 0.556 | RandomForest_hERG_ECFP4 |
| 58 | compound_058 | hERG_Karim | `COc1ccc([C@@H]2CC(=O)c3c(O)cc(O)cc3O2)cc…` | Safe | Safe | 0.280 | RandomForest_hERG_ECFP4 |
| 59 | compound_059 | hERG_Karim | `COc1ccc(CC(=O)Nc2cc3c(cc2[N+](=O)[O-])OC…` | Safe | Toxic | 0.615 | RandomForest_hERG_ECFP4 |
| 60 | compound_060 | hERG_Karim | `CCCCC1=NC2(CCCC2)C(=O)N1Cc1ccc(-c2ccccc2…` | Safe | Safe | 0.290 | RandomForest_hERG_ECFP4 |

## Benchmark: 30 Toxic Compounds with Visualizations

| # | ID | Dataset | SMILES | P(toxic) | Atom Heatmap | Bond Heatmap | Bar Chart |
|---|-----|---------|--------|----------|--------------|--------------|-----------|
| 1 | compound_001 | Tox21 | `O=C1c2ccccc2Cc2ccccc21` | 0.321 | ![atom](compound_001/atom_heatmap.png) | ![bond](compound_001/bond_heatmap.png) | ![bar](compound_001/atom_score_bar_chart.png) |
| 2 | compound_002 | Tox21 | `CCCCCC[C@@H](O)C/C=C\CCCCCCCC(…` | 0.164 | ![atom](compound_002/atom_heatmap.png) | ![bond](compound_002/bond_heatmap.png) | ![bar](compound_002/atom_score_bar_chart.png) |
| 3 | compound_003 | Tox21 | `O=C(O)c1ccccc1Nc1cccc(C(F)(F)F…` | 0.322 | ![atom](compound_003/atom_heatmap.png) | ![bond](compound_003/bond_heatmap.png) | ![bar](compound_003/atom_score_bar_chart.png) |
| 4 | compound_004 | Tox21 | `C=CC[C@]1(O)CC[C@H]2[C@@H]3CCC…` | 0.309 | ![atom](compound_004/atom_heatmap.png) | ![bond](compound_004/bond_heatmap.png) | ![bar](compound_004/atom_score_bar_chart.png) |
| 5 | compound_005 | Tox21 | `C[C@]12CC[C@@H]3c4ccc(O)cc4CC[…` | 0.293 | ![atom](compound_005/atom_heatmap.png) | ![bond](compound_005/bond_heatmap.png) | ![bar](compound_005/atom_score_bar_chart.png) |
| 6 | compound_006 | Tox21 | `O=C(O)Cc1sc(-c2ccccc2)nc1-c1cc…` | 0.298 | ![atom](compound_006/atom_heatmap.png) | ![bond](compound_006/bond_heatmap.png) | ![bar](compound_006/atom_score_bar_chart.png) |
| 7 | compound_007 | Tox21 | `O=C(c1ccccc1)c1cccc(O)c1` | 0.402 | ![atom](compound_007/atom_heatmap.png) | ![bond](compound_007/bond_heatmap.png) | ![bar](compound_007/atom_score_bar_chart.png) |
| 8 | compound_008 | Tox21 | `CC(C)C(C(=O)OC(C#N)c1cccc(Oc2c…` | 0.376 | ![atom](compound_008/atom_heatmap.png) | ![bond](compound_008/bond_heatmap.png) | ![bar](compound_008/atom_score_bar_chart.png) |
| 9 | compound_009 | Tox21 | `Oc1cc(Cl)ccc1Cl` | 0.478 | ![atom](compound_009/atom_heatmap.png) | ![bond](compound_009/bond_heatmap.png) | ![bar](compound_009/atom_score_bar_chart.png) |
| 10 | compound_010 | Tox21 | `COc1ccc(C(Cl)=C(c2ccc(OC)cc2)c…` | 0.473 | ![atom](compound_010/atom_heatmap.png) | ![bond](compound_010/bond_heatmap.png) | ![bar](compound_010/atom_score_bar_chart.png) |
| 11 | compound_021 | ClinTox | `CN1CCC(CC1)CNC2=NN3C(=NC=C3C4=…` | 0.675 | ![atom](compound_021/atom_heatmap.png) | ![bond](compound_021/bond_heatmap.png) | ![bar](compound_021/atom_score_bar_chart.png) |
| 12 | compound_022 | ClinTox | `CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)…` | 0.805 | ![atom](compound_022/atom_heatmap.png) | ![bond](compound_022/bond_heatmap.png) | ![bar](compound_022/atom_score_bar_chart.png) |
| 13 | compound_023 | ClinTox | `CN1C2=C(C=C(C=C2)N(CCCl)CCCl)N…` | 0.802 | ![atom](compound_023/atom_heatmap.png) | ![bond](compound_023/bond_heatmap.png) | ![bar](compound_023/atom_score_bar_chart.png) |
| 14 | compound_024 | ClinTox | `CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C…` | 0.526 | ![atom](compound_024/atom_heatmap.png) | ![bond](compound_024/bond_heatmap.png) | ![bar](compound_024/atom_score_bar_chart.png) |
| 15 | compound_025 | ClinTox | `[Se]` | 0.585 | ![atom](compound_025/atom_heatmap.png) | ![bond](compound_025/bond_heatmap.png) | ![bar](compound_025/atom_score_bar_chart.png) |
| 16 | compound_026 | ClinTox | `C1=NC2=C(N1[C@H]3[C@H]([C@@H](…` | 0.652 | ![atom](compound_026/atom_heatmap.png) | ![bond](compound_026/bond_heatmap.png) | ![bar](compound_026/atom_score_bar_chart.png) |
| 17 | compound_027 | ClinTox | `C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)…` | 0.635 | ![atom](compound_027/atom_heatmap.png) | ![bond](compound_027/bond_heatmap.png) | ![bar](compound_027/atom_score_bar_chart.png) |
| 18 | compound_028 | ClinTox | `C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C…` | 0.508 | ![atom](compound_028/atom_heatmap.png) | ![bond](compound_028/bond_heatmap.png) | ![bar](compound_028/atom_score_bar_chart.png) |
| 19 | compound_029 | ClinTox | `CCOC1=CC=C(C=C1)C2=CC(=CN2C3=C…` | 0.690 | ![atom](compound_029/atom_heatmap.png) | ![bond](compound_029/bond_heatmap.png) | ![bar](compound_029/atom_score_bar_chart.png) |
| 20 | compound_030 | ClinTox | `C1=NC2=C(N1)C(=S)N=C(N2)N` | 0.428 | ![atom](compound_030/atom_heatmap.png) | ![bond](compound_030/bond_heatmap.png) | ![bar](compound_030/atom_score_bar_chart.png) |
| 21 | compound_041 | hERG_Karim | `N#Cc1ccc(Cn2cncc2C[NH2+][C@H]2…` | 0.965 | ![atom](compound_041/atom_heatmap.png) | ![bond](compound_041/bond_heatmap.png) | ![bar](compound_041/atom_score_bar_chart.png) |
| 22 | compound_042 | hERG_Karim | `CC(C)[NH2+]C[C@@H](O)COc1cccc2…` | 0.930 | ![atom](compound_042/atom_heatmap.png) | ![bond](compound_042/bond_heatmap.png) | ![bar](compound_042/atom_score_bar_chart.png) |
| 23 | compound_043 | hERG_Karim | `CN(C(=O)N1CC(c2cc(F)ccc2F)=C[C…` | 0.990 | ![atom](compound_043/atom_heatmap.png) | ![bond](compound_043/bond_heatmap.png) | ![bar](compound_043/atom_score_bar_chart.png) |
| 24 | compound_044 | hERG_Karim | `Fc1ccc(Cn2c(NC3CC[NH2+]CC3)nc3…` | 0.940 | ![atom](compound_044/atom_heatmap.png) | ![bond](compound_044/bond_heatmap.png) | ![bar](compound_044/atom_score_bar_chart.png) |
| 25 | compound_045 | hERG_Karim | `F[C@@H]1CC[NH2+]C[C@H]1c1c(-c2…` | 0.940 | ![atom](compound_045/atom_heatmap.png) | ![bond](compound_045/bond_heatmap.png) | ![bar](compound_045/atom_score_bar_chart.png) |
| 26 | compound_046 | hERG_Karim | `CCCCNCCC(O)c1cc2c(Cl)cc(Cl)cc2…` | 0.750 | ![atom](compound_046/atom_heatmap.png) | ![bond](compound_046/bond_heatmap.png) | ![bar](compound_046/atom_score_bar_chart.png) |
| 27 | compound_047 | hERG_Karim | `CCCN1CCCC[C@@H]1C(=O)Nc1c(C)cc…` | 0.915 | ![atom](compound_047/atom_heatmap.png) | ![bond](compound_047/bond_heatmap.png) | ![bar](compound_047/atom_score_bar_chart.png) |
| 28 | compound_048 | hERG_Karim | `CC(C)Cn1c(=O)n(C)c(=O)c2[nH]cn…` | 0.782 | ![atom](compound_048/atom_heatmap.png) | ![bond](compound_048/bond_heatmap.png) | ![bar](compound_048/atom_score_bar_chart.png) |
| 29 | compound_049 | hERG_Karim | `COc1ccc(CCN(C)CCC[C@@](C#N)(c2…` | 0.923 | ![atom](compound_049/atom_heatmap.png) | ![bond](compound_049/bond_heatmap.png) | ![bar](compound_049/atom_score_bar_chart.png) |
| 30 | compound_050 | hERG_Karim | `O=C(NC1CCN(Cc2ccc3c(c2)OCO3)CC…` | 0.950 | ![atom](compound_050/atom_heatmap.png) | ![bond](compound_050/bond_heatmap.png) | ![bar](compound_050/atom_score_bar_chart.png) |

## Sample Atom Scores (First 5 Toxic Compounds)

### compound_001 (Tox21)

SMILES: `O=C1c2ccccc2Cc2ccccc21`

| Atom Index | Atom | Score |
|------------|------|-------|
| 0 | O | 1.0000 |
| 1 | C | 0.6705 |
| 2 | C | 0.7120 |
| 3 | C | 0.3519 |
| 4 | C | 0.3036 |
| 5 | C | 0.3171 |
| 6 | C | 0.2863 |
| 7 | C | 0.7012 |
| 8 | C | 0.5621 |
| 9 | C | 0.7012 |
| 10 | C | 0.2863 |
| 11 | C | 0.3171 |
| 12 | C | 0.3036 |
| 13 | C | 0.3519 |
| 14 | C | 0.7120 |

### compound_002 (Tox21)

SMILES: `CCCCCC[C@@H](O)C/C=C\CCCCCCCC(=O)[O-]`

| Atom Index | Atom | Score |
|------------|------|-------|
| 0 | C | 0.0672 |
| 1 | C | 0.1188 |
| 2 | C | 0.0611 |
| 3 | C | 0.0699 |
| 4 | C | 0.0677 |
| 5 | C | 0.0994 |
| 6 | C | 0.3257 |
| 7 | O | 0.4024 |
| 8 | C | 0.4311 |
| 9 | C | 0.9446 |
| 10 | C | 1.0000 |
| 11 | C | 0.4830 |
| 12 | C | 0.1493 |
| 13 | C | 0.0900 |
| 14 | C | 0.0844 |
| 15 | C | 0.0618 |
| 16 | C | 0.1317 |
| 17 | C | 0.1913 |
| 18 | C | 0.1652 |
| 19 | O | 0.1656 |
| 20 | O | 0.2792 |

### compound_003 (Tox21)

SMILES: `O=C(O)c1ccccc1Nc1cccc(C(F)(F)F)c1`

| Atom Index | Atom | Score |
|------------|------|-------|
| 0 | O | 0.2600 |
| 1 | C | 0.3997 |
| 2 | O | 0.4977 |
| 3 | C | 0.2939 |
| 4 | C | 0.1996 |
| 5 | C | 0.1265 |
| 6 | C | 0.1114 |
| 7 | C | 0.1475 |
| 8 | C | 0.3084 |
| 9 | N | 1.0000 |
| 10 | C | 0.3462 |
| 11 | C | 0.1515 |
| 12 | C | 0.0925 |
| 13 | C | 0.0940 |
| 14 | C | 0.4517 |
| 15 | C | 0.5167 |
| 16 | F | 0.4581 |
| 17 | F | 0.4581 |
| 18 | F | 0.4581 |
| 19 | C | 0.2177 |

### compound_004 (Tox21)

SMILES: `C=CC[C@]1(O)CC[C@H]2[C@@H]3CCC4=CCCC[C@@H]4[C@H]3CC[C@@]21C`

| Atom Index | Atom | Score |
|------------|------|-------|
| 0 | C | 0.7791 |
| 1 | C | 1.0000 |
| 2 | C | 0.5179 |
| 3 | C | 0.2506 |
| 4 | O | 0.1265 |
| 5 | C | 0.0644 |
| 6 | C | 0.0621 |
| 7 | C | 0.1984 |
| 8 | C | 0.3166 |
| 9 | C | 0.1300 |
| 10 | C | 0.2617 |
| 11 | C | 0.4705 |
| 12 | C | 0.5842 |
| 13 | C | 0.2911 |
| 14 | C | 0.2361 |
| 15 | C | 0.2421 |
| 16 | C | 0.3406 |
| 17 | C | 0.2783 |
| 18 | C | 0.0574 |
| 19 | C | 0.0562 |
| 20 | C | 0.4098 |
| 21 | C | 0.1149 |

### compound_005 (Tox21)

SMILES: `C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2OC(=O)CCC1CCCC1`

| Atom Index | Atom | Score |
|------------|------|-------|
| 0 | C | 0.1595 |
| 1 | C | 1.0000 |
| 2 | C | 0.1053 |
| 3 | C | 0.1168 |
| 4 | C | 0.3462 |
| 5 | C | 0.1781 |
| 6 | C | 0.1998 |
| 7 | C | 0.0918 |
| 8 | C | 0.2035 |
| 9 | O | 0.2996 |
| 10 | C | 0.0871 |
| 11 | C | 0.2157 |
| 12 | C | 0.0824 |
| 13 | C | 0.1083 |
| 14 | C | 0.2696 |
| 15 | C | 0.4978 |
| 16 | C | 0.1286 |
| 17 | C | 0.1205 |
| 18 | C | 0.1825 |
| 19 | O | 0.3426 |
| 20 | C | 0.4328 |
| 21 | O | 0.3306 |
| 22 | C | 0.2892 |
| 23 | C | 0.2489 |
| 24 | C | 0.5822 |
| 25 | C | 0.1466 |
| 26 | C | 0.1325 |
| 27 | C | 0.1325 |
| 28 | C | 0.1466 |
