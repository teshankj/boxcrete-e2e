# BOxCrete Data Quality and Correction Record

## C42 GWP anomaly

During validation of the collaborator-provided BOxCrete dataset, mix `C42` was identified as having inconsistent Global Warming Potential (GWP) values across repeated measurements of the same mix.

The five observations for C42 are:

| Curing time | GWP (kg CO₂e/m³) |
|---:|---:|
| 1 day | 160.0 |
| 3 days | 161.0 |
| 5 days | 161.0 |
| 14 days | 161.0 |
| 28 days | 161.0 |

The material composition, material source, and other composition-defining inputs are identical across these observations.

## Correction

The 1-day observation was corrected:

**160.0 → 161.0 kg CO₂e/m³**

The correction is based on the repeated value observed for the same C42 composition at the other four curing ages.

## Rationale

GWP in this dataset represents the embodied environmental impact associated with the concrete/mortar mix composition. It is therefore treated as a composition-dependent quantity rather than a quantity expected to change with curing age.

Because the C42 material composition is unchanged across the five observations, the isolated value of 160.0 at 1 day is considered a data-entry inconsistency.

The value 161.0 is used because it is consistently reported for the same mix at four other curing ages.

## Provenance policy

The original collaborator-provided file is preserved unchanged under:

`data/raw/boxcrete_data.csv`

The correction is recorded separately in:

`data/corrections/boxcrete_corrections.csv`

The corrected dataset is generated programmatically from the raw dataset and correction manifest. No manual modification of the raw dataset is performed.

This ensures that:

1. the original source data remain recoverable;
2. every correction is explicitly documented;
3. the correction can be reproduced automatically;
4. future corrections can be added without modifying the raw dataset;
5. model-training data can be traced back to the original collaborator data.

## Validation status

After applying the correction, C42 has a consistent GWP value of:

**161.0 kg CO₂e/m³**

for all five curing-age observations.