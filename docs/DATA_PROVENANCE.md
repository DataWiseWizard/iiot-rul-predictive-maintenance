# Data provenance
Source: NASA PCoE, "CMAPSS Jet Engine Simulated Data"
Page: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
File: CMAPSSData.zip, 12425978 bytes, downloaded 2026-11-12
SHA-256 (computed by me; NASA publishes none):
74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f
License: none listed on data.gov. Data is simulated.

## Checked against the readme (FD001)
100 train engines, 100 test engines, 26 columns: unit, cycle, setting1-3,
sensor1-21 (readme text says "sensor 26", a typo; 6..26 is 21 sensors).
Rows: 20631 train, 13096 test, 100 RUL values.
NOT stated in readme: contents of RUL_FD001.txt and the scoring convention.
Status: to be verified in Stage 1 (PDF and an empirical check).

## Checked against the PHM08 paper (Saxena et al., 2008)
Confirmed: 21 sensors (readme "sensor 26" is a typo), HPC-only degradation,
test sets pruned before failure, one score term per engine.
Not explicit in paper: last-row convention (implied); sensor-to-column order.
