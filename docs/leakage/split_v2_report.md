# Split v2: deduplicated, group-aware, seeded

Source: `data\final_merged_datasets` (1782 images: {'train': 1247, 'val': 356, 'test': 179}); seed 42; scene threshold dHash <= 4; target 0.70,0.15,0.15

- Duplicate copies removed (dHash 0, e.g. same frame as .jpg and .png): **879**
- Unique images kept: **903**, in 588 scenes
- Cross-split pairs within threshold after the split: **0** (checked)

| split | images | share | scenes | Bike | Bus | Car | Number_plate | Person | Truck | Auto |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 629 | 69.7% | 413 | 775 | 197 | 2385 | 1151 | 56 | 257 | 552 |
| val | 151 | 16.7% | 93 | 151 | 39 | 439 | 225 | 18 | 49 | 101 |
| test | 123 | 13.6% | 82 | 186 | 50 | 543 | 286 | 24 | 60 | 137 |
