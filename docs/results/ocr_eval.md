# OCR evaluation: 37 legible Indian plate crops = 13 unique plates (48 legible crops in total)

Crops come from the split-v2 test set, cut from ground-truth boxes (detector errors excluded). Many crops are the same car in consecutive frames, so `per-plate exact` averages per unique plate first.

| config | exact match | per-plate exact | CER | read rate | precision | exact, plates >=20 px |
|---|---|---|---|---|---|---|
| legacy (original repo) | 0.000 | 0.000 | 0.914 | 0.189 | 0.000 | 0.000 (n=8) |
| new/otsu | 0.000 | 0.000 | 0.936 | 0.081 | 0.000 | 0.000 (n=8) |
| new/gray | 0.054 | 0.154 | 0.867 | 0.162 | 0.333 | 0.250 (n=8) |
| new/clahe | 0.027 | 0.077 | 0.889 | 0.135 | 0.200 | 0.125 (n=8) |
| new/adaptive | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 (n=8) |
| new/otsu, no correction | 0.000 | 0.000 | 0.765 | 0.973 | 0.000 | 0.000 (n=8) |
| fastplate cct-s-v2, raw | 0.162 | 0.325 | 0.234 | 1.000 | 0.162 | 0.375 (n=8) |
| fastplate cct-s-v2 + correction | 0.162 | 0.325 | 0.768 | 0.243 | 0.667 | 0.375 (n=8) |
| fastplate cct-s-v2, margin | 0.135 | 0.248 | 0.256 | 1.000 | 0.135 | 0.375 (n=8) |
| fastplate cct-s-v2, margin + correction | 0.135 | 0.248 | 0.798 | 0.216 | 0.625 | 0.375 (n=8) |

## Breakdown for best config: fastplate cct-s-v2, raw

| group | n | exact | CER |
|---|---|---|---|
| all_countries | 48 | 0.271 | 0.223 |
| IN, plate >=20px | 8 | 0.375 | 0.144 |
| cond=angle+blur | 1 | 0.000 | 0.444 |
| cond=blur | 17 | 0.000 | 0.353 |
| cond=day | 2 | 0.500 | 0.250 |
| cond=untagged | 28 | 0.429 | 0.135 |
| country=IN | 37 | 0.162 | 0.234 |
| country=MX | 2 | 1.000 | 0.000 |
| country=OTHER | 9 | 0.556 | 0.230 |
| height 20-40px | 8 | 0.500 | 0.164 |
| height <20px | 29 | 0.103 | 0.258 |
| height >=40px | 11 | 0.545 | 0.174 |
| lines=1 | 43 | 0.256 | 0.223 |
| lines=2 | 5 | 0.400 | 0.222 |
