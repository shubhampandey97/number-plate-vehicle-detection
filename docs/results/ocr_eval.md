# OCR evaluation: 42 legible Indian plate crops = 17 unique plates (48 legible crops in total)

Crops come from the split-v2 test set, cut from ground-truth boxes (detector errors excluded). Many crops are the same car in consecutive frames, so `per-plate exact` averages per unique plate first.

| config | exact match | per-plate exact | CER | read rate | precision | exact, plates >=20 px |
|---|---|---|---|---|---|---|
| legacy (original repo) | 0.000 | 0.000 | 0.891 | 0.238 | 0.000 | 0.000 (n=13) |
| new/otsu | 0.000 | 0.000 | 0.926 | 0.095 | 0.000 | 0.000 (n=13) |
| new/gray | 0.048 | 0.118 | 0.864 | 0.167 | 0.286 | 0.154 (n=13) |
| new/clahe | 0.024 | 0.059 | 0.883 | 0.143 | 0.167 | 0.077 (n=13) |
| new/adaptive | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 (n=13) |
| new/otsu, no correction | 0.024 | 0.059 | 0.748 | 0.929 | 0.026 | 0.077 (n=13) |
| fastplate cct-s-v2-global-model, raw | 0.238 | 0.425 | 0.219 | 1.000 | 0.238 | 0.538 (n=13) |
| fastplate cct-s-v2-global-model + correction | 0.143 | 0.248 | 0.659 | 0.405 | 0.353 | 0.231 (n=13) |

## Breakdown for best config: fastplate cct-s-v2-global-model, raw

| group | n | exact | CER |
|---|---|---|---|
| all_countries | 48 | 0.271 | 0.223 |
| IN, plate >=20px | 13 | 0.538 | 0.133 |
| cond=angle+blur | 1 | 0.000 | 0.444 |
| cond=blur | 17 | 0.000 | 0.353 |
| cond=day | 2 | 0.500 | 0.250 |
| cond=untagged | 28 | 0.429 | 0.135 |
| country=IN | 42 | 0.238 | 0.219 |
| country=MX | 2 | 1.000 | 0.000 |
| country=OTHER | 4 | 0.250 | 0.375 |
| height 20-40px | 8 | 0.500 | 0.164 |
| height <20px | 29 | 0.103 | 0.258 |
| height >=40px | 11 | 0.545 | 0.174 |
| lines=1 | 43 | 0.256 | 0.223 |
| lines=2 | 5 | 0.400 | 0.222 |
