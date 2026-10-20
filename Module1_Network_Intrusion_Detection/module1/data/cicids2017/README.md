# CIC-IDS2017 dataset setup

OPTIVION Module 1 expects the **CIC-IDS2017** machine-learning CSV files in this directory.

Do not hard-code machine-specific absolute paths. Configure `data.directory` in
`module1/config/default.yaml` (or override it at train time) if you store the files elsewhere.

## Download

The dataset is distributed by the Canadian Institute for Cybersecurity:

- https://www.unb.ca/cic/datasets/ids-2017.html

Use the **MachineLearningCSV** (or equivalent) CSV dumps, not packet captures.

Typical files include names such as:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

Place the `.csv` files directly in this folder (or point `data.directory` at the folder that contains them).

## Label convention

Module 1 maps labels to a **binary** task:

- `BENIGN` → `0` (Benign)
- any other original label → `1` (Attack)

The original attack category string is preserved internally for later analysis. Multi-class attack classification is **not** the primary objective of this version.

## Development sample vs final evaluation

`use_development_sample: true` (default) trains on a stratified subsample so the QP solver remains tractable.

That protocol is **development/debugging**. It is **not** a full CIC-IDS2017 evaluation.

For a configured-dataset run, set:

```yaml
data:
  use_development_sample: false
  max_rows: null
```

Quadratic programs with one slack variable per training row are memory- and time-intensive. Increase hardware resources, or set a documented `max_rows` cap, rather than silently reporting a tiny sample as the full result.

## Tests

Unit tests use **synthetic** flows and do not require these CSV files.
