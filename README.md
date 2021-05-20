# Machine Learning Assessment

## Dependencies

List may not be complete.

- `python3`
- `curl`
- `numpy`
- `pandas`
- `parquet`
- `sklearn`

## Generating code

- `main.ipynb` generates all source codes except for testing modules
- `tester.ipynb` generates testing modules
- `test.ipynb` runs testing modules

## Usage

```
python3 main.py -w weather.csv [-c calls.csv] [-o output.csv]
```

The Seattle 911 calls database will be automatically downloaded to `./data/calls.csv`.
