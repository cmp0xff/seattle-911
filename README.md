# Machine Learning Assessment

## Dependencies

List may not be complete.

- `python3`
- `curl`
- `numpy`
- `pandas`
- `parquet`
- `sklearn`

## Data

The Seattle 911 calls database will be automatically downloaded to `./data/calls.csv`. If this fails (which happens on Windows), run
```
bash get_calls.sh
```

## Generating code

- `main.ipynb` generates all source codes except for testing modules
- `tester.ipynb` generates testing modules
- `test.ipynb` runs testing modules

Please generate _all_ modules before running the `main` module!

On Windows, please make a directory `./tmp` manually.

## Usage

### Help

```
python3 main.py -h
```

### Back-testing

This will generate training data from Jan 2020.
```
python3 main.py -M
```

This will generate training data from 1 to 7 in Jan 2020.
```
python3 main.py -W
```

## Input

Use the same format as the original weather and calls databases.

The output includes the predicted and real total calls within in the databases.

The scores and R-squares quantify the quality of the prediction. The prediction interval shows the uncertainty. Unfortunately, the quantities are all poorly estimated.

```
python3 main.py -w weather.csv [-c calls.csv]
```

## Clean

Delete call database, all `.sh`, and `./tmp`
```
bash clean.sh
```