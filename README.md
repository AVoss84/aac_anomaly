# Package for anomaly detection in claims time series

Claims anomaly detection using an ensemble of seasonal time series methods. This outputs different markdown based PDF/HTML reports which are dynamically updated. The periodically predicted outliers in the claims portfolio will update a Postgres database which also restricts the detected anomalies of the model.

<img src= "https://openclipart.org/image/400px/svg_to_png/319677/microscope-retro.png" width="80" />


### Create conda environment with require packages installed

```bash
conda env create -f environment.yml
```

### Requirements

To install the package locally, execute the following steps:

```bash
cd aac_ts_anomaly
pip install -e src
```
