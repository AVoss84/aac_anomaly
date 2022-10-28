# Package for anomaly detection in claims time series

An ensemble of different time series anomaly detection methods is being constructed to estimate the probability of an anomaly at time $t$. Preprocessing is done via (hierarchically) aggregating the data to univariate conditional time series. If certain criteria are not met (e.g. minimum sample size), then a particular time series is further aggregated (i.e. aggregation follows a binary-tree). 

<img src= "https://openclipart.org/image/400px/svg_to_png/319677/microscope-retro.png" width="80" />


### Create conda environment with require packages installed

```bash
conda env create -f environment.yml
```

To install the package locally, execute the following steps:

```bash
cd aac_ts_anomaly
pip install -e src
```

Start streamlit application by running:

```bash                                 
bash run_app_locally.sh 
```

Build image and start container:
```bash                                 
docker-compose up -d 
```

 