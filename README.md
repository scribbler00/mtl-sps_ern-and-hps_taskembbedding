# Emergin Relation Network and Task Embedding for Multi-Task Regression Problems

Impressum can be found [here](https://www.uni-kassel.de/uni/index.php?id=372).

# Scatter Plots for Wind2016 and Solar2016 Datasets

## Wind

![Example wind scatter plot](doc/images/sc_wind_04.png)

## Solar
![Example solar scatter plot](doc/images/sc_pv_02.png)

# Time Series Plots for Wind2016 and Solar2016 Datasets

## Wind

![Example wind time series](doc/images/ts_wind_04.png)

## Solar
![Example solar time series](doc/images/ts_pv_02.png)

## Pearson Correlation

## Wind2015 Dataset

Pearson correlation coefficient between power generation of wind parks based on the trainings and validation data.

![Wind2015 correlation](doc/images/wind_correlation_cosmo.png)


## PV2015 Dataset

Pearson correlation coefficient between power generation of solar parks based on the trainings and validation data.

![PV2015 correlation](doc/images/pv_correlation_cosmo.png)

## EuropeWindFarm Dataset

Pearson correlation coefficient between power generation of wind parks based on the trainings and validation data.



![EuropeWindFarm correlation](doc/images/wind_correlation.png)

## GermanSolarFarm Dataset

Pearson correlation coefficient between power generation of solar parks based on the trainings and validation data.


![GermanSolarFarm correlation](doc/images/pv_correlation.png)

# Forecasts 

## Exemplary Combined Forecast

![Exemplary forecasts](doc/images/sample_forecast.png)

## EuropeWindFarm  Dataset

![STL wind forecast](doc/images/stl_wind.png)
![CS wind forecast](doc/images/cs_wind.png)
![ERN wind forecast](doc/images/ern_wind.png)
![MLP wind forecast](doc/images/mlp_wind.png)
![SN wind forecast](doc/images/sn_wind.png)
![LSTM wind forecast](doc/images/lstm_wind.png)



## GermanSolarFarm Dataset

![STL solar forecast](doc/images/stl_pv.png)
![CS solar forecast](doc/images/cs_pv.png)
![ERN solar forecast](doc/images/ern_pv.png)
![MLP solar forecast](doc/images/mlp_pv.png)
![SN solar forecast](doc/images/sn_pv.png)
![LSTM solar forecast](doc/images/lstm_pv.png)

# Number of Parameters

In case of the BASELINE and the LSTM we sum up the parameters of all parks from a dataset.

## Wind2015 Dataset

|    | ModelType   |   NParametes |
|---:|:------------|-------------:|
|  0 | BASELINE    |       826714 |
|  1 | CS          |       588293 |
|  2 | ERN         |     10019804 |
|  3 | HPS         |      5627216 |
|  4 | LSTM        |      2161824 |
|  5 | MLP         |        60691 |
|  6 | SN          |      1926684 |


## EuropeWindFarm Dataset

|    | ModelType   |   NParametes |
|---:|:------------|-------------:|
|  0 | BASELINE    |       826714 |
|  1 | CS          |       588293 |
|  2 | ERN         |     10019804 |
|  3 | HPS         |      5627216 |
|  4 | LSTM        |      3959696 |
|  5 | MLP         |        60691 |
|  6 | SN          |      1926684 |


## Solar2015 Dataset

|    | ModelType   |   NParametes |
|---:|:------------|-------------:|
|  0 | BASELINE    |      1058598 |
|  1 | CS          |       655223 |
|  2 | ERN         |     16532983 |
|  3 | HPS         |      9381629 |
|  4 | LSTM        |     11774928 |
|  5 | MLP         |        48114 |
|  6 | SN          |     15428431 |


## GermanSolarFarm Dataset

|    | ModelType   |   NParametes |
|---:|:------------|-------------:|
|  0 | BASELINE    |      4396464 |
|  1 | CS          |      4091247 |
|  2 | ERN         |    112158602 |
|  3 | HPS         |     62739840 |
|  4 | LSTM        |      3238944 |
|  5 | MLP         |       249438 |
|  6 | SN          |     53722448 |
