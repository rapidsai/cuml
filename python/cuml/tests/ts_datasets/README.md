# Time Series datasets

This folder contains various datasets to test our time series analysis. Using datasets from the real world allows more generic testing than using a data generator.

**Disclaimer:** the data has been filtered and organized in a way that makes it suitable to test times series models. If you wish to use this data for other purposes, please take the data from its source.

## From Statistics New Zealand

**Source:** [Stats NZ](http://archive.stats.govt.nz/infoshare/) and licensed by Stats NZ for re-use under the Creative Commons Attribution 4.0 International licence.

- `alcohol.csv`: Alcohol available for consumption (millions of litres), quarterly 1994-2019.
- `cattle.csv`: Agricultural survey: counts of different types of cattle (units) per year, 2002-2018.
- `deaths_by_region.csv`: Deaths (units) in 16 regions per year, 1991-2018.
- `guest_nights_by_region.csv`: Guest nights (thousands) in 12 regions, monthly 1996-2019.
- `hourly_earnings_by_industry.csv`: Hourly earnings ($) in 14 industries, quarterly 1989-2019.
- `long_term_arrivals_by_citizenship.csv`: Long-term arrivals (units) from 8 countries per year, 2004-2018.
- `net_migrations_auckland_by_age.csv`: Net migrations in Auckland by age range (from 0 to 49) per year, 1991-2010.
- `passenger_movements.csv`: Passenger movements (thousands), quarterly 1975-2019.
- `police_recorded_crime.csv`: Recorded crimes (units) per year, 1878-2014.
- `population_estimate.csv`: Population estimates (thousands) per year, 1875-2011.

The following files are derived from the Stats NZ dataset by removing observations (to test support for missing observations) and/or adding procedural exogenous variables:
- `guest_nights_by_region_missing.csv`
- `hourly_earnings_by_industry_missing.csv`
- `population_estimate_missing.csv`
- `endog_deaths_by_region_exog.csv`
- `endog_guest_nights_by_region_missing_exog.csv`
- `endog_hourly_earnings_by_industry_missing_exog.csv`

The following files represent procedural exogenous variables linked to the series above (normalized):
- `exog_deaths_by_region_exog.csv`
- `exog_guest_nights_by_region_missing_exog.csv`
- `exog_hourly_earnings_by_industry_missing_exog.csv`
