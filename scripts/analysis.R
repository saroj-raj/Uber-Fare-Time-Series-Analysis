#loading libraries
library(fpp3)
library(readxl)
library(tidyverse)
library(latex2exp)
library(readr)
library(tseries)

# Load Uber Fare Data
uber_data <- read.csv("uber.csv", stringsAsFactors = TRUE)

# converting dataset to tsibble
uber_selected <- uber_data |>
  select(pickup_datetime, fare_amount, passenger_count, X)
uber_selected

uber_daily <- uber_selected |>
  mutate(pickup_datetime = as_date(pickup_datetime)) |>
  as_tsibble(index = pickup_datetime, key = X)
uber_daily

uber_monthly <- uber_daily |>
  mutate(pickup_datetime = yearmonth(pickup_datetime)) |>
  as_tsibble(index = pickup_datetime)
uber_monthly

uber_fare_monthly <- uber_monthly |>
  select(fare_amount, pickup_datetime) |>
  summarise(total_fare = sum(fare_amount))
uber_fare_monthly

# Plotting the data
autoplot(uber_fare_monthly)

uber_fare_monthly |>
  gg_season(total_fare)

uber_fare_monthly |>
  gg_subseries(total_fare)

# Transforming data using Box-Cox
uber_boxcox_params <- uber_fare_monthly |>
  features(total_fare, features = guerrero)
pull(uber_boxcox_params)

uber_fare_monthly |>
  autoplot(box_cox(total_fare, 0.923)) +
  labs(y = "Box-Cox Transformed Total Fare")

# Splitting data into training & testing sets
uber_train <- uber_fare_monthly |>
  select(total_fare) |>
  slice(1:68)

uber_test <- uber_fare_monthly |>
  select(total_fare) |>
  slice(69:78)

# Fitting some models to the training data
uber_fit <- uber_train |>
  model(
    average = MEAN(box_cox(total_fare, uber_boxcox_params)),
    naive = NAIVE(box_cox(total_fare, uber_boxcox_params)),
    snaive = SNAIVE(box_cox(total_fare, uber_boxcox_params)),
    drift = RW(box_cox(total_fare, uber_boxcox_params) ~ drift()),
    stl_uber = decomposition_model(
      STL(box_cox(total_fare, uber_boxcox_params)),
      NAIVE(season_adjust)
    ),
    tslm = TSLM(box_cox(total_fare, uber_boxcox_params) ~ trend() + season())
  )

uber_train |>
  model(STL(box_cox(total_fare, uber_boxcox_params))) |>
  components() |>
  autoplot()

uber_train |>
  model(STL(total_fare)) |>
  components() |>
  autoplot()

uber_forecast <- uber_fit |>
  forecast(h = "1 year")

# Plotting forecasted data and model accuracy
uber_fare_monthly |>
  autoplot(total_fare) +
  geom_line(aes(y = .fitted, color = .model), data = fitted(uber_fit)) +
  labs(y = "Total Fare", title = "Simple Forecasting Methods") +
  autolayer(uber_forecast)

accuracy(uber_forecast, uber_test) |>
  arrange(RMSE)

uber_forecast

# KPSS Test
uber_fare_monthly |>
  features(total_fare, unitroot_kpss)

uber_fare_monthly |>
  features(total_fare, unitroot_nsdiffs)

# DF Test
uber_df_test <- adf.test(uber_fare_monthly$total_fare, alternative = "stationary")

head(uber_df_test$p.value)
uber_df_test

# ACF and PACF plots
uber_train |>
  gg_tsdisplay(difference(total_fare), plot_type = 'partial') +
  labs(title = "Non-Seasonal Differenced", y = "")

uber_train |>
  gg_tsdisplay(difference(total_fare, 12), plot_type = 'partial', lag = 12) +
  labs(title = "Seasonally Differenced", y = "")

# Fitting ETS and ARIMA models to training data
uber_fit_ets_arima <- uber_train |>
  model(
    # ETS Models
    ETS_auto = ETS(total_fare),
    ETS_ses = ETS(total_fare ~ error("A") + trend("A") + season("N")),
    ETS_hw_mul = ETS(total_fare ~ error("M") + trend("A") + season("M")),
    Ets_damped_add = ETS(total_fare ~ error("A") + trend("Ad") + season("A")),
    Ets_damped_mul = ETS(total_fare ~ error("A") + trend("Ad") + season("M")),
    # ETS with Box-Cox transformations
    ETS_auto_box = ETS(box_cox(total_fare, uber_boxcox_params)),
    ETS_ses_box = ETS(box_cox(total_fare, uber_boxcox_params) ~ error("A") + trend("A") + season("N")),
    ETS_hw_mul_box = ETS(box_cox(total_fare, uber_boxcox_params) ~ error("M") + trend("A") + season("M")),
    Ets_damped_add_box = ETS(box_cox(total_fare, uber_boxcox_params) ~ error("A") + trend("Ad") + season("A")),
    Ets_damped_mul_box = ETS(box_cox(total_fare, uber_boxcox_params) ~ error("A") + trend("Ad") + season("M")),
    
    # ARIMA Models
    Arima_stepwise = ARIMA(total_fare),
    Arima_search = ARIMA(total_fare, stepwise = FALSE),
    # ARIMA Trend
    Arima_311 = ARIMA(total_fare ~ pdq(3,1,1)),
    Arima_410 = ARIMA(total_fare ~ pdq(4,1,0)),
    Arima_012 = ARIMA(total_fare ~ pdq(0,1,2)),
    # Both Seasonal & Trend
    Arima012011 = ARIMA(total_fare ~ pdq(0,1,2) + PDQ(0,1,1)),
    Arima210011 = ARIMA(total_fare ~ pdq(2,1,0) + PDQ(0,1,1)),
    Arima011011 = ARIMA(total_fare ~ pdq(0,1,1) + PDQ(0,1,1)),
    Arima212011 = ARIMA(total_fare ~ pdq(2,1,2) + PDQ(0,1,1)),
    Arima210111 = ARIMA(total_fare ~ pdq(2,1,0) + PDQ(1,1,1)),
    Arima212111 = ARIMA(total_fare ~ pdq(2,1,2) + PDQ(1,1,1)),
    Arima_stepwise_box = ARIMA(box_cox(total_fare, uber_boxcox_params)),
    Arima_search_box = ARIMA(box_cox(total_fare, uber_boxcox_params), stepwise = FALSE),
    # Trend with Box-Cox transformations
    Arima_311_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(3,1,1)),
    Arima_410_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(4,1,0)),
    Arima_012_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(0,1,2)),
    # Both Seasonal & Trend with Box-Cox transformations
    Arima012011_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(0,1,2) + PDQ(0,1,1)),
    Arima210011_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(2,1,0) + PDQ(0,1,1)),
    Arima011011_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(0,1,1) + PDQ(0,1,1)),
    Arima212011_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(2,1,2) + PDQ(0,1,1)),
    Arima210111_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(2,1,0) + PDQ(1,1,1)),
    Arima212111_box = ARIMA(box_cox(total_fare, uber_boxcox_params) ~ pdq(2,1,2) + PDQ(1,1,1))
  )

# Forecasting ETS and ARIMA models
uber_fc_ets_arima <- uber_fit_ets_arima |>
  forecast(h = "1 year")

accuracy(uber_fc_ets_arima, uber_test) |>
  arrange(RMSE)

# AIC, AICc, BIC values for top 5 models
top_models <- uber_fit_ets_arima |>
  select(Arima012011_box, Arima011011_box, Arima210011_box, Arima212011_box, Arima210111_box)

# Forecasting top 5 models
refit_train_fc <- top_models |>
  forecast(h = "1 year")

uber_train |>
  autoplot(total_fare) +
  geom_line(aes(y = .fitted, color = .model), data = fitted(top_models)) +
  labs(y = "Total Fare", title = "Best Top 5 Models") +
  autolayer(refit_train_fc)

# Residual analysis for top 2 models
gg_tsresiduals(uber_fit_ets_arima |> select(Arima012011_box), lag_max = 12) + 
  labs(title = "Residuals for Arima012011 Box-Cox Transformed Model")
gg_tsresiduals(uber_fit_ets_arima |> select(Arima011011_box), lag_max = 12) + 
  labs(title = "Residuals for Arima011011 Box-Cox Transformed Model")

# Ljung-Box test for residuals
uber_fit_ets_arima |>
  select(Arima012011_box) |>
  report()

augment(uber_fit_ets_arima) |>
  filter(.model == "Arima012011_box") |>
  features(.innov, ljung_box, lag = 10, dof = 3)

# Plotting forecasted values
top_models |>
  forecast(h = "1 year") |>
  autoplot(uber_fare_monthly) +
  labs(title = "Forecasted Models", y = "Total Fare")

# Forecasted top model
forecasted_values <- top_models |>
  select(Arima012011_box) |>
  forecast(h = "1 year")

forecasted_values$.mean

