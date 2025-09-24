#set page(width: 21cm, height: 29.7cm, margin: 2cm)
#set text(font: "Arial", size: 11pt)
#set par(justify: true)

#align(center)[
  #text(size: 16pt, weight: "bold")[House Price Regression — Performance Report]
]

= Introduction

This report evaluates the performance of a neural network trained for house price regression.
The analysis is based on the processed dataset, the trained PyTorch Lightning model, and predictions on the held-out test set.

= Results

== Training vs Validation Loss

#figure(
  image("figures/training_vs_validation_loss.png", width: 60%),
  caption: [
    Training and validation loss curves.
    Both losses decrease and stabilize, showing that the model converges properly without overfitting.
  ]
)

== Predictions vs Actual Values

#figure(
  image("figures/pred_vs_actual.png", width: 40%),
  caption: [
    Predicted house prices compared to actual values.
    Points are well aligned with the diagonal, showing strong predictive power.
  ]
)

== Residual Analysis

#figure(
  image("figures/residuals_hist.png", width: 60%),
  caption: [
    Distribution of residuals.
    Errors are centered around zero and approximately normal, indicating a well calibrated model.
  ]
)

== Worst-case Samples

#figure(
  image("figures/absolute_error_vs_mse.png", width: 60%),
  caption: [
    The top 20 worst samples show only moderate absolute errors and very small MSE values, 
    confirming the model has no severe outliers.”
  ]
)

== Error by Target Quantiles

#figure(
  image("figures/error_by_target_quantiles.png", width: 60%),
  caption: [
    RMSE and MAE across quintiles of the target distribution.
    Errors are lowest for mid-range house prices and slightly higher at the extremes.
  ]
)

== Error vs Features

#figure(
  image("figures/abs_error_vs_top_features.png", width: 60%),
  caption: [
    Absolute error compared with key features.
    Errors remain balanced across variables, with no evidence of strong bias.
  ]
)

= Conclusion

The regression model demonstrates strong predictive performance, with low global error metrics and residuals distributed symmetrically around zero.  
Although errors increase slightly at the lowest and highest price ranges, overall generalization is robust, making the model suitable for practical use in house price prediction tasks.
