## Results

A selection of training curves, validation metrics, and test-set outputs is provided in the `results/` directory to illustrate the model’s behavior and performance.

### Metrics and Training Curves
- **mse_trend_per_trait.png** – MSE progression across epochs for all five traits  
- **ma_trend_per_trait.png** – Mean Accuracy (1 − |y_pred − y_true|) trend per trait  
- **training_loss_curve.png** – overall training loss across epochs  
- **validation_loss_curve.png** – validation loss across epochs  

### Validation & Test Metrics
- **validation_metrics_overall.xls / validation_metrics_per_trait.xls** – summarized validation statistics  
- **test_metrics_overall.csv / test_metrics_per_trait.csv** – final regression metrics on the held-out test set  
- **test_best_predictions_per_trait.csv** – closest‐match predictions for each Big Five trait  

### Residual Analysis (Summarized)
The `residual_plots/` folder contains **14 representative residual scatter plots**, each corresponding to selected epochs from the 50-epoch training schedule.

Instead of storing all 50 residual plots, we provide a **curated subset** that captures the most meaningful changes during training—early, mid, and late epochs—allowing a clear visualization of how the prediction errors evolve over time.

These diagnostic plots show:
- reduction and stabilization of residual spread for most traits over training  
- higher variability in **Neuroticism**, consistent with prior findings in apparent personality analysis  

This summarized residual analysis provides an interpretable view of model convergence without overwhelming the repository with dozens of nearly redundant plots.
