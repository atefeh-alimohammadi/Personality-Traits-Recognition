def calculate_metrics(predictions, targets):
    
    mse = ((predictions - targets) ** 2).mean().item()  # MSE
    mae = (abs(predictions - targets)).mean().item()   # MAE
    
    ss_total = ((targets - targets.mean()) ** 2).sum()
    ss_residual = ((targets - predictions) ** 2).sum()
    r2_score = 1 - (ss_residual / ss_total)
    
    return mse, mae, r2_score
