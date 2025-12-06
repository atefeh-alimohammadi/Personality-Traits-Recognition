import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import accuracy, ModelEma
import pickle
import csv

from utils.mixup import Mixup

import utils

from scipy.special import softmax
import pickle
import time

start = time.time()
import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from utils.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import pickle
print("Time taken:", time.time() - start)

import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from utils.mixup import Mixup
from timm.utils import ModelEma
import utils
import pickle
import matplotlib.pyplot as plt


def train_class_batch(model, samples, landmarks, target, criterion):
    outputs, feat = model(samples, landmarks)

    loss = criterion(outputs, target)
    
    return loss, outputs, feat

@torch.no_grad()
def update_class_quene(sdl, outputs, feat):
    sdl.update(feat, outputs)

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


def train_one_epoch(
    model: torch.nn.Module,
    sdl,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    sdl_update_freq=None
):
    model.train(True)

    sdl_loss_fn = criterion['sdl_loss_fn']
    criterion = criterion['criterion']

    if sdl_loss_fn:
        sdl_loss_fn.update_weight(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    optimizer.zero_grad()

    it = 0  

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
      
        samples, landmarks, targets = batch[0], batch[1], batch[2]

        
        samples = samples.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        
        with torch.cuda.amp.autocast():
            
            loss, output, feat = train_class_batch(model, samples, landmarks, targets, criterion)
            print("feat after train_class_batch:", feat)

        loss_value = loss.item()
        print("feat after train_class_batch:", feat)
        sdl_loss_value = -1.0
        if sdl_loss_fn and epoch > 20:
            soft_targets = sdl(feat, output) 
            sdl_loss = sdl_loss_fn(output, soft_targets)
            sdl_loss_value = sdl_loss.item()
            loss += sdl_loss


        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

       
        loss /= update_freq
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % update_freq == 0,
        )

       
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        if sdl_loss_fn and epoch % sdl_update_freq == 0 and epoch > 20:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs, feat = model(samples, landmarks)
                    print("feat after model forward pass:", feat)
                    update_class_quene(sdl, output, feat) 


        if lr_schedule_values is not None or wd_schedule_values is not None:
            if data_iter_step % update_freq == 0:  
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]
                       
        del samples, targets, loss
        torch.cuda.empty_cache()                  
        
        it += 1

        metric_logger.update(loss=loss_value)
        metric_logger.update(sdl_loss=sdl_loss_value)

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(sdl_loss=sdl_loss_value, head="sdl_loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def validation_one_epoch(data_loader, model,args, device, regression=True):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    metrics_file = os.path.join(args.output_dir, "validation_metrics_overall.csv")
    traits_file = os.path.join(args.output_dir, "validation_metrics_per_trait.csv")
    
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    model.eval()

    outputs, targets = [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0] 
        landmarks = batch[1]  
        target = batch[2]  

        videos = videos.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, _ = model(videos, landmarks)  
            loss = criterion(output, target)

        outputs.append(output.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy())

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())

    preds, labels = np.concatenate(outputs), np.concatenate(targets)
    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    ma = np.mean(1 - np.abs(preds - labels))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((labels - preds) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
    ccc = concordance_correlation_coefficient(labels, preds)

    metric_logger.meters['mse'].update(mse, n=len(preds))
    metric_logger.meters['mae'].update(mae, n=len(preds))
    metric_logger.meters['rmse'].update(rmse, n=len(preds))
    metric_logger.meters['r2'].update(r2, n=len(preds))
    metric_logger.meters['ma'].update(ma, n=len(preds))
    metric_logger.meters['ccc'].update(ccc, n=len(preds))
    
    
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MA: {ma:.4f}, ccc: {ccc:.4f}')
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, preds, alpha=0.5)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], color='red', linestyle='--', linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Validation: True vs Predicted Values")
    plt.grid()
    plt.savefig("scatter_plot_validation.png")  # ذخیره نمودار
    plt.show()

    residuals = labels - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Validation: Residual Plot")
    plt.grid()
    plt.savefig("residual_plot_validation.png")  
    plt.show()
    
    metrics_file = "validation_metrics_overall.csv"
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["MSE", "MAE", "RMSE", "R²","MA","CCC"])
        writer.writerow([mse, mae, rmse, r2, ma, ccc])

    
    ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    mse_per_trait = []
    mae_per_trait = []
    ma_per_trait = []
    ccc_per_trait = []

    for i, trait in enumerate(ocean_traits):
        mse_trait = np.mean((preds[:, i] - labels[:, i]) ** 2)
        mae_trait = np.mean(np.abs(preds[:, i] - labels[:, i]))
        accuracy_trait = np.mean(1 - np.abs(preds[:, i] - labels[:, i]))
        ccc_trait = concordance_correlation_coefficient(labels[:, i], preds[:, i])
        mse_per_trait.append(mse_trait)
        mae_per_trait.append(mae_trait)
        ma_per_trait.append(accuracy_trait)
        ccc_per_trait.append(ccc_trait)
        
        
    traits_file = "validation_metrics_per_trait.csv"
    file_exists = os.path.isfile(traits_file)

    with open(traits_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(ocean_traits + ["MSE","Mean Accuracy", "CCC"])
        writer.writerow(mse_per_trait+ma_per_trait+ccc_per_trait)    

    return {
        "metrics": {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        "outputs": preds,
        "targets": labels,
    }


@torch.no_grad()
def final_test(data_loader, model, args, device, file, regression=True, save_feature=False):
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    overall_metrics_file = os.path.join(args.output_dir, "test_metrics_overall.csv")
    traits_metrics_file = os.path.join(args.output_dir, "test_metrics_per_trait.csv")

    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    final_result = []

    saved_features = {}
    outputs, targets = [], []
   
    overall_results = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "ma":0.0, "ccc":0.0}  # متریک‌های کلی
    traits_results = []  

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]  
        landmarks = batch[1]  
        target = batch[2]  

        videos = videos.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, saved_feature = model(videos, landmarks)  # حذف model2
            loss = criterion(output, target)
            
        outputs.append(output.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy())    
        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())

    if not os.path.exists(file):
        with open(file, 'w') as f:
            pass
            
    with open(file, 'w') as f:
        f.write("Loss\n")
        for line in final_result:
            f.write(line)

    if save_feature:
        feature_file = file.replace(file[-4:], '_feature.pkl')
        pickle.dump(saved_features, open(feature_file, 'wb'))
    preds, labels = np.concatenate(outputs), np.concatenate(targets)
    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    ma = np.mean(1 - np.abs(preds - labels))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((labels - preds) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
    ccc = concordance_correlation_coefficient(labels, preds)

    
    overall_results["mse"] = mse
    overall_results["mae"] = mae
    overall_results["rmse"] = rmse
    overall_results["r2"] = r2
    overall_results["ma"] = ma
    overall_results["ccc"] = ccc


    ocean_traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    num_traits = labels.shape[1]  
    for i, trait_name in enumerate(ocean_traits):
        trait_preds = preds[:, i]
        trait_labels = labels[:, i]
        accuracy_trait = np.mean(1 - np.abs(trait_preds - trait_labels))
        ccc_trait = concordance_correlation_coefficient(trait_labels, trait_preds)
        trait_mse = np.mean((trait_preds - trait_labels) ** 2)
        trait_mae = np.mean(np.abs(trait_preds - trait_labels))
        trait_rmse = np.sqrt(trait_mse)
        trait_r2 = 1 - (np.sum((trait_labels - trait_preds) ** 2) / np.sum((trait_labels - np.mean(trait_labels)) ** 2))
        traits_results.append({
            "trait": trait_name,
            "mse": trait_mse,
            "mae": trait_mae,
            "rmse": trait_rmse,
            "r2": trait_r2,
            "ma": accuracy_trait,
            "ccc": ccc_trait
        })



    with open(overall_metrics_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mse", "mae", "rmse", "r2","ma","ccc"])
        writer.writeheader()
        writer.writerow(overall_results)

    with open(traits_metrics_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trait", "mse", "mae", "rmse", "r2", "ma", "ccc"])
        writer.writeheader()
        writer.writerows(traits_results)

    

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, ma:{ma:.4f}, ccc:{ccc:.4f}")
    
    traits_best_predictions_file = os.path.join(args.output_dir, "test_best_predictions_per_trait.csv")
    
    with open(traits_best_predictions_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trait", "true_value", "predicted_value", "error"])
    
        for i, trait_name in enumerate(ocean_traits):
            trait_preds = preds[:, i]
            trait_labels = labels[:, i]
    
            errors = np.abs(trait_preds - trait_labels)
            
           
            best_index = np.argmin(errors)
            best_true_value = trait_labels[best_index]
            best_predicted_value = trait_preds[best_index]
            best_error = errors[best_index]
    
            
            writer.writerow([trait_name, best_true_value, best_predicted_value, best_error])


    plt.figure(figsize=(8, 8))
    plt.scatter(labels, preds, alpha=0.5)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], color='red', linestyle='--', linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Test: True vs Predicted Values")
    plt.grid()
    plt.savefig("scatter_plot_test.png") 
    plt.show()

    residuals = labels - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Test: Residual Plot")
    plt.grid()
    plt.savefig("residual_plot_test.png")  
    plt.show()

    print('Loss: {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {"overall_results": overall_results, "traits_results": traits_results, "outputs": outputs, "targets": targets}



def merge(data):
    """
    Merge the results of tasks (stored in memory) and compute regression metrics.

    Args:
        data (dict): A dictionary where keys are sample names and values are dictionaries
                     containing 'features' (list of numpy arrays) and 'label' (float).

    Returns:
        tuple: MAE, MSE, RMSE, R², and prediction dictionary.
    """
    print("Processing in-memory results")

    pred_dict = {'id': [], 'label': [], 'pred': []}

    y_true = []
    y_pred = []

    
    for name, values in data.items():
        features = np.array(values["features"])  
        label = float(values["label"])  

       
        mean_features = np.mean(features, axis=0)
        pred = float(np.mean(mean_features))  

      
        pred_dict['id'].append(name)
        pred_dict['label'].append(label)
        pred_dict['pred'].append(pred)

     
        y_true.append(label)
        y_pred.append(pred)

   
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))  # Mean Absolute Error
    mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))  # R²

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return mae, mse, rmse, r2, pred_dict





def compute_video(lst):
    """
    Compute the prediction and error for a single video in a regression task.

    Args:
        lst (tuple): A tuple containing (index, video_id, data, label),
                     where data is a list of features and label is the true value.

    Returns:
        list: [pred, error, true_label], where:
              - pred: Predicted value (float)
              - error: Absolute error (float)
              - true_label: True value (float)
    """
    i, video_id, data, label = lst

    feat = [x for x in data]
    feat = np.mean(feat, axis=0)

    pred = float(np.mean(feat))

    error = abs(pred - label)

    return [pred, error, label]







