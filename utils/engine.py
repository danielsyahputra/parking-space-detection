import os
import time
import json
import shutil
import torch
import torch.nn.functional as F
from . import transforms
from timeit import default_timer as timer
import mlflow
import mlflow.pytorch as mp

def train_one_epoch(model, optimizer, data_loader, res):
    model.train()
    device = next(model.parameters()).device
    loss_list = []
    label_match_list = []

    for image_batch, rois_batch, labels_batch in data_loader:
        optimizer.zero_grad()
        for image, rois, labels in zip(image_batch, rois_batch, labels_batch):
            image = image.to(device)
            rois = rois.to(device)
            labels = labels.to(device)

            image, rois = transforms.augment(image, rois)
            image = transforms.preprocess(image, res=res)
            class_logits = model(image, rois)
            loss = F.cross_entropy(class_logits, labels)
            loss.backward()
            loss_list += [loss.tolist()]

            pred_lab = torch.argmax(class_logits, 1)
            label_match_list += (pred_lab == labels).tolist()

        optimizer.step()

    mean_loss = float(torch.mean(torch.tensor(loss_list)))
    mean_accuracy = float(torch.mean(torch.tensor(label_match_list, dtype=torch.float32)))

    return mean_loss, mean_accuracy

@torch.no_grad()
def eval_one_epoch(model, data_loader, res):
    model.eval()
    device = next(model.parameters()).device
    loss_list = []
    label_match_list = []

    for image_batch, rois_batch, labels_batch in data_loader:
        for image, rois, labels in zip(image_batch, rois_batch, labels_batch):
            image = image.to(device)
            rois = rois.to(device)
            labels = labels.to(device)
            image = transforms.preprocess(image, res=res)
            class_logits = model(image, rois)
            loss = F.cross_entropy(class_logits, labels)
            loss_list += [float(loss)]

            pred_lab = torch.argmax(class_logits, 1)
            label_match_list += (pred_lab == labels)

    mean_loss = float(torch.mean(torch.tensor(loss_list)))
    mean_accuracy = float(torch.mean(torch.tensor(label_match_list, dtype=torch.float32)))
    return mean_loss, mean_accuracy

def train_model(model, train_loader, valid_loader, test_loader, model_dir, device, experiment_name,
                optimizer=torch.optim.AdamW, lr=1e-4, epochs=10, lr_decay=50, res=None, verbose=True, params=None):
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay, gamma=0.1)
    experiment_id = mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id):
        start_time = timer()
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train_one_epoch(model, optimizer, train_loader, res)
            scheduler.step()
            valid_loss, valid_accuracy = eval_one_epoch(model, valid_loader, res)
            
            # MLFlow Log Metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy
                },
                step=epoch
            )

            if verbose:
                print(f"Epoch {epoch:3} -- Train acc: {train_accuracy:.4f} -- Valid acc: {valid_accuracy:.4f} -- {time.time() - t0:.0f} sec")
            
            if epoch == 1:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=False)

                with open(f"{model_dir}/train_log.csv", 'w', newline='\n', encoding='utf-8') as f:
                    f.write("train_loss,train_accuracy,valid_loss,valid_accuracy\n")

            with open(f"{model_dir}/train_log.csv", 'a', newline='\n', encoding='utf-8') as f:
                f.write(f'{train_loss:.4f},{train_accuracy:.4f},{valid_loss:.4f},{valid_accuracy:.4f}\n')
            
            torch.save(model.state_dict(), f"{model_dir}/weights_epoch_{epoch}.pt")
        end_time = timer()
        test_loss, test_accuracy = eval_one_epoch(model, test_loader, res)

        mp.log_model(model, "Model")
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "time": end_time - start_time
        })

        with open(f"{model_dir}/test_logs.json", "w") as f:
            json.dump({"loss": test_loss, "accuracy": test_accuracy}, f)

        del model
        mlflow.end_run()