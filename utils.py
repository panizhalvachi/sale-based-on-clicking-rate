# import the necessary libraries
import numpy as np
import torch
from mlflow import log_metric, log_param, log_artifacts
from sklearn.metrics import classification_report, f1_score, accuracy_score
import mlflow
from datasets import build_dataloader, RecomenderDataset


# train model
def eval_model(step, model, dev_dataloader, print_res=True, device='cpu', log_result=True):
    with torch.no_grad():
        model.eval()

        loss_func = torch.nn.BCELoss()
        outputs = []
        targets = []
        for step, (x, y) in enumerate(dev_dataloader):
            # forward pass
            x, y = x.to(device), y.to(device)

            output = model(x)
            outputs.append(output)
            targets.append(y)

        targets = torch.cat(targets).bool()
        outputs = (torch.cat(outputs) > 0.5).squeeze()

        tp = torch.sum(torch.logical_and(targets, outputs))
        tn = torch.sum(torch.logical_and(~targets, ~outputs))
        fn = torch.sum(torch.logical_and(targets, ~outputs))
        fp = torch.sum(torch.logical_and(~targets, outputs))

        try:
            presision = tp / (tp + fp)
        except:
            presision = torch.Tensor(0)

        try:
            recall = tp / (tp + fn)
        except:
            recall = torch.Tensor(0)

        try:
            f1_score_co = 2 * presision * recall / (presision + recall)
            if print_res:
                print("manual f1 score=", f1_score_co.item())
        except:
            f1_score_co = torch.Tensor(0)
            if print_res:
                print("problem with manual f1 score")

        if print_res:
            r = classification_report(targets.bool(), (outputs > 0.5))
            print(r)

        if log_result:
            log_metric("f1 score", f1_score_co.item(), step)
            log_metric("accuracy", accuracy_score(targets.bool(), outputs > 0.5), step)
        return f1_score_co.item(), recall.item(), presision.item()


def calc_loss(target, output, criterion, weight):
    weight_ = weight[target.data.view(-1).long()].view_as(target)
    loss = criterion(output.squeeze(dim=1), target)
    loss_class_weighted = loss * weight_
    return loss_class_weighted.mean()


def train_model(epoch, model: torch.nn.Module, optimizer: torch.optim.Adam, train_dataloader, weight, device='cpu'):
    model.train()
    loss_func = torch.nn.BCELoss(reduction="none")
    offset = len(train_dataloader) * epoch
    for step, (x, y) in enumerate(train_dataloader):
        # forward pass
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)

        loss = calc_loss(y, output, loss_func, weight)

        log_metric("training loss", loss.item(), step + offset)
        loss.backward()
        optimizer.step()


def train_and_eval_model(epochs, lr, batch_size, device, model, class_weight, train_data, dev_data, print_res=True, run_id=None):
    train_dataloader = build_dataloader(train_data, batch_size)
    dev_dataloader = build_dataloader(dev_data, batch_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.01)

    last_f1 = 0
    with mlflow.start_run():
        log_param("lr", lr)
        log_param("epochs", epochs)
        log_param("batch size", batch_size)
        log_param("device", device)
        log_param("model", model.__class__.__name__)
        log_param("class weight", class_weight.cpu().numpy())
        log_param("hyper parameter tuning", not print_res)

        for epoch in range(epochs):
            train_model(epoch, model, optimizer, train_dataloader, class_weight)
            for g in optimizer.param_groups:
                g['lr'] *= 0.3

            if epoch % 1 == 0:
                last_f1, recall, precision = eval_model((epoch + 1) * len(train_dataloader), model, dev_dataloader,
                                                        print_res)
    return last_f1, recall, precision


def hyper_parameter_tuning(lrs, batch_sizes, device, model_classes, model_input, class_weight, train_data, dev_data):
    best_f1 = None
    best_parameter = None
    for model_class in model_classes:
        for lr in lrs:
            for batch_size in batch_sizes:
                np.random.seed(0)
                torch.manual_seed(0)

                f1, recall, precision = train_and_eval_model(1, lr, batch_size, device, model_class(*model_input),
                                                             class_weight,
                                                             train_data,
                                                             dev_data, False)
                print(
                    f"model_classes={model_class.__name__} lr={lr} batch_size={batch_size} f1={round(f1, 3)} recall={round(recall, 3)} precision={round(precision, 3)}")
                if best_f1 is None or best_f1 < f1:
                    best_f1 = f1
                    best_parameter = [lr, batch_size]
    return best_parameter
