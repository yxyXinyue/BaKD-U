# coding:utf-8
import os
import csv
import random
import torch
import numpy as np
import tqdm
import warnings
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn import functional as F

from sklearn import metrics
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score
)

warnings.filterwarnings("ignore")


from models.mobilenetv2 import mobilenetv2
from utils.dataloader import DatasetCFP


def set_seed(seed=888):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[*] 随机种子已固定为: {seed}")



def calculate_fmue_threshold(dataloader, model, device, num_classes=3):
    model.eval()

    u_list = []
    u_label_list = []

    with torch.no_grad():
        for img, label in tqdm.tqdm(dataloader, desc="正在校准 FMUE 阈值", leave=False):
            img = img.to(device)
            label = label.long().to(device)

            pred = model(img)

            alpha = F.softplus(pred) + 1
            S = torch.sum(alpha, dim=1, keepdim=True)

            u = (num_classes / S).cpu().numpy().squeeze()

            b = (alpha - 1) / S
            wrong = 1 - torch.eq(b.argmax(dim=-1), label).float()

            if isinstance(u, np.ndarray):
                u_list.extend(u.tolist())
            else:
                u_list.append(float(u))

            u_label_list.extend(wrong.cpu().numpy().tolist())

    precision, recall, thresh = metrics.precision_recall_curve(
        u_label_list,
        u_list
    )

    f1_scores = []
    for p, r in zip(precision, recall):
        f1_scores.append(2 * p * r / (p + r + 1e-8))

    best_idx = np.argmax(f1_scores)

    if best_idx < len(thresh):
        return thresh[best_idx]
    else:
        return thresh[-1]



def compute_raw_metrics(y_true, y_pred):
    wr = recall_score(
        y_true, y_pred,
        average='weighted',
        zero_division=0
    )

    wp = precision_score(
        y_true, y_pred,
        average='weighted',
        zero_division=0
    )

    wf1 = f1_score(
        y_true, y_pred,
        average='weighted',
        zero_division=0
    )

    kappa = cohen_kappa_score(y_true, y_pred)

    return wr, wp, wf1, kappa



if __name__ == '__main__':

    set_seed(888)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3

    all_results = []

    print("\n" + "=" * 80)
    print("                 5-Fold FMUE评估（输出阈值前结果表格）")
    print("=" * 80)

    for k in range(1, 6):

        ckpt_path = f'./checkpoints/best_model/model_best_fold_{k}.pth.tar'

        if not os.path.exists(ckpt_path):
            print(f"[!] Fold {k} 模型不存在，跳过")
            continue


        model = mobilenetv2(num_classes=num_classes).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

 
        train_loader = DataLoader(
            DatasetCFP(root=f'./data/fold_{k}', mode='train'),
            batch_size=32,
            shuffle=False
        )

        val_loader = DataLoader(
            DatasetCFP(root=f'./data/fold_{k}', mode='val'),
            batch_size=1,
            shuffle=False
        )


        pred_thresh = calculate_fmue_threshold(
            train_loader,
            model,
            device,
            num_classes
        )


        y_true = []
        y_pred = []

        with torch.no_grad():
            for img, label in tqdm.tqdm(
                val_loader,
                desc=f"Fold {k} 推理",
                leave=False
            ):
                img = img.to(device)

                pred = model(img)
                pred_label = pred.argmax(dim=1).item()

                y_true.append(label.item())
                y_pred.append(pred_label)


        wr, wp, wf1, kappa = compute_raw_metrics(y_true, y_pred)

        all_results.append([k, wr, wp, wf1, kappa])

        print(
            f"Fold {k} | Threshold={pred_thresh:.4f} | "
            f"Recall={wr:.4f} | Precision={wp:.4f} | "
            f"F1={wf1:.4f} | Kappa={kappa:.4f}"
        )


    df = pd.DataFrame(
        all_results,
        columns=[
            "Fold",
            "Weighted Recall",
            "Weighted Precision",
            "Weighted F1",
            "Cohen's Kappa"
        ]
    )

    mean_vals = df.iloc[:, 1:].mean()
    std_vals = df.iloc[:, 1:].std()

    summary_df = pd.DataFrame({
        "Metric": [
            "Weighted Recall",
            "Weighted Precision",
            "Weighted F1",
            "Cohen's Kappa"
        ],
        "Mean": mean_vals.values,
        "Std": std_vals.values
    })


    print("\n")
    print(df.to_string(index=False))

    print("\n")
    print(summary_df.to_string(index=False))


    with pd.ExcelWriter("fmue_raw_results.xlsx") as writer:
        df.to_excel(writer, sheet_name="Fold Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    df.to_csv("fmue_raw_fold_results.csv", index=False)
    summary_df.to_csv("fmue_raw_summary.csv", index=False)

    print("\n[*] 已保存结果文件：")
    print("1. fmue_raw_results.xlsx")
    print("2. fmue_raw_fold_results.csv")
    print("3. fmue_raw_summary.csv")