import os
import random
import time
def get_least_used_gpu():
    import torch
    gpus = list(range(torch.cuda.device_count()))
    free_mem = [torch.cuda.mem_get_info(i)[0] for i in gpus]
    return free_mem.index(max(free_mem))

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
import torch
import numpy as np
import warnings
import shutil
import logging
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.config import config
from utils.dataloader import *
from utils.utils import *
from utils.progress_bar import *
from models.des import densenet121
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models.mobilenetv2 import mobilenetv2


# ===============================
# 1. 提取的不确定性（EDL）核心损失函数
# ===============================
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss_edl(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    E = alpha - 1
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)


logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')





os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_checkpoint(state, epoch, save_dir, k_fold):
    ensure_dir_exists(save_dir)
    filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth.tar")
    torch.save(state, filename)


def train(k):
    fold = config.model_name

    ensure_dir_exists(os.path.join(config.weights, "model", str(fold)))
    ensure_dir_exists(os.path.join(config.best_models, str(fold)))

    teacher_model = densenet121().to(device)
    student_model = mobilenetv2(num_classes=3).to(device)

    student_pretrained_path = '/BakD-U/models/mobilenetv2_1.0-0c6065bc.pth'
    try:
        student_model.load_pretrained_weights(student_pretrained_path)
        logging.info(f"成功加载学生模型预训练权重: {student_pretrained_path}")
    except Exception as e:
        logging.error(f"加载学生模型预训练权重失败: {e}")

    teacher_best_model_path = f'BaKD-U/checkpoints-T/best_model/model_best_fold_{k}.pth.tar'
    try:
        teacher_checkpoint = torch.load(teacher_best_model_path, map_location=device)
        teacher_model.load_state_dict(teacher_checkpoint["state_dict"])
        logging.info(f"成功加载教师模型第 {k} 折的最佳权重")
    except Exception as e:
        logging.error(f"加载教师模型第 {k} 折的最佳权重失败: {e}")

    optimizer = optim.Adam(student_model.parameters(), lr=0.0005, amsgrad=True, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    path = f'BaKD-U/data/fold_{k}'
    train_dataloader = DataLoader(DatasetCFP(root=path, mode='train'), batch_size=config.batch_size, shuffle=True,
                                  pin_memory=True)

    best_acc = 0
    train_losses_list = []
    temperature = 4.0
    alpha_distill = 0.7

    for epoch in range(config.epochs):
        train_losses = AverageMeter()
        train_top1 = AverageMeter()

        teacher_model.eval()
        student_model.train()

        for iter, (input, target) in enumerate(train_dataloader):
            input = input.to(device)
            target = target.long().to(device)

            with torch.no_grad():
                teacher_output = teacher_model(input)
                teacher_probs = F.softmax(teacher_output / temperature, dim=1)

            student_output = student_model(input)


            evidence = F.softplus(student_output)

            alpha = evidence + 1

            S = torch.sum(alpha, dim=1, keepdim=True)

            b = evidence / S

            student_log_probs = F.log_softmax(
                student_output / temperature,
                dim=1
            )

            distillation_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (temperature ** 2)

            loss_edl = torch.mean(ce_loss_edl(target, alpha, 3, epoch, config.epochs, device))

            loss = alpha_distill * distillation_loss + (1 - alpha_distill) * loss_edl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            precision1, _ = accuracy(student_output, target, topk=(1, 2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1[0], input.size(0))

        scheduler.step()
        epoch_loss = train_losses.avg
        train_losses_list.append(epoch_loss)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Acc: {train_top1.avg:.2f}")

        if train_top1.avg > best_acc:
            best_acc = train_top1.avg
            save_best_dir = 'BaKD-U/checkpoints/best_model'
            ensure_dir_exists(save_best_dir)
            torch.save({
                "epoch": epoch + 1,
                "state_dict": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_best_dir, f"model_best_fold_{k}.pth.tar"))

    plt.plot(train_losses_list)
    plt.savefig(f'training_loss_curve_fold_{k}.png')
    plt.close()


if __name__ == "__main__":
    for k_fold_test in range(1, 6):
        train(k_fold_test)