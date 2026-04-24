import os


class DefaultConfigs(object):
    # 1.string parameters
    root = '/public/home/lz_yxy_2706/fenlei/sanfenlei_fg-bqd/data/fold_{}'
    model_name = "mobilenetv2"  # 修改为正确的学生模型名称
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    save_path = './checkpoints/best_model/mobilenetv2/model_best.pth.tar'
    logs = "./logs/"
    runs = './runs/'
    submit = "./submit/"
    gpus = "1"
    validation_step = 5
    # 2.numeric parameters
    epochs = 50
    # patience = 20
    batch_size = 16
    img_height = 320
    img_weight = 320
    vis_img_height = 2592  # 可视化尺寸
    vis_img_width = 1728
    num_classes = 3
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4


config = DefaultConfigs()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus