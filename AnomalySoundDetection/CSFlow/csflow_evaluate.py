import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from model import load_model, FeatureExtractor
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from copy import deepcopy

localize = True
upscale_mode = 'bilinear'
score_export_dir = join('./viz/scores/', c.modelname)
os.makedirs(score_export_dir, exist_ok=True)
map_export_dir = join('./viz/maps/', c.modelname)
os.makedirs(map_export_dir, exist_ok=True)


def compare_histogram(scores, classes, thresh=2.5, n_bins=64):
    classes = deepcopy(classes)
    scores = deepcopy(scores)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(r'$-log(p(z))$')
    plt.ylabel('Count (normalized)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(join(score_export_dir, 'score_histogram.png'), bbox_inches='tight', pad_inches=0)


def viz_roc(values, classes, class_names):
    """
   绘制接收者操作特征（ROC）曲线，并保存这些曲线的图像。ROC曲线是评估分类模型性能的重要工具，特别是在二分类或多分类问题中。
    ### 函数参数
    - `values`: 模型预测的输出值，通常是概率或分数。
    - `classes`: 真实标签，是一个数组或列表。
    - `class_names`: 类别名称的列表，用于在多分类情况下区分不同类别的ROC曲线。

    ### 实现原理
    1. **导出ROC曲线（`export_roc`函数）**:
       - 使用`roc_curve`函数计算每个类别的假阳性率（FPR）和真阳性率（TPR）。
       - 使用`auc`函数计算ROC曲线下面积（AUC），这是一个衡量模型性能的指标。
       - 使用`matplotlib`库绘制ROC曲线，并保存图像。

    2. **绘制所有类别的ROC曲线**:
       - 调用`export_roc`函数绘制所有类别的ROC曲线。

    3. **绘制每个类别的ROC曲线**:
       - 对于每个类别，过滤出该类别的样本，并调用`export_roc`函数绘制该类别的ROC曲线。

    ### 注意事项
    - `roc_curve`和`auc`函数来自`sklearn.metrics`模块，需要确保已安装`scikit-learn`库。
    - `matplotlib`库用于绘制ROC曲线，需要确保已安装。
    - `deepcopy`函数用于深拷贝`classes`数组，以避免在计算过程中修改原始数据。
    - `join`函数用于拼接文件路径，需要确保已导入`os.path`模块。
    - `score_export_dir`变量应该提前定义，表示ROC曲线图像保存的目录。
    - `c.class_name`变量应该提前定义，表示当前类别的名称，在绘制ROC曲线时使用。

    """
    def export_roc(values, classes, export_name='all'):
        # Compute ROC curve and ROC area for each class
        classes = deepcopy(classes)
        classes[classes > 0] = 1
        fpr, tpr, _ = roc_curve(classes, values)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class ' + c.class_name)
        plt.legend(loc="lower right")
        plt.axis('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(join(score_export_dir, export_name + '.png'))

    export_roc(values, classes)
    for cl in range(1, classes.max() + 1):
        filtered_indices = np.concatenate([np.where(classes == 0)[0], np.where(classes == cl)[0]])
        classes_filtered = classes[filtered_indices]
        values_filtered = values[filtered_indices]
        export_roc(values_filtered, classes_filtered, export_name=class_names[filtered_indices[-1]])


def viz_maps(maps, name, label):
    """
    可视化给定的地图（maps）并将其保存为图像文件。函数接受三个参数：`maps`（需要可视化的地图数据），`name`（保存图像的文件名前缀），以及`label`（用于决定是否保存原始图像和叠加图）。

    ### 实现原理

    1. **加载图像**：
       - 使用`PIL.Image.open`打开指定路径的图像文件，并将其转换为RGB模式。
       - 将图像转换为NumPy数组格式，以便后续处理。

    2. **调整地图大小**：
       - 使用`F.interpolate`函数将地图数据调整到与图像相同的大小，以便进行可视化。
       - `t2np`函数将调整后的地图数据转换为NumPy数组格式。

    3. **绘制地图图像**：
       - 使用`matplotlib.pyplot`库绘制地图图像，并保存为文件。
       - 如果`label`大于0，则绘制原始图像和叠加图，并保存为文件。

    """
    img_path = img_paths[c.viz_sample_count]
    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    map_to_viz = t2np(F.interpolate(maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    if label > 0:
        plt.clf()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
        plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
        plt.savefig(join(map_export_dir, name + '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return


def viz_map_array(maps, labels, n_col=8, subsample=1, max_figures=-1):
    """
    可视化一组图像及其对应的特征图（maps）
    maps: 要可视化的特征图数组。
    labels: 与每个特征图对应的标签数组。
    n_col: 每行显示的图像列数，默认为8。
    subsample: 采样间隔，即每多少个特征图进行一次可视化，默认为4。
    max_figures: 最多保存的图像数量，默认为-1，表示不限制。

    """
    plt.clf()
    fig, subplots = plt.subplots(3, n_col)
    fig_count = -1
    col_count = -1
    for i in range(len(maps)):
        if i % subsample != 0:
            continue

        if labels[i] == 0:
            continue

        col_count = (col_count + 1) % n_col
        if col_count == 0:
            if fig_count >= 0:
                plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
                plt.close()
            fig, subplots = plt.subplots(3, n_col, figsize=(44, 16))
            fig_count += 1
            if fig_count == max_figures:
                return

        anomaly_description = img_paths[i].split(os.sep)[-2]
        image = PIL.Image.open(img_paths[i]).convert('RGB')
        image = np.array(image)
        map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
            0, 0]
        subplots[1][col_count].imshow(map)
        subplots[1][col_count].axis('off')
        subplots[0][col_count].imshow(image)
        subplots[0][col_count].axis('off')
        subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
        subplots[2][col_count].imshow(image)
        subplots[2][col_count].axis('off')
        subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.5)
    for i in range(col_count, n_col):
        subplots[0][i].axis('off')
        subplots[1][i].axis('off')
        subplots[2][i].axis('off')
    if col_count > 0:
        plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
    return


def evaluate(model, test_loader):
    model.to(c.device)
    model.eval()
    if not c.pre_extracted:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(c.device)
        for param in fe.parameters():
            param.requires_grad = False

    print('\nCompute maps, loss and scores on test set:')
    anomaly_score = list()
    test_labels = list()
    c.viz_sample_count = 0
    all_maps = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            if not c.pre_extracted:
                inputs = fe(inputs)
            z = model(inputs)

            z_concat = t2np(concat_maps(z))
            nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
            anomaly_score.append(nll_score)
            test_labels.append(t2np(labels))

            if localize:
                z_grouped = list()
                likelihood_grouped = list()
                for i in range(len(z)):
                    z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
                all_maps.extend(likelihood_grouped[0])
                for i_l, l in enumerate(t2np(labels)):
                    # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
                    c.viz_sample_count += 1

    anomaly_score = np.concatenate(anomaly_score)
    test_labels = np.concatenate(test_labels)

    compare_histogram(anomaly_score, test_labels)

    class_names = [img_path.split(os.sep)[-2] for img_path in img_paths]
    viz_roc(anomaly_score, test_labels, class_names)

    test_labels = np.array([1 if l > 0 else 0 for l in test_labels])
    auc_score = roc_auc_score(test_labels, anomaly_score)
    print('AUC:', auc_score)

    if localize:
        viz_map_array(all_maps, test_labels)
    return


train_set, test_set = load_datasets(c.dataset_path, c.class_name)
img_paths = test_set.paths if c.pre_extracted else [p for p, l in test_set.samples]
_, test_loader = make_dataloaders(train_set, test_set)
mod = load_model(c.modelname)
evaluate(mod, test_loader)
