import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from AnomalySoundDetection.ASDBase import AnomalySoundDetectionBase
from AnomalySoundDetection.DifferNet.model import DifferNet
from AnomalySoundDetection.DifferNet.utils import *


class DifferNetInterface(AnomalySoundDetectionBase):
    def __init__(self):
        super(DifferNetInterface, self).init()
        self.model_path = "AnomalySoundDetection/DifferNet/weights/32k"
        self.model = None
        self.optimizer = None
        self.load_model()

    def load_model(self):
        """

        """
        self.model = DifferNet()
        self.model.load_state_dict(torch.load(self.model_path))
        self.optimizer = torch.optim.Adam(self.model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04,
                                          weight_decay=1e-5)
        self.model.to(c.device)

    def predict(self, input_file_path):
        pass

    def judge_is_normal(self, file_path):
        test_z = list()
        eval_set = load_eval_dataset(file_path, class_name="32k")
        eval_loader = torch.utils.data.DataLoader(eval_set, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                                  drop_last=False)
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                z = self.model(inputs)
                test_z.append(z)

        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))

        return anomaly_score
