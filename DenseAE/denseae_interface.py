"""
模型测试脚本
"""
import os
import sys
import numpy
import DenseAE.common as com
from  DenseAE import keras_model


class DenseAEInterface:
    def __init__(self):
        self.model = None
        self.param = com.yaml_load()
        self.load_model()
        self.anomaly_threshold = self.param["anomaly_threshold"]

    def load_model(self):
        """
        load model file
        """
        machine_type = "spk"
        # set model path
        model_file = "DenseAE/{model}/model_{machine_type}.hdf5".format(model=self.param["model_directory"],
                                                                machine_type=machine_type)
        # load model file
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)
        self.model = keras_model.load_model(model_file)

    @staticmethod
    def check_file_path(input_file_path):
        """
        检查输入的音频文件格式是否正确
        """
        if not os.path.exists(input_file_path):
            com.logger.error("file not found!!: {}".format(input_file_path))
            sys.exit(-1)
        # 确定是文件
        if not os.path.isfile(input_file_path):
            com.logger.error("not file!!: {}".format(input_file_path))
            sys.exit(-1)
        # 检查文件格式，只支持wav
        if not input_file_path.lower().endswith(".wav"):
            com.logger.error("file not wav!!: {}".format(input_file_path))
            sys.exit(-1)

    def predict(self, file_path):
        """
        predict anomaly score
        """
        try:
            data = com.file_to_vector_array(file_path,
                                            n_mels=self.param["feature"]["n_mels"],
                                            frames=self.param["feature"]["frames"],
                                            n_fft=self.param["feature"]["n_fft"],
                                            hop_length=self.param["feature"]["hop_length"],
                                            power=self.param["feature"]["power"])
            errors = numpy.mean(numpy.square(data - self.model.predict(data)), axis=1)
            anomaly_score = numpy.mean(errors)
            return anomaly_score
        except:
            com.logger.error("file broken!!: {}".format(file_path))
            return None

    def judge_is_normal(self, file_path):
        """
        判断是否为异常；
        重构损失小则认为是正常的。
        """
        self.check_file_path(file_path)
        anomaly_score = self.predict(file_path)
        if anomaly_score <= self.anomaly_threshold:
            return True, anomaly_score
        else:
            return False, anomaly_score


if __name__ == '__main__':
    d = DenseAEInterface()
    print(d.judge_is_normal(r"E:\音频素材\异音检测\dev_data\spk\test\anomaly_id_02_00000001.wav"))
