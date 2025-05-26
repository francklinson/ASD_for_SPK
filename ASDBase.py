from DenseAE.denseae_interface import DenseAEInterface
from ConvolutionalAE.cae_interface import CAEInterface


if __name__ == '__main__':
    # d = DenseAEInterface()
    # print(d.judge_is_normal(r"E:\音频素材\异音检测\dev_data\spk\test\anomaly_id_02_00000001.wav"))
    c = CAEInterface()
    print(c.judge_is_normal(r"E:\音频素材\异音检测\dev_data\spk\test\anomaly_id_02_00000001.wav"))
