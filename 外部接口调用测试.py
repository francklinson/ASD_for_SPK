from AnomalySoundDetection.DenseAE.denseae_interface import DenseAEInterface
from AnomalySoundDetection.ConvolutionalAE.cae_interface import CAEInterface


if __name__ == '__main__':
    d = DenseAEInterface()
    print(d.judge_is_normal(r"C:\data\音频素材\异音检测\dev_data\spk\test\anomaly_id_01_00000000.wav"))
    c = CAEInterface()
    print(c.judge_is_normal(r"C:\data\音频素材\异音检测\dev_data\spk\test\anomaly_id_01_00000000.wav"))
