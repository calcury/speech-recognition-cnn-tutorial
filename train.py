# train.py
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 配置
# -------------------------
TARGET_SR = 8000
DURATION_SEC = 0.5
AUDIO_LEN = int(TARGET_SR * DURATION_SEC)  # 4000
WIN_SIZE = int(0.025 * TARGET_SR)  # 25 ms -> 200
HOP_SIZE = int(0.01 * TARGET_SR)   # 10 ms -> 80

DATA_DIR = "./recordings"  # 训练数据目录，所有 wav 文件放这里
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
NUM_CLASSES = 10
MODEL_IMPORT_PATH = "cnn_spec_v7.pth"
MODEL_SAVE_PATH = "cnn_spec_v7.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 工具函数：读取、重采样、多声道->单声道、归一化、补/截
# -------------------------
def read_wav_mono(path, target_sr=TARGET_SR):
    sr, audio = wavfile.read(path)  # audio: int16 or int32
    # to float32 in [-1,1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # multi-channel -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample if needed
    if sr != target_sr:
        num = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num)

    return audio.astype(np.float32)


def pad_or_cut(audio, target_len=AUDIO_LEN):
    if len(audio) >= target_len:
        return audio[:target_len]
    else:
        pad_len = target_len - len(audio)
        return np.pad(audio, (0, pad_len), mode="constant")


# -------------------------
# STFT -> 频谱（与之前保持一致：rfft，mag，log，归一化到0-1）
# 输出 shape: (freq_bins, time_frames) = (WIN_SIZE//2+1, n_frames)
# -------------------------
def wav_to_spectrogram(audio):
    # audio assumed length == AUDIO_LEN
    frames = []
    for start in range(0, AUDIO_LEN - WIN_SIZE + 1, HOP_SIZE):
        frame = audio[start:start + WIN_SIZE]
        frame = frame * np.hanning(WIN_SIZE)
        fft_vals = np.fft.rfft(frame)  # length WIN_SIZE//2 + 1
        mag = np.abs(fft_vals)
        frames.append(mag)
    spec = np.array(frames).T  # (freq_bins, time_frames)
    # log scale & normalize to [0,1]
    spec_db = np.log(spec + 1e-10)
    spec_db -= spec_db.min()
    spec_db /= (spec_db.max() + 1e-10)
    return spec_db.astype(np.float32)


# -------------------------
# Dataset
# -------------------------
class FSDDDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(".wav"):
                self.files.append(os.path.join(data_dir, fname))
        assert len(self.files) > 0, f"在 {data_dir} 未找到 wav 文件"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # 文件名第一个字符是标签（0-9）
        basename = os.path.basename(path)
        try:
            label = int(basename[0])
        except Exception:
            raise ValueError(f"文件名格式不符合要求：{basename}（首字符应为数字标签）")

        audio = read_wav_mono(path)
        audio = pad_or_cut(audio, AUDIO_LEN)
        spec = wav_to_spectrogram(audio)  # shape (freq, time)
        # convert to tensor shape (1, freq, time)
        return torch.from_numpy(spec).unsqueeze(0), label


# -------------------------
# 简单 CNN（自动计算 flatten_dim）
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),      # [B,16,50,24]
            nn.Dropout(0.4),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),      # [B,32,25,12]
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),      # [B,64,12,6]
            nn.Dropout(0.4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


# -------------------------
# 训练主函数
# -------------------------
def train():
    dataset = FSDDDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------------------------
    # 继续训练支持
    # -------------------------
    start_epoch = 1
    if os.path.exists(MODEL_IMPORT_PATH):
        print(f"检测到已有模型：{MODEL_IMPORT_PATH}，加载并继续训练 ...")
        state_dict = torch.load(MODEL_IMPORT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        # (可选) optimizer 也恢复，但需要保存完整 checkpoint 才行
        start_epoch = 1  # 保持记录简单，不存 epoch 信息
    else:
        print("未找到已有模型，将从头开始训练 ...")

    # -------------------------
    # 训练循环
    # -------------------------
    print("\n开始训练 ...")
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE, dtype=torch.long)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 预测准确数量
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total * 100

        print(f"Epoch [{epoch}/{start_epoch + EPOCHS - 1}]  "
              f"loss={running_loss:.4f}  acc={acc:.2f}%")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
