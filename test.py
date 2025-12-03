# test.py
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import torch
import torch.nn.functional as F
from train import SimpleCNN, TARGET_SR, AUDIO_LEN, WIN_SIZE, HOP_SIZE, MODEL_SAVE_PATH, DEVICE

MODEL_SAVE_PATH = "cnn_spec_initialize.pth"
MODEL_SAVE_PATH = "cnn_spec_v3.pth"

# 若 train.py 在不同文件名，把 import 调整成你的 train 模块名
# 这里假设 train.py 与 test.py 同目录，且 train.py 中定义了这些常量和 SimpleCNN

EXAMPLE_DIR = "./example"  # 放测试 wav 的目录，文件名首字符为真实标签
STEP_SEC = 0.1
STEP_SAMPLES = int(STEP_SEC * TARGET_SR)  # 800 samples


def read_wav_mono(path, target_sr=TARGET_SR):
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

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


def wav_to_spectrogram(audio):
    frames = []
    for start in range(0, AUDIO_LEN - WIN_SIZE + 1, HOP_SIZE):
        frame = audio[start:start + WIN_SIZE]
        frame = frame * np.hanning(WIN_SIZE)
        fft_vals = np.fft.rfft(frame)
        mag = np.abs(fft_vals)
        frames.append(mag)
    spec = np.array(frames).T
    spec_db = np.log(spec + 1e-10)
    spec_db -= spec_db.min()
    spec_db /= (spec_db.max() + 1e-10)
    return spec_db.astype(np.float32)


def predict_on_audio(model, audio):
    # audio: 1d float32 numpy array
    # 若长度 <= 0.5s，补齐并直接预测；否则滑窗预测、取置信度最高
    if len(audio) <= AUDIO_LEN:
        seg = pad_or_cut(audio, AUDIO_LEN)
        spec = wav_to_spectrogram(seg)
        inp = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)
            conf, label = torch.max(probs, dim=1)
            return int(label.item()), float(conf.item())

    best_label = None
    best_conf = -1.0
    # 滑窗步长为 0.1s
    for start in range(0, len(audio) - AUDIO_LEN + 1, STEP_SAMPLES):
        seg = audio[start:start + AUDIO_LEN]
        spec = wav_to_spectrogram(seg)
        inp = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)
            conf, label = torch.max(probs, dim=1)
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_label = int(label.item())
    # 如果 best_label 仍然 None（应当不会），则预测中间段
    if best_label is None:
        seg = pad_or_cut(audio, AUDIO_LEN)
        spec = wav_to_spectrogram(seg)
        inp = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)
            conf, label = torch.max(probs, dim=1)
            return int(label.item()), float(conf.item())

    return best_label, best_conf


if __name__ == "__main__":
    assert os.path.exists(
        MODEL_SAVE_PATH), f"找不到模型：{MODEL_SAVE_PATH}，请先运行 train.py"
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    files = [f for f in sorted(os.listdir(EXAMPLE_DIR))
             if f.lower().endswith(".wav")]
    if not files:
        print("example 目录下未找到 wav 文件")
        exit(0)

    total = 0
    correct = 0
    print("开始逐文件识别：")
    for fname in files:
        path = os.path.join(EXAMPLE_DIR, fname)
        try:
            audio = read_wav_mono(path)
        except Exception as e:
            print(f"读取 {fname} 失败：{e}")
            continue

        try:
            true_label = int(fname[0])
        except:
            print(f"跳过文件（文件名首字符不是数字标签）：{fname}")
            continue

        pred_label, conf = predict_on_audio(model, audio)
        total += 1
        if pred_label == true_label:
            correct += 1
        print(
            f"{fname:30s} -> 预测: {pred_label:1d}  真实: {true_label:1d}  置信度: {conf:.4f}")

    if total > 0:
        acc = 100.0 * correct / total
        print("-" * 60)
        print(f"测试样本数: {total}， 准确率: {acc:.2f}%")
