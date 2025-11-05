import os
import torch
import numpy as np
from PIL import Image
from midas.dpt_depth import DPTDepthModel
import torchvision.transforms as transforms

def dpt_transform():
    return transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

def small_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def process_folder(input_dir, output_dir, model_type="DPT_Large", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)

    # 模型路径与加载
    model_path = "/home/user/chentaiyang/SSLSOD-main/dpt_swin2_base_384.pt"
    model = DPTDepthModel(
        backbone="swin2l24_384",
        non_negative=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 选择变换
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = dpt_transform()
    else:
        transform = small_transform()

    # 获取图像列表
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"警告：输入文件夹 '{input_dir}' 中未找到图像文件")
        return

    total = len(image_files)
    for i, filename in enumerate(image_files, 1):
        try:
            input_path = os.path.join(input_dir, filename)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_depth.png")

            # 读取图像
            img = Image.open(input_path).convert("RGB")
            img_input = transform(img).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                prediction = model(img_input)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.size[::-1],  # size=(H, W)
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

            # 归一化并保存
            depth_np = prediction.cpu().numpy()
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_gray = (depth_norm * 255).astype(np.uint8)
            Image.fromarray(depth_gray).save(output_path)

            print(f"[{i}/{total}] 完成：{filename}")
        except Exception as e:
            print(f"[{i}/{total}] 失败：{filename} - {e}")

if __name__ == "__main__":
    INPUT_FOLDER = "/home/user/chentaiyang/SSLSOD-main/test"
    OUTPUT_FOLDER = "/home/user/chentaiyang/SSLSOD-main/result"
    MODEL_TYPE = "DPT_Large"  # 也可选择 DPT_Hybrid、MiDaS_small
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用设备：{DEVICE}，模型：{MODEL_TYPE}")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_TYPE, DEVICE)
    print("处理完成！")
