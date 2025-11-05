import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from SSLSOD_main.utils_downstream.config import dutrgbd, njud, nlpr, stere, sip, rgbd135, ssd, lfsd
from SSLSOD_main.utils_downstream.misc import check_mkdir
from SSLSOD_main.model.model_stage3 import RGBD_sal
import ttach as tta

torch.manual_seed(2018)
torch.cuda.set_device(0)
ckpt_path = '/home/cty/science/my_loda/SSLSOD-main'
args = {
    'snapshot': 'imagenet_based_model-50',
    'crf_refine': False,
    'save_results': True
}

# 图像转换
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# 测试数据集路径
to_test = {
    'test_ssol': '/home/user/chentaiyang/SSLSOD-main/RGBD_SOD_Datasets/Testing_dataset/test_ssol2'
}

# 测试时增强
transforms = tta.Compose(
    [
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)


def main():
    t0 = time.time()
    net = RGBD_sal().cuda()
    print(f"加载模型: {args['snapshot']}")
    net.load_state_dict(torch.load(
        os.path.join(ckpt_path, args['snapshot'] + '.pth'),
        map_location={'cuda:1': 'cuda:1'}
    ))
    net.eval()

    with torch.no_grad():
        for name, root in to_test.items():
            root1 = os.path.join(root, 'depth')
            # 检查depth目录是否存在
            if not os.path.exists(root1):
                print(f"警告: 目录 {root1} 不存在，跳过该数据集")
                continue

            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            for idx, img_name in enumerate(img_list):
                print(f"处理 {name}: {idx + 1}/{len(img_list)}")

                # 构建图像路径
                rgb_png_path = os.path.join(root, 'RGB', img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, 'RGB', img_name[0] + '.jpg')
                depth_jpg_path = os.path.join(root, 'depth', img_name[0] + '.jpg')
                depth_png_path = os.path.join(root, 'depth', img_name[0] + '.png')

                # 读取RGB图像和深度图像
                try:
                    if os.path.exists(rgb_png_path):
                        img = Image.open(rgb_png_path).convert('RGB')
                    else:
                        img = Image.open(rgb_jpg_path).convert('RGB')

                    if os.path.exists(depth_jpg_path):
                        depth = Image.open(depth_jpg_path).convert('L')
                    else:
                        depth = Image.open(depth_png_path).convert('L')
                except Exception as e:
                    print(f"读取图像失败: {e}，跳过该图像")
                    continue

                # 保存原始图像尺寸和数据
                w_, h_ = img.size
                original_img = np.array(img)  # 保留原始图像数据用于融合

                # 预处理
                img_resize = img.resize([256, 256], Image.BILINEAR)
                depth_resize = depth.resize([256, 256], Image.BILINEAR)
                img_var = Variable(img_transform(img_resize).unsqueeze(0)).cuda()
                depth_var = Variable(depth_transform(depth_resize).unsqueeze(0)).cuda()
                depth_3 = torch.cat((depth_var, depth_var, depth_var), 1)

                # 模型预测（带测试时增强）
                mask = []
                for transformer in transforms:
                    rgb_trans = transformer.augment_image(img_var)
                    d_trans = transformer.augment_image(depth_3)
                    model_output = net(rgb_trans, d_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)

                # 计算平均预测结果
                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()  # 得到0-1之间的显著性掩码

                # 将掩码调整为原始图像尺寸
                mask_resized = to_pil(prediction.data.squeeze(0).cpu())
                mask_resized = mask_resized.resize((w_, h_), Image.BILINEAR)
                mask_np = np.array(mask_resized) / 255.0  # 转换为0-1范围的数组

                # 将单通道掩码扩展为3通道，与RGB图像匹配
                mask_3ch = np.stack([mask_np, mask_np, mask_np], axis=2)

                # 融合：显著性高的区域保留原图颜色，低的区域变为黑色
                # 公式：result = original_img * mask_3ch
                fused_result = (original_img * mask_3ch).astype(np.uint8)

                # 转换为PIL图像
                result_img = Image.fromarray(fused_result)

                # 应用CRF优化（如果启用）
                if args['crf_refine']:
                    # 注意：需要确保crf_refine函数存在且正确导入
                    result_img = Image.fromarray(crf_refine(np.array(img), fused_result))

                # 保存结果
                if args['save_results']:
                    save_dir = os.path.join(ckpt_path, args['snapshot'], name)
                    check_mkdir(save_dir)
                    save_path = os.path.join(save_dir, img_name[0] + '.png')
                    result_img.save(save_path)

    print(f"处理完成，总耗时: {time.time() - t0:.2f}秒")


if __name__ == '__main__':
    main()
