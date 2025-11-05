## PLMPI
Source codes for paper “HVS-inspired Blind Image Quality Index with Prominent Perception Learning and Multi-level Progressive Integration”
<img width="1376" height="776" alt="QQ_1762321044055" src="https://github.com/user-attachments/assets/f37026e1-e777-4dfd-a91f-3962e5bce381" />
## Installation
Install Requirements
<div style="background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 3px solid #ddd;">
  <ul style="margin: 0; padding-left: 20px;">
    <li>csv</li>
    <li>argparse</li>
    <li>random</li>
    <li>numpy</li>
    <li>scipy</li>
    <li>torch</li>
    <li>torchvision</li>
    <li>timm</li>
    <li>mamba-ssm</li>
    <li>causal-conv1d</li>
  </ul>
</div>

## Pretrain model weight
https://github.com/CQUPT-HuBo90/PLMPI/releases/download/mamba_model/vim_t_midclstok_ft_78p3acc.pth
https://github.com/CQUPT-HuBo90/PLMPI/releases/download/dpt_levit_224/dpt_levit_224.pt
https://github.com/CQUPT-HuBo90/PLMPI/releases/download/SSL_based_model-50/SSL_based_model-50.pth

## Usage
train and test DCPI-DQA
```markdown
python train_test.py

Follow the given prompts to select the parameters you need. For example:
```markdown
python train_test.py --dataset LIVEC --patch_num 50 --batch_size 64 --lr 2e-5 --epochs 6
