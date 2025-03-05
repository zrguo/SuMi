import argparse
import os

import torch

import dataloader
import train

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataset",
    type=str,
    default="ks50",
    choices=["vggsound", "ks50"],
    help="dataset name",
)
parser.add_argument(
    "--json-root",
    type=str,
    default="json_csv_files/ks50/audio/",
    help="validation data json",
)
parser.add_argument(
    "--label-csv",
    type=str,
    default="json_csv_files/class_labels_indices_ks50.csv",
    help="csv with class labels",
)
parser.add_argument("--n_class", type=int, default=50, help="number of classes")
parser.add_argument("--model", type=str, default="cav-mae-ft", help="the model used")
parser.add_argument(
    "--dataset_mean",
    type=float,
    default=-5.081,
    help="the dataset mean, used for input normalization",
)
parser.add_argument(
    "--dataset_std",
    type=float,
    default=4.4849,
    help="the dataset std, used for input normalization",
)
parser.add_argument(
    "--target_length", type=int, default=1024, help="the input length in frames"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "-b", "--batch-size", default=32, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "-w",
    "--num-workers",
    default=32,
    type=int,
    metavar="NW",
    help="# of workers for dataloading (default: 32)",
)
parser.add_argument(
    "--pretrain_path",
    type=str,
    default="pretrained/cav_mae_ks50.pth",
    help="pretrained model path",
)
parser.add_argument("--gpu", type=str, default="0", help="gpu device number")
parser.add_argument(
    "--testmode", type=str, default="multimodal", help="how to test the model"
)
parser.add_argument(
    "--tta-method",
    type=str,
    default="sumi",
    help="which TTA method to be used",
)
parser.add_argument(
    "--corruption-modality",
    type=str,
    default="video",
    choices=["video", "audio", "none"],
    help="which modality to be corrupted",
)
parser.add_argument(
    "--severity",
    default=5,
    type=int,
    help="severity of the corruption",
)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.dataset == "vggsound":
    args.n_class = 309
elif args.dataset == "ks50":
    args.n_class = 50


if args.corruption_modality == "video":
    corruption_list = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
elif args.corruption_modality == "audio":
    corruption_list = ["gaussian_noise", "traffic", "crowd", "rain", "thunder", "wind"]


im_res = 224
val_audio_conf = {
    "num_mel_bins": 128,
    "target_length": args.target_length,
    "freqm": 0,
    "timem": 0,
    "mixup": 0,
    "dataset": args.dataset,
    "mode": "eval",
    "mean": args.dataset_mean,
    "std": args.dataset_std,
    "noise": False,
    "im_res": im_res,
}


def traintta():
    for corr in corruption_list:
        tta_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(
                os.path.join(args.json_root, f'{corr}/severity_{args.severity}.json'), label_csv=args.label_csv, audio_conf=val_audio_conf
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        methods = ['sumi']
        print(f'corruption is {corr}')
        for method in methods:
            args.tta_method = method
            train.initiate(args, tta_loader)


if __name__ == '__main__':
    traintta()