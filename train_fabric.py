from calendar import c
from re import A
import monai
import torch
import argparse

import torch.nn.functional as F
from tqdm import tqdm
import yaml
from models.my_smp.utils.meter import AverageValueMeter
from utils.utils import *
from utils.dist_helper import setup_distributed
import pprint
import wandb
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam, AdamW
from adamp import SGDP
from torch.utils.data import DataLoader
import models.my_smp as smp
import torch.nn as nn
from datasets.kelp import KelpDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ttach as tta
import lightning as L
from monai.inferers import SlidingWindowInferer
from models.seg_hrnet import hrnetv2
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser(
    description="SCANet: Split Coordinate Attention Network for Building Footprint Extraction"
)

parser.add_argument("--config", type=str, required=True)
parser.add_argument("--train-id-path", type=str, required=True)
parser.add_argument("--test-id-path", type=str, required=True)
parser.add_argument("--val_id_path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--exp_name", default=None, type=str)
parser.add_argument("--decoder_name", default=None, type=str)
parser.add_argument("--encoder_name", default=None, type=str)
parser.add_argument("--dataset", default=None, type=str)
parser.add_argument("--data_root", default=None, type=str)
parser.add_argument("--mixed_precision", action="store_true", default=False)
parser.add_argument("--warmup_step", default=0, type=int)
parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,default="/home/lenovo/codes/SCANet/models/hrnet_w48.yaml")

parser.add_argument("--freeze_decoder", action="store_true", default=False)

# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision('high')

def main():
    fabric = L.Fabric(
        # precision="bf16-mixed"
        precision="32-true"
        )
    fabric.launch()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_seed(42)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # rank, world_size = setup_distributed(backend="nccl", port=args.port)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    # if rank == 0:

    os.makedirs(args.save_path, exist_ok=True)

    run = wandb.init(
        project=f"SCANet_{args.dataset}",
        group=f"{args.exp_name}",
        name=f"{args.exp_name}",
        # tags=["semi-supervised", args.exp_name],
    )

    cudnn.enabled = True
    cudnn.benchmark = True

    model = None

    if args.decoder_name == "unet" or args.decoder_name is None:
        model = smp.Unet(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
        )
    elif args.decoder_name == "unet_tiny":
        model = smp.Unet(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
            decoder_channels=(128, 64, 32, 16, 8),
        )
    elif args.decoder_name == "unet_tiny_tiny":
        model = smp.Unet(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
            decoder_channels=(64, 32, 16, 8, 4),
        )
    elif args.decoder_name == "unet_tiny_tiny_tiny":
        model = smp.Unet(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
            decoder_channels=(32, 16, 8, 4, 2),
        )
    elif args.decoder_name == "upp_tiny":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
            decoder_channels=(128, 64, 32, 16, 8),
        )
    elif args.decoder_name == "upp_tiny_tiny":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
            decoder_channels=(64, 32, 16, 8, 4),
        )
    elif args.decoder_name == "upp_tiny_tiny_tiny":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            encoder_depth=5,
            decoder_channels=(32, 16, 8, 4, 2),
        )
    elif args.decoder_name == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            classes=cfg["nclass"],
            encoder_weights=None,
            in_channels=7,
            # encoder_depth=4,
            # decoder_channels=(128, 64, 32, 16)
        )
    elif args.decoder_name == "hrnet_w48":
        model = hrnetv2()
    elif args.decoder_name == "swinunetr":
        model = monai.networks.nets.SwinUNETR(
            spatial_dims=2,
            in_channels=7,
            out_channels=1,
            img_size=(cfg["crop_size"], cfg["crop_size"]),
            depths = (3, 4, 23, 3),
            num_heads = (3, 6, 12, 24),
            feature_size = 60,
        )
    elif args.decoder_name == "upernet":
        from models.upernet import UPerNet
        model = UPerNet(in_channel=7, layers=[3, 4, 6, 3], num_class=1)

    print("Using {}".format(args.decoder_name))

    # if rank == 0:
    logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    # local_rank = int(os.environ["LOCAL_RANK"])

    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank],
    #     broadcast_buffers=False,
    #     output_device=local_rank,
    #     find_unused_parameters=False,
    #     # static_graph=True
    # )
    if args.freeze_decoder:
        print("# Freeze decoder #\n" * 5)
        for param in model.decoder.parameters():
            param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), cfg["lr"], weight_decay=0.0001)
    # optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.001)
    
    sliding_window_inferer = SlidingWindowInferer(roi_size=(cfg["crop_size"], cfg["crop_size"]), sw_batch_size=4, sw_device=torch.device('cuda'), device=torch.device('cuda'))
    
    model = torch.compile(model,
                          mode='reduce-overhead',
                        #   fullgraph=True
                          )
    
    if os.path.exists(os.path.join(args.save_path, "best.pth")):
        print("Loading best model")
        pre_dict = {k: v for k, v in torch.load(os.path.join(args.save_path, "best.pth"))['model'].items() if "encoder.conv1.0.weight" not in k}
        model.load_state_dict(pre_dict, strict=False)
    
    model, optimizer = fabric.setup(model, optimizer)
    
    patience_map = {
        'kelp': 6,
    }
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.64)

    criterion_l = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE)

    data_root = cfg["data_root"] if args.data_root is None else args.data_root

    if args.dataset == "kelp":
        train_set = KelpDataset(
            dataset_root=data_root, ids_filepath=args.train_id_path, test_mode=False, size=cfg["crop_size"]
        )

        val_set = KelpDataset(
            dataset_root=data_root, ids_filepath=args.test_id_path, test_mode=True
        )
    trainloader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=12,
        drop_last=True,
        # sampler=trainsampler,
        shuffle=False,
        prefetch_factor=2,
    )
    trainloader = fabric.setup_dataloaders(trainloader)

    trainloader_mix = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=12,
        drop_last=True,
        # sampler=trainsampler_mix,
        shuffle=False,
        prefetch_factor=2,
    )
    trainloader_mix = fabric.setup_dataloaders(trainloader_mix)

    valloader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=12,
        drop_last=False,
        shuffle=False,
    )
    valloader = fabric.setup_dataloaders(valloader)
    

    loader_len = len(trainloader)

    total_iters = loader_len * cfg["epochs"]

    previous_best = 0.0

    epoch = -1

    with tqdm(total=loader_len * cfg["epochs"]) as pbar:
        for epoch in range(epoch + 1, cfg["epochs"]):
            logger.info(
                "\n\n===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.5f}\n".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best
                )
            )

            total_loss = AverageMeter()
            loader_len = len(trainloader)

            loader = zip(trainloader, trainloader_mix)

            model.train()

            for i, (img, mask) in enumerate(
                trainloader
            ):
                pbar.update(1)

                output = model(img)
                loss = criterion_l(output, mask)
                optimizer.zero_grad()
                fabric.backward(loss)

                optimizer.step()
                total_loss.update(loss.item())
                iters = epoch * loader_len + i
                if iters < args.warmup_step:
                    lr = cfg["lr"] * (iters / args.warmup_step)
                    optimizer.param_groups[0]["lr"] = lr
                    
                if iters % 100 == 0:
                    log_dict = {
                        "loss_all": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    wandb.log(log_dict)

            logger.info(
                "\nIters: {:}, Total loss: {:.3f}\n".format(i, total_loss.avg)
            )
            if (
                epoch % (patience_map[args.dataset] * 3) == 0
                # or epoch == cfg["epochs"] - 1
                or 
                epoch > cfg["epochs"] * 4 // 5
                # True
            ):
                model.eval()
                dice_class = [0] * 1
                iou_class = [0] * 1
                with torch.no_grad():
                    for i, (img, mask) in enumerate(valloader):
                        pred = sliding_window_inferer(img, model)

                        pred = (pred.sigmoid() > 0.5).long()
                        cls = 1
                        inter = ((pred == cls) * (mask == cls)).sum().item()
                        union = (pred == cls).sum().item() + (mask == cls).sum().item()
                        dice_class[cls - 1] += (
                            2.0 * inter / union if union != 0 else 1.0
                        )
                        iou_class[cls - 1] += cacl_iou(pred == cls, mask == cls)
                dice_class = [dice / len(valloader) for dice in dice_class]
                iou_class = [iou / len(valloader) for iou in iou_class]
                mean_dice = sum(dice_class) / len(dice_class)
                # mean_iou = sum(iou_class) / len(iou_class)
                mean_iou = mean_dice

                logger.info(
                    "\n***** Evaluation ***** >>>> MeanIOU: {:.4f}, MeanDice: {:.4f}\n".format(
                        mean_iou, mean_dice
                    )
                )
                wandb.log(
                    {
                        "eval/MeanDice": mean_dice,
                        "eval/MeanIOU": mean_iou,
                    }
                )
                scheduler.step(mean_iou)
                is_best = mean_iou > previous_best
                previous_best = max(mean_iou, previous_best)
                checkpoint = {
                    "model": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "previous_best": previous_best,
                    "encoder_name": args.encoder_name,
                }
                torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
                if is_best:
                    wandb.run.summary["best_score"] = mean_iou
                    logger.info(
                        f"\n\n***** Best MeanScore: {mean_iou:.4f} Saved! ***** \n\n"
                    )
                    torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))
                
            if optimizer.param_groups[0]["lr"] < 0.00005:
                break


if __name__ == "__main__":
    main()
