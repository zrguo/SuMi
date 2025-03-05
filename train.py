import warnings
import torch

import models
from src.utils import accuracy
from src import sumi
from tqdm import tqdm



def initiate(args, ttaloader):
    va_model = models.CAVMAEFT(
        label_dim=args.n_class, modality_specific_depth=11
    )
    if args.pretrain_path == "None":
        warnings.warn("Note no pre-trained models are specified.")
    else:
        # TTA based on a CAV-MAE finetuned model
        mdl_weight = torch.load(args.pretrain_path)
        if not isinstance(va_model, torch.nn.DataParallel):
            va_model = torch.nn.DataParallel(va_model)
        miss, unexpected = va_model.load_state_dict(mdl_weight, strict=False)
    
    va_model = sumi.configure_model(va_model)
    params, param_names = sumi.collect_params(va_model)

    if args.tta_method == 'sumi':
        optimizer = torch.optim.Adam(
            [{"params": params, "lr": args.lr}],
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )

    if not isinstance(va_model, torch.nn.DataParallel):
        va_model = torch.nn.DataParallel(va_model)

    va_model.cuda()
    train_model(va_model, optimizer, ttaloader, args)


def train_model(model, optimizer, ttaloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tta_method == 'sumi':
        tta_model = sumi.SuMi(model, optimizer, device, args)

    tta_model.eval()

    with torch.no_grad():
        for epoch in range(1):
            data_bar = tqdm(ttaloader)
            batch_accs = []
            iters = len(data_bar)

            for i, (a_input, v_input, corrupt_type, labels) in enumerate(data_bar):
                a_input = a_input.to(device)
                v_input = v_input.to(device)
                if args.tta_method == 'sumi':
                    outputs = tta_model(
                    (a_input, v_input), i, iters
                )  # now it infers and adapts!
                else:
                    outputs = tta_model(
                        (a_input, v_input)
                    )  # now it infers and adapts!

                batch_acc = accuracy(outputs[1], labels, topk=(1,))
                batch_acc = round(batch_acc[0].item(), 2)
                batch_accs.append(batch_acc)

                data_bar.set_description(
                    f"Batch#{i}:  ACC#{batch_acc:.2f}"
                )

            epoch_acc = round(sum(batch_accs) / len(batch_accs), 2)

            print(f"Epoch{epoch}: all acc is {epoch_acc}")