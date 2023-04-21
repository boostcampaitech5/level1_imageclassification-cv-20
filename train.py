import os
import yaml
import torch
import wandb
from tqdm import tqdm
from data_loader.data_loader import get_trainloader
import model.model as module_model
import loss.loss as module_loss
from util.util import get_train_transform, assign_pretrained_weight
from sklearn.metrics import f1_score


def main(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"{device=}")
    epochs = config["epoch"]

    wandb.login()
    wandb.init(
        project="mask_classification_project",
        name=config["run_name"],
        config=config,
        reinit=True,
    )

    train_loader = get_trainloader(
        path=config["data_path"],
        transform=get_train_transform(),
        batch_size=config["batch_size"],
        shuffle=True,
        weighted_sampler=False,
        collate=False,
    )
    model = getattr(module_model, config["model"])(3, 64, 8, drop_ratio=0).to(device)
    if config["pretrained"]:
        model = assign_pretrained_weight(model, config["weight_path"])

    wandb.watch(model, log_freq=100)

    mask_criterion = getattr(module_loss, config["loss"])(
        weight=torch.tensor(
            [0.2857142857142857, 0.8571428571428571, 0.8571428571428571]
        ).to(device)
    )
    gender_criterion = getattr(module_loss, config["loss"])(
        weight=torch.tensor([0.3859259259259259, 0.614074074074074]).to(device)
    )
    age_criterion = getattr(module_loss, config["loss"])(
        weight=torch.tensor(
            [0.5255555555555556, 0.5455555555555556, 0.9288888888888889]
        ).to(device)
    )

    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"Epoch : {epoch}")
            sum_len = 0
            sum_loss = 0
            sum_mask_acc = 0
            sum_gender_acc = 0
            sum_age_acc = 0
            sum_mask_score = 0
            sum_gender_score = 0
            sum_age_score = 0
            label_cnt = torch.zeros(5)
            for imgs, labels in pbar:
                imgs, (mask_labels, gender_labels, age_labels) = imgs.to(device), labels
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                optimizer.zero_grad()

                pred = model(imgs)
                (mask_pred, gender_pred, age_pred) = torch.split(pred, [3, 2, 3], dim=1)

                mask_loss = mask_criterion(mask_pred, mask_labels)
                gender_loss = gender_criterion(gender_pred, gender_labels)
                age_loss = age_criterion(age_pred, age_labels)

                _, mask_pred_idx = torch.max(mask_pred, dim=1)
                _, gender_pred_idx = torch.max(gender_pred, dim=1)
                _, age_pred_idx = torch.max(age_pred, dim=1)

                loss = mask_loss + gender_loss + age_loss
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                sum_mask_acc += torch.sum(mask_pred_idx == mask_labels).item()
                sum_gender_acc += torch.sum(gender_pred_idx == gender_labels).item()
                sum_age_acc += torch.sum(age_pred_idx == age_labels).item()

                sum_len += imgs.size(0)
                sum_mask_score += f1_score(
                    mask_pred_idx.detach().cpu(),
                    mask_labels.detach().cpu(),
                    average="macro",
                )
                sum_gender_score += f1_score(
                    gender_pred_idx.detach().cpu(),
                    gender_labels.detach().cpu(),
                    average="macro",
                )
                sum_age_score += f1_score(
                    age_pred_idx.detach().cpu(),
                    age_labels.detach().cpu(),
                    average="macro",
                )

                train_loss = sum_loss / sum_len
                train_mask_acc = sum_mask_acc / sum_len
                train_gender_acc = sum_gender_acc / sum_len
                train_age_acc = sum_age_acc / sum_len
                train_mask_f1 = sum_mask_score / len(train_loader)
                train_gender_f1 = sum_gender_score / len(train_loader)
                train_age_f1 = sum_age_score / len(train_loader)

                pbar.set_postfix(
                    Loss=f"{train_loss:.3f}",
                    Mask_Acc=f"{train_mask_acc:.3f}",
                    Gender_Acc=f"{train_gender_acc:.3f}",
                    Age_Acc=f"{train_age_acc:.3f}",
                    Mask_F1=f"{train_mask_f1:.3f}",
                    Gender_F1=f"{train_gender_f1:.3f}",
                    Age_F1=f"{train_age_f1:.3f}",
                )

        wandb.log(
            {
                "train_loss": train_loss,
                "mask_acc": train_mask_acc,
                "gender_acc": train_gender_acc,
                "age_acc": train_age_acc,
                "mask_f1_score": train_mask_f1,
                "gender_f1_score": train_gender_f1,
                "age_f1_score": train_age_f1,
            },
            step=epoch,
        )

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "check_point/model_" + str(epoch) + ".pth")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs("check_point", exist_ok=True)
    main(config)
