import os
import yaml
import torch
import wandb
from tqdm import tqdm
from data_loader.data_loader import get_trainloader
import model.model as module_model
from util.util import get_train_transform
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
        label_type="age",
        path=config["data_path"],
        transform=get_train_transform(),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    model = getattr(module_model, config["model"])(3, 64, 5, drop_ratio=0.3).to(device)
    wandb.watch(model, log_freq=100)

    criterion = getattr(torch.nn, config["loss"])(
        weight=torch.tensor([4193, 4774, 2009, 6580, 1344]).to(
            device=device, dtype=torch.float32
        )
    )
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    for epoch in range(epochs):
        model.train()
        correct = torch.zeros(5)
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"Epoch : {epoch}")
            sum_loss = 0
            sum_acc = 0
            sum_len = 0
            sum_score = 0
            label_cnt = torch.zeros(5)
            for imgs, labels in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                pred = model(imgs)
                loss = criterion(pred, labels)

                _, pred_idx = torch.max(pred, dim=1)

                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                sum_acc += torch.sum(pred_idx == labels).item()
                sum_len += imgs.size(0)

                sum_score += f1_score(
                    pred_idx.detach().cpu(), labels.detach().cpu(), average="macro"
                )

                unique, correct_cnt = torch.unique(
                    pred_idx[pred_idx == labels].detach().cpu(), return_counts=True
                )
                correct[unique] += correct_cnt

                idx, cnt = torch.unique(labels.detach().cpu(), return_counts=True)
                label_cnt[idx] += cnt

                train_loss = sum_loss / sum_len
                train_acc = sum_acc / sum_len
                train_f1 = sum_score / len(train_loader)

                pbar.set_postfix(
                    Loss=f"{train_loss:.3f}",
                    Acc=f"{train_acc:.3f}",
                    F1=f"{train_f1:.3f}",
                )

        correct_class = correct / label_cnt
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "f1_score": train_f1,
            },
            step=epoch,
        )

        print(f"correct : {correct_class}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "check_point/model_" + str(epoch) + ".pth")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs("check_point", exist_ok=True)
    main(config)
