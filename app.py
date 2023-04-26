import io
import numpy as np
import torch
import timm
from PIL import Image
import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = A.Compose(
    [
        A.CenterCrop(320, 320, p=1),
        A.Normalize(p=1),
        ToTensorV2(),
    ]
)


def model_load():
    model = timm.create_model("resnet18", pretrained=True, num_classes=8).to(device)
    model.load_state_dict(torch.load("/opt/ml/code/check_point/model_50.pth"))

    return model


def inv_normalize(img):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    for i in range(3):
        img[0, i, ...] = img[0, i, ...] * std[i] + mean[i]

    return img


def grad_cam(feature, img, model):
    feature_map = []
    backward_grad = []

    def forward_hook(module, input, output):
        feature_map.append(output)

    def backward_hook(module, grad_input, grad_output):
        backward_grad.append(grad_output)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

    pred = model(img)
    (mask_pred, gender_pred, age_pred) = torch.split(pred, [3, 2, 3], dim=1)

    mask_pred = mask_pred.squeeze()
    gender_pred = gender_pred.squeeze()
    age_pred = age_pred.squeeze()

    _, mask_pred_idx = torch.max(mask_pred, dim=0)
    _, gender_pred_idx = torch.max(gender_pred, dim=0)
    _, age_pred_idx = torch.max(age_pred, dim=0)

    if feature == "Mask":
        out = mask_pred[mask_pred_idx]
        out.backward(retain_graph=True)
    elif feature == "Gender":
        out = gender_pred[gender_pred_idx]
        out.backward(retain_graph=True)
    elif feature == "Age":
        out = age_pred[age_pred_idx]
        out.backward(retain_graph=True)

    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    img = inv_normalize(img)

    y = feature_map[19].squeeze()
    a = torch.nn.functional.adaptive_avg_pool2d(backward_grad[0][0].squeeze(), 1)

    out = torch.sum(a * y, dim=0).cpu()
    out = torch.relu(out) / torch.max(out)
    out = torch.nn.functional.interpolate(
        out.unsqueeze(0).unsqueeze(0), [320, 320], mode="bilinear"
    )
    out = out.detach().cpu().squeeze().numpy()

    ax.set_title(feature, fontsize=16)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(img.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    ax.imshow(out, alpha=0.5)

    return (mask_pred_idx, gender_pred_idx, age_pred_idx), fig


st.title("Mask Classification Model")
model = model_load()
model.eval()

upload_img = st.file_uploader("Upload Your Image", ["jpg", "jpeg", "png"])

if upload_img:
    st.info("Please Wait")

    col1, col2, col3, col4 = st.columns(4)

    img = transform(image=np.array(Image.open((io.BytesIO(upload_img.getvalue())))))
    img = img["image"].unsqueeze(0)
    invnorm_img = inv_normalize(img)

    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Origin Image", fontsize=16)
    ax.imshow(invnorm_img.squeeze().permute(1, 2, 0).detach().cpu().numpy())

    col1.pyplot(fig)

    for col, feat in zip([col2, col3, col4], ["Mask", "Gender", "Age"]):
        (mask_pred, gender_pred, age_pred), fig = grad_cam(feat, img.to(device), model)
        col.pyplot(fig)

    st.write(f"Mask : {mask_pred} | Gender : {gender_pred} | Age : {age_pred}")
    st.success("Done!")
    st.balloons()
    st.snow()
