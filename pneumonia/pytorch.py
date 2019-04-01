"""Reference PyTorch implementation of the pneumonia kaggle challange."""

# %% imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pathlib
from multiprocessing import cpu_count


# %% definitions
class FLAGS:
    """Flags."""

    batch_size = 32
    shuffle_buffer_size = 500
    data_root = pathlib.Path("chest_xray")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_names = sorted(
    dir.name for dir in FLAGS.data_root.glob("train/*") if dir.is_dir())

transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    ])

criterion = nn.BCELoss()

phases = ["train", "val", "test"]


# %% datasets
image_datasets = {phase: datasets.ImageFolder(FLAGS.data_root/phase, transform)
                  for phase in phases}

dataloaders = {phase: DataLoader(image_datasets[phase], FLAGS.batch_size,
                                 shuffle=True, num_workers=cpu_count())
               for phase in phases}


dataset_sizes = {phase: len(image_datasets[phase]) for phase in phases}


# %% training
def train_model(model: nn.Module, optimizer: optim.Optimizer, epochs=3):
    """Train model."""
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        print("-"*10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = torch.tensor(0)

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(FLAGS.device)
                labels = labels.to(FLAGS.device).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    preds = torch.sigmoid(model(inputs)).squeeze()
                    loss = criterion(preds, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += sum(torch.round(preds) == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


# %% evaluating
def eval_model(model: nn.Module):
    """Eval model."""
    model.eval()
    running_loss = 0.0
    running_corrects = torch.tensor(0)

    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(FLAGS.device)
        labels = labels.to(FLAGS.device).float()

        with torch.set_grad_enabled(False):
            preds = torch.sigmoid(model(inputs)).squeeze()
            loss = criterion(preds, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += sum(torch.round(preds) == labels.data)

    epoch_loss = running_loss / dataset_sizes["test"]
    epoch_acc = running_corrects.double() / dataset_sizes["test"]

    print(f"test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# %% model
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 1)

model.to(FLAGS.device)

optimizer = optim.Adam(model.parameters())

train_model(model, optimizer)

eval_model(model)
