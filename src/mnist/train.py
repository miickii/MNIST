import matplotlib.pyplot as plt
import torch
import typer
import wandb
from mnist.data import corrupt_mnist
from mnist.model import MyAwesomeModel
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # Add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # Add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0).cpu()
                wandb.log({"gradients": wandb.Histogram(grads.numpy())})

        # Add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot,
                preds[:, class_id],
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )
        
        # Log the ROC curve plot using wandb.Image
        wandb.log({"roc": wandb.Image(plt)})
        plt.close()  # Close the plot to avoid memory leaks and overlapping figures

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # Save the model to a file and log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)


def main():
    typer.run(train)
