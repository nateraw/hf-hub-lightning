import torch
from huggingface_hub import hf_hub_download
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from hf_hub_lightning import HuggingFaceHubCallback


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST('.', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=64)

    # Initialize a trainer
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=3,
        callbacks=[HuggingFaceHubCallback('nateraw/lit-mnist')],
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)

    # To reload from 🤗 Hub
    ckpt_path = hf_hub_download('nateraw/lit-mnist', 'lit_model.ckpt')
    reloaded_model = MNISTModel.load_from_checkpoint(ckpt_path)
