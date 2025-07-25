import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, encoded_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the decoder MLP with four layers


class Decoder(nn.Module):
    def __init__(self, encoded_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(encoded_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x



num_classes = 21
encoded_dim = 10
lossf = nn.MSELoss()

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        ground_truth = (batch['gt'].to(device)).to(dtype=torch.float32)

        num_obs = voxel_observations.shape[1]

        ground_truth = torch.mean(voxel_observations, dim=1)

        encodeinput = voxel_observations.view(-1, num_classes)
        # print(encodeinput.shape)

        fused_encoded_vectors = self.encoder(encodeinput)
        # fused_encoded_vector = torch.mean(fused_encoded_vector, dim=0)
        # print(f' fused shape {fused_encoded_vectors.shape}')
        fused_encoded_vectors = fused_encoded_vectors.view(-1, num_obs, encoded_dim)
        fused_encoded_vector = torch.mean(fused_encoded_vectors, dim=1)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # Compute the loss
        loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("train_loss",loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        ground_truth = (batch['gt'].to(device)).to(dtype=torch.float32)

        num_obs = voxel_observations.shape[1]

        ground_truth = torch.mean(voxel_observations, dim=1)

        encodeinput = voxel_observations.view(-1, num_classes)
        # print(encodeinput.shape)

        fused_encoded_vectors = self.encoder(encodeinput)
        # fused_encoded_vector = torch.mean(fused_encoded_vector, dim=0)
        # print(f' fused shape {fused_encoded_vectors.shape}')
        fused_encoded_vectors = fused_encoded_vectors.view(-1, num_obs, encoded_dim)
        fused_encoded_vector = torch.mean(fused_encoded_vectors, dim=1)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # Compute the loss
        test_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("test_loss", test_loss)
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        voxel_observations = (batch['obs'].to(device)).to(dtype=torch.float32)
        ground_truth = (batch['gt'].to(device)).to(dtype=torch.float32)

        num_obs = voxel_observations.shape[1]

        ground_truth = torch.mean(voxel_observations, dim=1)

        encodeinput = voxel_observations.view(-1, num_classes)
        # print(encodeinput.shape)

        fused_encoded_vectors = self.encoder(encodeinput)
        # fused_encoded_vector = torch.mean(fused_encoded_vector, dim=0)
        # print(f' fused shape {fused_encoded_vectors.shape}')
        fused_encoded_vectors = fused_encoded_vectors.view(-1, num_obs, encoded_dim)
        fused_encoded_vector = torch.mean(fused_encoded_vectors, dim=1)

        output_vector_encoded_fusion = self.decoder(fused_encoded_vector)
        # Compute the loss
        val_loss = lossf(output_vector_encoded_fusion, ground_truth)
        self.log("val_loss", val_loss)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self,x):
        classes = x.shape[1]
        y = self.encoder(x)
        # y = y.view(-1, num_obs, encoded_dim)
        # y_hat = torch.mean(y, dim=1)
        # z = self.decoder(y_hat)
        return y
    
    def encode(self, x):
        y = self.encoder(x)
        return y


    def decode(self, x):
        y = self.decoder(x)
        # y = y.view(-1, num_obs, encoded_dim)
        # y_hat = torch.mean(y, dim=1)
        # z = self.decoder(y_hat)
        return y


       
