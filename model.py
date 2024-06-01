import torch
import torch.nn as nn
from einops import *
import torchvision
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments, Trainer
from datasets import load_dataset
from torch import Tensor
from jaxtyping import Int, Float
from typing import Callable



class Config(PretrainedConfig):
    def __init__(
        self,
        d_input: int = 784,
        d_output: int = 10,
        n_layer: int= 1,
        d_model: int = 64,
        mlp: str = 'bilinear',
        mlp_expansion: int = 1,
        normalization: str | None = None,
        mlp_bias: bool = False,
        logit_bias: bool = False,
        random_seed: int = 0,

        dataset = 'mnist',
        lr: float = 0.001,
        weight_decay: float = 0,
        latent_noise: float | None = 0.33,
        batch_size: int = 100,
        epochs: int = 10,
        optimizer: str = 'adamw',
        scheduler_lambda:  Callable[[int], float]| None = None,

        **kwargs
    ):
        self.d_input = d_input
        self.d_output = d_output
        self.n_layer = n_layer
        self.d_model = d_model
        self.mlp = mlp
        self.mlp_expansion = mlp_expansion
        self.d_hidden = d_model * mlp_expansion
        self.normalization = normalization
        self.mlp_bias = mlp_bias
        self.logit_bias = logit_bias
        self.random_seed = random_seed

        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.latent_noise = latent_noise
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler_lambda = scheduler_lambda
        
        super().__init__(**kwargs)


class GatedMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.W = nn.Linear(config.d_model, 2 * config.d_hidden, bias=config.mlp_bias)
        self.Proj = nn.Linear(config.d_hidden, config.d_model, bias=False)

        if config.mlp == 'bilinear':
            self.activation = lambda x: x
        
        if config.d_hidden == config.d_model:
            self.Proj.weight = nn.Parameter(torch.eye(config.d_model), requires_grad=False)
    
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        left, right = self.W(x).chunk(2, dim=-1)
        return self.Proj(self.activation(left) * right)
    

class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.W = nn.Linear(config.d_model, config.d_hidden, bias = config.mlp_bias)
        self.Proj = nn.Linear(config.d_hidden, config.d_model, bias=False)
        
        if config.mlp == 'relu':
            self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.Proj(self.activation(self.W(x)))
    

class LatentNoise(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.scale = config.latent_noise
    
    def forward(self, x):
        if self.training and self.scale is not None:
            # return x + torch.randn_like(x) * self.scale * torch.std(x, dim=-1, keepdim=True)
            return x + torch.randn_like(x) * self.scale * x.std()
        else:
            return x


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)


class Norm(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.norm = {
            'rms': RMSNorm,
            None: nn.Identity
        }[config.normalization](config)
        
        self.noise = LatentNoise(config)
        
    def forward(self, x):
        return self.noise(self.norm(x))
    

class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        mlp_fn = dict(bilinear=GatedMLP, 
                      relu=MLP
                      )
        
        self.mlp = mlp_fn[config.mlp](config)
        
        self.norm = Norm(config)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x
    
class MLPModel(PreTrainedModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)

        # self.Noise = LatentNoise(config)
        self.Embed = nn.Linear(config.d_input, config.d_model, bias = False)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        self.Unembed = nn.Linear(config.d_model, config.d_output, bias = config.logit_bias)
        
    def forward(self, x):
        # x = self.Noise(x)
        x = self.Embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.Unembed(x)
    
    def criterion(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
    
    def dataset(self, loader = True):
        if self.config.dataset == 'mnist':
            train = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
            test = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=torchvision.transforms.ToTensor())
            if loader:
                train_loader = torch.utils.data.DataLoader(dataset=train,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=True)
                test_loader = torch.utils.data.DataLoader(dataset=test,
                                                    batch_size=self.config.batch_size,
                                                    shuffle=False)
                return train_loader, test_loader
            else:
                return train, test
    
    def fit(self):
        train, test = self.dataset()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.config.scheduler_lambda)

        epochs = self.config.epochs
        n_total_steps = len(train)
        for epoch in range(epochs):
            _ = self.validation(test)
            for i, (inputs, labels) in enumerate(train):
                self.train()
                inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
                labels = labels.to(self.device)

                if self.input_noise is not None:
                    inputs += self.config.input_noise * torch.randn_like(inputs)
                
                # Forward pass
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            scheduler.step()
            print(f'learning rate = {scheduler.get_last_lr()[0]}')
        _ = self.validation(test)

    def validation(self, test_loader, print_bool=True):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            loss_sum = 0
            count = 0
            for inputs, labels in test_loader:
                inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                loss_sum += self.criterion(outputs, labels).item()
                count += 1

            acc = 100.0 * n_correct / n_samples
            loss = loss_sum / count
            if print_bool:
              print(f'Evaluation | Accuracy: {acc:.2f} %, Loss: {loss:.4f}')
        return acc, loss



