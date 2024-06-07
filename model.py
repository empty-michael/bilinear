import torch
import torch.nn as nn
from einops import *
import numpy as np
import torchvision
from transformers import PretrainedConfig, PreTrainedModel
from torch import Tensor
from jaxtyping import Int, Float
from typing import List
from copy import deepcopy
from collections import defaultdict
from utils import define_scheduler_lambda



class MLPConfig(PretrainedConfig):
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
        input_noise: float | None = 0.33,
        batch_size: int = 100,
        validation_batch_size: int = 100,
        epochs: int = 10,
        optimizer: str = 'adamw',
        scheduler_epochs: List[int] = [2, 2, 6],
        scheduler_steps_per: int = 1,
        scheduler_min_lambda: float | None = 0.03,
        scheduler_start_lambda: float | None = None,
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
        self.input_noise = input_noise
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler_epochs = scheduler_epochs
        self.scheduler_steps_per = scheduler_steps_per
        self.scheduler_min_lambda = scheduler_min_lambda
        self.scheduler_start_lambda = scheduler_start_lambda
        
        super().__init__(**kwargs)

class BaseModel(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, x):
        pass

    def compute_metrics(self, outputs, labels):
        metrics = {}
        metrics['CE_loss'] = nn.CrossEntropyLoss()(outputs, labels)
        metrics['loss'] = nn.CrossEntropyLoss()(outputs, labels)
        return metrics
    
    def get_predictions(self, outputs):
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    def transform_inputs(self, inputs, labels):
        if self.config.dataset == 'mnist':
            inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
            labels = labels.to(self.device)
        return inputs, labels
    
    def set_dataset(self):
        if self.config.dataset == 'mnist':
            self.train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
            self.test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=torchvision.transforms.ToTensor())
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                    batch_size=self.config.validation_batch_size,
                                                    shuffle=False)
                
    def fit(self):
        self.set_dataset()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler_lambda = define_scheduler_lambda(self.config.scheduler_epochs, 
                                                   min_lambda = self.config.scheduler_min_lambda, 
                                                   steps_per = self.config.scheduler_steps_per, 
                                                   start_lambda = self.config.scheduler_start_lambda)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda)

        epochs = self.config.epochs
        n_total_steps = len(self.train_loader)
        for epoch in range(epochs):
            _ = self.validation()
            for i, (inputs, labels) in enumerate(self.train_loader):
                self.train()
                inputs, labels = self.transform_inputs(inputs, labels)
                
                # Forward pass
                outputs = self.forward(inputs)
                metrics = self.compute_metrics(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                metrics['loss'].backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    metrics_str = [f'{k}: {v.item():.4f}' for k, v in metrics.items()]
                    print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], ' + ', '.join(metrics_str))

            scheduler.step()
            print(f'learning rate = {scheduler.get_last_lr()[0]}')
        _ = self.validation()

    def validation(self, print_bool=True):
        # if not hasattr(self, 'test_loader'):
        self.set_dataset()

        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            loss_sum = 0
            count = 0
            for inputs, labels in self.test_loader:
                inputs, labels = self.transform_inputs(inputs, labels)
                outputs = self.forward(inputs)
                predicted = self.get_predictions(outputs)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                loss_sum += self.compute_metrics(outputs, labels)['CE_loss'].item()
                count += 1

            acc = 100.0 * n_correct / n_samples
            loss = loss_sum / count
            if print_bool:
              print(f'Evaluation | Accuracy: {acc:.2f} %, Loss: {loss:.4f}')
        return acc, loss


class GatedMLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        
        self.W = nn.Linear(config.d_model, 2 * config.d_hidden, bias=config.mlp_bias)
        self.Proj = nn.Linear(config.d_hidden, config.d_model, bias=False)

        if config.mlp == 'bilinear':
            self.activation = nn.Identity()
        
        if config.d_hidden == config.d_model:
            self.Proj = nn.Identity()
    
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        left, right = self.W(x).chunk(2, dim=-1)
        return self.Proj(self.activation(left) * right)
    

class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        
        self.W = nn.Linear(config.d_model, config.d_hidden, bias = config.mlp_bias)
        self.Proj = nn.Linear(config.d_hidden, config.d_model, bias=False)
        
        if config.mlp == 'relu':
            self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.Proj(self.activation(self.W(x)))
    

class LatentNoise(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.scale = config.latent_noise
    
    def forward(self, x):
        if self.training and self.scale is not None:
            # return x + torch.randn_like(x) * self.scale * torch.std(x, dim=-1, keepdim=True)
            return x + torch.randn_like(x) * self.scale * x.std()
        else:
            return x


class RMSNorm(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)


class Norm(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        
        self.norm = {
            'rms': RMSNorm,
            None: nn.Identity
        }[config.normalization](config)
        
        self.noise = LatentNoise(config)
        
    def forward(self, x):
        return self.noise(self.norm(x))
    

class Layer(nn.Module):
    def __init__(self, config: MLPConfig):
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
    

class MLPModel(BaseModel):
    config_class = MLPConfig

    def __init__(self, config: MLPConfig) -> None:
        super().__init__(config)
        self.config = config

        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)

        self.Noise = LatentNoise(config)
        self.Embed = nn.Linear(config.d_input, config.d_model, bias = False)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        self.Unembed = nn.Linear(config.d_model, config.d_output, bias = config.logit_bias)
        
    def forward(self, x):
        x = self.Noise(x)
        x = self.Embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.Unembed(x)
    
    def transform_inputs(self, inputs, labels):
        inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
        labels = labels.to(self.device)
        if self.training and (self.config.input_noise is not None):
            inputs += self.config.input_noise * torch.randn_like(inputs)
        return inputs, labels
    
    def get_B(self, layer, out_vecs = None, device = 'cpu'):
        with torch.no_grad():
            W1, W2 = self.layers[layer].mlp.W.weight.to(device).chunk(2, dim=0)
            tensors = [W1, W2]
            equation = 'h in1, h in2'
            out_var = 'h'
            if not isinstance(self.layers[layer].mlp.Proj, nn.Identity):
                # add term to einsum
                P = self.layers[layer].mlp.Proj.weight.to(device)
                tensors.append(P)
                equation += f', outp {out_var}'
                out_var = 'outp'

            if out_vecs is not None:
                # add term to einsum
                tensors.append(out_vecs.to(device))
                equation += f', outv {out_var}'
                out_var = 'outv'
            
            equation += f' -> {out_var} in1 in2'
            tensors.append(equation)
            B = einsum(*tensors)
            B = 0.5 * (B + B.transpose(-1, -2))
        return B
    

class EigLayer(nn.Module):
    def __init__(self, eigvals, config, eigvecs = None):
        super().__init__()
        self.config = config
        self.eigvals = nn.Parameter(eigvals, requires_grad=False) #[... eig]

        if eigvecs is not None:
            self.eigvecs = nn.Parameter(eigvecs, requires_grad=False) #[... eig d_model]
        else:
            self.eigvecs = None

    def forward(self, x):
        if self.eigvecs is not None:
            x = einsum(self.eigvecs, x, '... eig d_model, batch d_model -> batch ... eig')
        else:
            x = x.sum(dim=-1) # sum over previous layer's activations
        x = x ** 2
        # x = einsum(self.eigvals, x, '... eig, batch ... eig -> batch ...')
        x = self.eigvals.unsqueeze(0) * x
        
        if self.config.topk is not None:
            x = self.apply_topk(x, self.config.topk)
        return x

    def apply_topk(self, x, topk):
        shape = x.shape
        if self.config.topk_by_class:
            x = x.reshape(shape[0], shape[1], -1) # [batch, class, eig]
        else:
            x = x.reshape(shape[0], -1)
        topk = min(topk, x.shape[-1])
        _, indices = torch.topk(x.abs(), topk, dim=-1, sorted=False)
        values = x.gather(-1, indices)
        y = torch.zeros_like(x)
        y = y.scatter(-1, indices, values).reshape(shape)
        return y
        

class EigModel(BaseModel):
    def __init__(self, model: MLPModel):
        assert model.config.mlp == 'bilinear', "EigModel only works with bilinear MLPs"
        super().__init__(model.config)
        self.config = model.config
        self.config.beta_temp = 1.0
        self.config.topk = None
        self.config.topk_by_class = True
        
        self.Embed = nn.Linear(self.config.d_input, self.config.d_model, bias = False)
        self.Embed.weight = nn.Parameter(model.Embed.weight, requires_grad=False)
        self.Embed.device = self.device

        eigenvalues, eigenvectors = self.get_eigenvectors(model)
        self.layers = nn.ModuleList(self.define_layers(eigenvalues, eigenvectors))

        self.to('cpu')

    def forward(self, x):
        x = self.Embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.sum(dim=-1) * self.config.beta_temp
    
    def compute_metrics(self, outputs, labels):
        metrics = {}
        metrics['CE_loss'] = nn.CrossEntropyLoss()(outputs, labels)
        metrics['loss'] = metrics['CE_loss']
        return metrics

    def get_eigenvectors(self, model):
        with torch.no_grad():
            eigenvalues = []
            out_vectors = model.Unembed.weight.cpu()
            for layer in range(self.config.n_layer-1, -1, -1):
                B = model.get_B(layer, device='cpu')
                interaction_matrices = einsum(B, out_vectors, 'out in1 in2, ... out -> ... in1 in2')
                eigvals, eigvecs = torch.linalg.eigh(interaction_matrices)
                eigvecs = eigvecs.transpose(-1, -2) # convert to [... eig d_model]

                if self.config.normalization is not None:
                    eigvals = eigvals / eigvals.abs().max()

                eigenvalues.append(eigvals)
                out_vectors = eigvecs
        return eigenvalues, eigvecs

    def define_layers(self, eigenvalues, eigvecs):
        layers = []
        for layer_idx in range(self.config.n_layer):
            idx = self.config.n_layer - 1 - layer_idx
            eigvals = eigenvalues[idx]
            if layer_idx == 0:
                layer = EigLayer(eigvals, self.config, eigvecs = eigvecs)
            else:
                layer = EigLayer(eigvals, self.config)
            layers.append(layer)
        return layers
            
    def get_effective_eigval_magnitude(self):
        with torch.no_grad():
            eigval_mag = self.layers[0].eigvals.abs()
            for idx, layer in enumerate(self.layers[1:]):
                layer_idx = idx + 1
                shape = list(layer.eigvals.shape) + [1] * layer_idx
                power = (0.5)**(layer_idx)
                magnitude = layer.eigvals.reshape(*shape).abs()**(power)
                eigval_mag *= magnitude
        return eigval_mag


class EffectiveEigModel(EigModel):

    def __init__(self, model: MLPModel):
        super().__init__(model)
        self.config.topk = None
        self.config.topk_by_class = False
        
        eff_eigval_mags = self.get_effective_eigval_magnitude()
        self.convert_to_effective_eigvals(eff_eigval_mags)
        

    def convert_to_effective_eigvals(self, eff_eigval_mags):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.eigvals = nn.Parameter(layer.eigvals.sign() * eff_eigval_mags, requires_grad=False)
            else:
                layer.eigvals = nn.Parameter(layer.eigvals.sign(), requires_grad=False)
        


class SparseEigModel(EigModel):
    def __init__(self, model: MLPModel):
        super().__init__(model)
        self.config.L1_param = 1e-1        

        self.left_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer.eigvals.shape[:-1]), requires_grad=True) for layer in self.layers]
            )
        self.right_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer.eigvals.shape[:-1]), requires_grad=True) for layer in self.layers]
            )
        # self.scales = nn.ParameterList(
        #     [nn.Parameter(1e-3 * layer.eigvals.max() * torch.ones(layer.eigvals.shape[:-1]), requires_grad=True) for layer in self.layers]
        #     )
        
        self.beta_temp = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        x = self.Embed(x)

        aux_outputs = defaultdict(list)
        for layer, left_bias, right_bias in zip(self.layers, self.left_biases, self.right_biases):
            x, acts = layer(x)

            acts_relu = (-1) * nn.functional.relu(-(acts + left_bias)) + nn.functional.relu((acts + right_bias))
            x = x * (acts_relu.abs() > 0).float()
            x = (acts_relu)**2

            aux_outputs['L1'].append( acts_relu.abs().mean() )
            aux_outputs['L0'].append( (acts_relu.abs() > 0).float().sum() / x.shape[0] )
        return x * self.beta_temp, aux_outputs
    
    def compute_metrics(self, outputs, labels):
        logits, aux_outputs = outputs

        # change to not average over batch
        metrics = {}
        metrics['CE_loss'] = nn.CrossEntropyLoss()(logits, labels)
        metrics['L1'] = torch.sum(torch.stack(aux_outputs['L1']))
        metrics['L0'] = torch.sum(torch.stack(aux_outputs['L0']))
        metrics['loss'] = metrics['CE_loss'] + self.config.L1_param * metrics['L1']
        metrics['loss'] = metrics['CE_loss']
        return metrics
    
    def get_predictions(self, outputs):
        logits, _ = outputs
        _, predicted = torch.max(logits, 1)
        return predicted
            


# class EigModel(BaseModel):
#     def __init__(self, model: MLPModel):
#         assert model.config.mlp == 'bilinear', "EigModel only works with bilinear MLPs"
#         super().__init__(model.config)
#         self.model = model
#         self.config = self.model.config
        

#     def forward(self, x):
#         with torch.no_grad():
#             x = x.to(self.model.device)
#             x = self.model.Embed(x).cpu()
#             return torch.stack([node.forward(x) for node in self.root_nodes], dim=-1)
    
#     def transform_inputs(self, inputs, labels):
#         inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
#         labels = labels.to(self.device)
#         return inputs, labels

#     def set_eigenvector_model(self, threshold = 1e-2):
#         with torch.no_grad():
#             self.root_nodes = self.get_root_nodes()

#             nodes = self.root_nodes
#             for layer in range(self.config.n_layer-1, -1, -1):
#                 print(f"Layer {layer},   Nodes: {len(nodes)}")
#                 B = self.model.get_B(layer)

#                 max_log_eigval = -np.inf
#                 for node in nodes:
#                     child_log_eigvals = node.get_child_eigenvalues(B)
#                     max_log_eigval = max([max_log_eigval] + child_log_eigvals.tolist())
                
#                 leaf_nodes = []
#                 log_thresh = max_log_eigval + np.log(threshold)
#                 for node in nodes:
#                     leaf_nodes.extend(node.add_child_nodes(log_thresh))
                
#                 nodes = leaf_nodes
#             self.leaf_nodes = nodes

#     def get_root_nodes(self):
#         nodes = []
#         for i in range(self.model.config.d_output):
#             out_vec = self.model.Unembed.weight[i].cpu()
#             nodes.append(EigenNode(None, out_vec, 0))
#         return nodes
        
        
# class EigenNode():
#     def __init__(self, parent, eigenvector, log_eigval):
#         self.parent = parent
#         self.eigenvector = eigenvector
#         self.log_eigval = log_eigval

#         self.child_nodes = None
#         self.children_signs = None
#         self.children_log_eigvals = None
#         self.child_eigvecs = None

#     def forward(self, x):
#         # x: [batch, d_model]
#         if self.child_nodes is not None:
#             out = torch.stack([child.forward(x) * sign for child, sign in zip(self.child_nodes, self.child_signs)], dim=-1)
#             return (out.sum(dim=-1))**2
#         else:
#             return (x @ self.eigenvector)**2 * torch.exp(self.log_eigval)

#     def get_child_eigenvalues(self, B):
#         interaction_matrix = einsum(B, self.eigenvector, 'out in1 in2, out -> in1 in2')
#         eigvals, eigvecs = torch.linalg.eigh(interaction_matrix)
#         self.child_log_eigvals = torch.log(eigvals.abs()) + 0.5 * self.log_eigval
#         self.child_signs = torch.sign(eigvals)
#         self.child_eigvecs = eigvecs
#         return self.child_log_eigvals
    
#     def add_child_nodes(self, log_thresh):
#         keep = self.child_log_eigvals >= log_thresh
#         self.child_log_eigvals = self.child_log_eigvals[keep]
#         self.child_signs = self.child_signs[keep]
#         self.child_eigvecs = self.child_eigvecs[:, keep]
#         self.child_nodes = [EigenNode(self, self.child_eigvecs[:,i], self.child_log_eigvals[i]) for i in range(len(self.child_log_eigvals))]
#         return self.child_nodes





