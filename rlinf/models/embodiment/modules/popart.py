import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlinf.models.embodiment.modules.value_head import ValueHead

class PopArtLayer(nn.Module):
    """
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets) Layer.
    Implements per-task normalization for multi-task value estimation.
    """
    def __init__(self, input_dim, output_dim, beta=0.0001):
        super().__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.output_dim = output_dim  # corresponds to num_tasks

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # Statistics buffers (per-task)
        self.register_buffer('mu', torch.zeros(output_dim, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_dim, requires_grad=False))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, task_ids=None):
        """
        Forward pass.
        If task_ids is provided, returns denormalized values for specific tasks.
        Otherwise, returns mean of all task values (for compatibility).
        """
        # Normalized output: [batch, num_tasks]
        hat_y = F.linear(input, self.weight, self.bias)
        
        if task_ids is not None:
            # Denormalize specific task output
            # Ensure task_ids are within bounds if possible
            
            mu = self.mu[task_ids]
            sigma = self.sigma[task_ids]
            
            # Select relevant output head: [batch]
            hat_y_selected = hat_y.gather(1, task_ids.unsqueeze(1)).squeeze(1)
            
            # Denormalize: y = sigma * hat_y + mu
            y = sigma * hat_y_selected + mu
            return y
        
        # When task_ids is None, return mean across all tasks for backward compatibility
        # This ensures shape [batch] instead of [batch, num_tasks]
        return hat_y.mean(dim=1)

    @torch.no_grad()
    def update_statistics(self, targets, task_ids):
        """
        Update running mean/std and adjust weights to preserve outputs.
        """
        # targets: [batch] (true unnormalized returns)
        # task_ids: [batch]
        
        # Distributed aggregation
        if torch.distributed.is_initialized():
            # Prepare tensors for aggregation
            global_sum = torch.zeros(self.output_dim, device=targets.device)
            global_sq_sum = torch.zeros(self.output_dim, device=targets.device)
            global_count = torch.zeros(self.output_dim, device=targets.device)
            
            # Local scatter add
            # Filter valid task_ids
            valid_mask = task_ids < self.output_dim
            if valid_mask.any():
                t_ids = task_ids[valid_mask]
                t_targets = targets[valid_mask]
                global_sum.scatter_add_(0, t_ids, t_targets)
                global_sq_sum.scatter_add_(0, t_ids, t_targets**2)
                global_count.scatter_add_(0, t_ids, torch.ones_like(t_targets))
            
            # All reduce
            torch.distributed.all_reduce(global_sum)
            torch.distributed.all_reduce(global_sq_sum)
            torch.distributed.all_reduce(global_count)
            
            # Compute batch stats
            mask = global_count > 0
            batch_mean = torch.zeros_like(global_sum)
            batch_mean[mask] = global_sum[mask] / global_count[mask]
            
            batch_sq_mean = torch.zeros_like(global_sum)
            batch_sq_mean[mask] = global_sq_sum[mask] / global_count[mask]
            
            # Var = E[X^2] - (E[X])^2
            batch_var = (batch_sq_mean - batch_mean**2).clamp(min=0)
            
            # Update active tasks
            if mask.any():
                old_mu = self.mu.clone()
                old_sigma = self.sigma.clone()
                
                # Update mu
                self.mu[mask] = (1 - self.beta) * self.mu[mask] + self.beta * batch_mean[mask]
                
                # Update sigma (approximate EMA on variance)
                # target_var = batch_var + (batch_mean - old_mu)^2
                target_var = batch_var[mask] + (batch_mean[mask] - self.mu[mask])**2
                old_var = self.sigma[mask]**2
                new_var = (1 - self.beta) * old_var + self.beta * target_var
                self.sigma[mask] = torch.sqrt(new_var).clamp(min=1e-4)
                
                # Adjust weights
                sigma_ratio = old_sigma / (self.sigma + 1e-8)
                self.weight.data = self.weight.data * sigma_ratio.unsqueeze(1)
                self.bias.data = (old_sigma * self.bias.data + old_mu - self.mu) / (self.sigma + 1e-8)
                
        else:
            # Fallback to local update (single GPU)
            unique_tasks = task_ids.unique()
            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()
            
            for t in unique_tasks:
                t = t.long().item()
                if t >= self.output_dim:
                    continue
                    
                mask = (task_ids == t)
                t_targets = targets[mask]
                
                if len(t_targets) == 0:
                    continue

                batch_mean = t_targets.mean()
                current_var = t_targets.var(unbiased=False) if len(t_targets) > 1 else torch.zeros_like(batch_mean)
                
                self.mu[t] = (1 - self.beta) * self.mu[t] + self.beta * batch_mean
                
                old_var = self.sigma[t] ** 2
                target_var = current_var + (batch_mean - self.mu[t])**2 
                new_var = (1 - self.beta) * old_var + self.beta * target_var
                self.sigma[t] = torch.sqrt(new_var).clamp(min=1e-4)

            sigma_ratio = old_sigma / (self.sigma + 1e-8)
            self.weight.data = self.weight.data * sigma_ratio.unsqueeze(1)
            self.bias.data = (old_sigma * self.bias.data + old_mu - self.mu) / (self.sigma + 1e-8)

    def normalize(self, values, task_ids):
        """
        Normalize values using current statistics.
        Used for computing normalized targets for loss.
        """
        return (values - self.mu[task_ids]) / (self.sigma[task_ids] + 1e-8)


class PopArtValueHead(ValueHead):
    def __init__(self, input_dim, hidden_sizes, num_tasks, beta=0.0001, activation="gelu"):
        # Initialize ValueHead with output_dim = num_tasks to set up MLP structure
        super().__init__(input_dim, hidden_sizes, output_dim=num_tasks, activation=activation, bias_last=True)
        
        # Replace last linear layer with PopArtLayer
        old_last = self.mlp[-1]
        # PopArtLayer takes (input_dim, output_dim) where output_dim is num_tasks
        self.mlp[-1] = PopArtLayer(old_last.in_features, num_tasks, beta=beta)
        
    def forward(self, x, task_ids=None):
        for layer in self.mlp:
            if isinstance(layer, PopArtLayer):
                x = layer(x, task_ids=task_ids)
            else:
                x = layer(x)
        return x
    
    def update_statistics(self, targets, task_ids):
        # Delegate to the PopArtLayer
        self.mlp[-1].update_statistics(targets, task_ids)

    def normalize(self, values, task_ids):
        # Delegate to the PopArtLayer
        return self.mlp[-1].normalize(values, task_ids)
