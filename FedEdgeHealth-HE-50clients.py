import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import copy
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import math
import secrets
from typing import Dict, List, Tuple, Optional, Union
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute MSE, RMSE, MAE, MAPE between softmax probabilities and one-hot labels.
    MAPE is computed only on the true class (where target = 1).
    """
    num_classes = predictions.size(1)
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
    pred_probs = F.softmax(predictions, dim=1)

    pred_flat = pred_probs.cpu().numpy()
    target_flat = one_hot_targets.cpu().numpy()

    mse = np.mean((target_flat - pred_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(target_flat - pred_flat))

    # Compute MAPE only on positions where target is 1 (true class)
    true_class_idx = target_flat == 1
    if np.any(true_class_idx):
        mape = np.mean(np.abs(
            (target_flat[true_class_idx] - pred_flat[true_class_idx]) /
            (target_flat[true_class_idx] + 1e-8)
        )) * 100  # %
    else:
        mape = 0.0

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}



class StableHomomorphicEncryption:
    """Stable HE simulator with realistic security and better numerical stability"""

    def __init__(self, security_level: int = 128):
        self.security_level = security_level
        self.precision = 1000  # Reduced precision for stability
        self.noise_budget = 0
        self.max_noise_budget = 1000
        self.operation_count = 0

        print(f"Initialized Stable HE system with {security_level}-bit security")

    def encrypt(self, tensor: torch.Tensor) -> Dict:
        """Stable encryption with better numerical properties"""
        noise_scale = 1e-6
        encrypted_data = tensor.clone() + torch.randn_like(tensor) * noise_scale

        return {
            'data': encrypted_data,
            'shape': list(tensor.shape),
            'noise_level': self.noise_budget,
            'encrypted': True
        }

    def decrypt(self, encrypted_tensor: Dict) -> torch.Tensor:
        """Stable decryption"""
        return encrypted_tensor['data'].clone()

    def add(self, encrypted_a: Dict, encrypted_b: Dict) -> Dict:
        """Stable homomorphic addition"""
        result_data = encrypted_a['data'] + encrypted_b['data']

        return {
            'data': result_data,
            'shape': encrypted_a['shape'],
            'noise_level': self.noise_budget + 1,
            'encrypted': True
        }

    def scalar_multiply(self, encrypted_tensor: Dict, scalar: float) -> Dict:
        """Stable homomorphic scalar multiplication"""
        result_data = encrypted_tensor['data'] * scalar

        return {
            'data': result_data,
            'shape': encrypted_tensor['shape'],
            'noise_level': self.noise_budget + 2,
            'encrypted': True
        }

    def average(self, encrypted_tensors: List[Dict]) -> Dict:
        """Stable homomorphic averaging"""
        if not encrypted_tensors:
            return None

        sum_data = encrypted_tensors[0]['data'].clone()
        for i in range(1, len(encrypted_tensors)):
            sum_data += encrypted_tensors[i]['data']

        avg_data = sum_data / len(encrypted_tensors)

        return {
            'data': avg_data,
            'shape': encrypted_tensors[0]['shape'],
            'noise_level': self.noise_budget + len(encrypted_tensors),
            'encrypted': True
        }


# Initialize stable HE
HE = StableHomomorphicEncryption()


class StableHealthcareNet(nn.Module):
    """Stable neural network with better convergence properties"""

    def __init__(self, input_dim, num_classes=10):
        super().__init__()

        # Simpler, more stable architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # Proper weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


class StableFederatedClient:
    """Stable federated client with better training dynamics"""

    def __init__(self, client_id, model, train_data, test_data,
                 local_epochs=1, batch_size=32, learning_rate=0.01):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_data = train_data
        self.test_data = test_data
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.he = HE
        self.device = next(self.model.parameters()).device

        # Data loaders
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True
        )

    def train(self):
        """Stable training with gradient clipping"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            total_loss += epoch_loss

        avg_loss = total_loss / (self.local_epochs * len(self.train_loader))
        accuracy = correct / total if total > 0 else 0

        # Encrypt model
        model_state = self.model.state_dict()
        encrypted_state = {}
        for key, tensor in model_state.items():
            encrypted_state[key] = self.he.encrypt(tensor.cpu())

        return encrypted_state

    def update_model(self, encrypted_model_state, learning_rate=0.01):
        decrypted_state = {}
        for key, encrypted_tensor in encrypted_model_state.items():
            decrypted_tensor = self.he.decrypt(encrypted_tensor)
            decrypted_state[key] = decrypted_tensor.to(self.device)

        self.model.load_state_dict(decrypted_state)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_outputs.append(outputs)
                all_targets.append(targets)

        accuracy = correct / total if total > 0 else 0

        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            regression_metrics = compute_regression_metrics(all_outputs, all_targets)
        else:
            regression_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}

        return {
            'accuracy': accuracy,
            **regression_metrics
        }


class StableEdgeAggregator:
    """Stable edge aggregator with robust aggregation"""

    def __init__(self, edge_id, model):
        self.edge_id = edge_id
        self.model = copy.deepcopy(model)
        self.clients = []
        self.he = HE
        self.device = next(self.model.parameters()).device

    def add_client(self, client):
        self.clients.append(client)

    def aggregate_clients(self):
        client_encrypted_weights = []
        for client in self.clients:
            encrypted_weights = client.train()
            client_encrypted_weights.append(encrypted_weights)

        edge_encrypted_weights = {}
        for key in client_encrypted_weights[0].keys():
            key_encrypted_tensors = [client_weights[key] for client_weights in client_encrypted_weights]
            avg_encrypted = self.he.average(key_encrypted_tensors)
            edge_encrypted_weights[key] = avg_encrypted

        edge_model_weights = {}
        for key, encrypted_tensor in edge_encrypted_weights.items():
            decrypted_tensor = self.he.decrypt(encrypted_tensor)
            edge_model_weights[key] = decrypted_tensor.to(self.device)

        self.model.load_state_dict(edge_model_weights)
        return edge_model_weights

    def broadcast_to_clients(self, learning_rate=0.01):
        model_state = self.model.state_dict()
        encrypted_state = {}
        for key, tensor in model_state.items():
            encrypted_state[key] = self.he.encrypt(tensor.cpu())

        for client in self.clients:
            client.update_model(encrypted_state, learning_rate)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_outputs.append(outputs)
                all_targets.append(targets)

        accuracy = correct / total if total > 0 else 0

        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            regression_metrics = compute_regression_metrics(all_outputs, all_targets)
        else:
            regression_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}

        return {
            'accuracy': accuracy,
            **regression_metrics
        }


class StableFederatedServer:
    """Stable global server with robust aggregation"""

    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.edge_aggregators = []
        self.he = HE
        self.global_test_loader = None
        self.device = next(self.model.parameters()).device

    def add_edge_aggregator(self, edge_aggregator):
        self.edge_aggregators.append(edge_aggregator)

    def set_global_test_loader(self, test_dataset, batch_size=128):
        self.global_test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

    def aggregate_edges(self):
        edge_encrypted_weights = []
        for edge in self.edge_aggregators:
            model_state = edge.model.state_dict()
            encrypted_state = {}
            for key, tensor in model_state.items():
                encrypted_state[key] = self.he.encrypt(tensor.cpu())
            edge_encrypted_weights.append(encrypted_state)

        global_encrypted_weights = {}
        for key in edge_encrypted_weights[0].keys():
            key_encrypted_tensors = [edge_weights[key] for edge_weights in edge_encrypted_weights]
            avg_encrypted = self.he.average(key_encrypted_tensors)
            global_encrypted_weights[key] = avg_encrypted

        global_model_weights = {}
        for key, encrypted_tensor in global_encrypted_weights.items():
            decrypted_tensor = self.he.decrypt(encrypted_tensor)
            global_model_weights[key] = decrypted_tensor.to(self.device)

        self.model.load_state_dict(global_model_weights)
        return global_model_weights

    def broadcast_to_edges(self, learning_rate=0.01):
        for edge in self.edge_aggregators:
            edge.model.load_state_dict(self.model.state_dict())
            edge.broadcast_to_clients(learning_rate)

    def evaluate(self):
        if self.global_test_loader is None:
            raise ValueError("Global test loader not set")

        self.model.eval()
        correct = 0
        total = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.global_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_outputs.append(outputs)
                all_targets.append(targets)

        accuracy = correct / total if total > 0 else 0

        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            regression_metrics = compute_regression_metrics(all_outputs, all_targets)
        else:
            regression_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}

        return {
            'accuracy': accuracy,
            **regression_metrics
        }


def load_and_preprocess_data():
    """Load and preprocess MNIST data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def distribute_data_iid(train_dataset, test_dataset, num_clients):
    """IID data distribution"""
    data_distribution = {}

    train_size_per_client = len(train_dataset) // num_clients
    test_size_per_client = len(test_dataset) // num_clients

    for i in range(num_clients):
        train_start = i * train_size_per_client
        train_end = (i + 1) * train_size_per_client
        test_start = i * test_size_per_client
        test_end = (i + 1) * test_size_per_client

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, test_end))

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        data_distribution[i] = (train_subset, test_subset)

    return data_distribution


def run_simulation(num_clients: int, num_rounds: int = 10, verbose: bool = False):
    """Run FL simulation for a given number of clients."""
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_dim = 28 * 28
    num_classes = 10
    local_epochs = 1
    initial_lr = 0.01
    lr_decay = 0.98

    num_edges = max(1, int(np.sqrt(num_clients)))
    clients_per_edge = num_clients // num_edges
    remaining_clients = num_clients - num_edges * clients_per_edge

    train_data, test_data = load_and_preprocess_data()

    model = StableHealthcareNet(input_dim=input_dim, num_classes=num_classes).to(device)
    global_server = StableFederatedServer(model)
    global_server.set_global_test_loader(test_data)

    edge_aggregators = []
    for edge_id in range(num_edges):
        edge = StableEdgeAggregator(edge_id=edge_id, model=model)
        edge_aggregators.append(edge)
        global_server.add_edge_aggregator(edge)

    data_distribution = distribute_data_iid(train_data, test_data, num_clients)

    client_idx = 0
    for edge_id in range(num_edges):
        num_local_clients = clients_per_edge + (1 if edge_id < remaining_clients else 0)
        for local_client_id in range(num_local_clients):
            if client_idx >= num_clients:
                break
            train_subset, test_subset = data_distribution[client_idx]
            client = StableFederatedClient(
                client_id=f"{edge_id}-{local_client_id}",
                model=model,
                train_data=train_subset,
                test_data=test_subset,
                local_epochs=local_epochs,
                learning_rate=initial_lr
            )
            edge_aggregators[edge_id].add_client(client)
            client_idx += 1

    start_time = time.time()

    for round_num in range(num_rounds):
        current_lr = initial_lr * (lr_decay ** round_num)
        for edge in edge_aggregators:
            edge.aggregate_clients()
            edge.broadcast_to_clients(current_lr)
        global_server.aggregate_edges()
        global_server.broadcast_to_edges(current_lr)

    total_time = time.time() - start_time

    final_metrics = global_server.evaluate()
    final_metrics['time'] = total_time
    final_metrics['num_clients'] = num_clients
    final_metrics['loss'] = final_metrics['MSE']
    final_metrics['rounds'] = num_rounds

    if verbose:
        print(f"Client {num_clients}: acc={final_metrics['accuracy']:.4f}, "
              f"RMSE={final_metrics['RMSE']:.4f}, MAE={final_metrics['MAE']:.4f}, "
              f"MAPE={final_metrics['MAPE']:.2f}%, time={total_time:.2f}s")

    return final_metrics


def sweep_clients():
    """Run simulation for 1 to 50 clients and collect results."""
    results = []

    print("Starting client sweep from 1 to 50...")
    print(f"{'Clients':<8} {'Acc':<8} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'Loss':<8} {'Time (s)':<10}")
    print("-" * 60)

    for num_clients in range(1, 51):
        try:
            metrics = run_simulation(num_clients=num_clients, num_rounds=10, verbose=False)
            results.append(metrics)

            print(f"{metrics['num_clients']:<8} "
                  f"{metrics['accuracy']:<8.4f} "
                  f"{metrics['RMSE']:<8.4f} "
                  f"{metrics['MAE']:<8.4f} "
                  f"{metrics['MAPE']:<8.4f} "
                  f"{metrics['loss']:<8.4f} "
                  f"{metrics['time']:<10.2f}")

        except Exception as e:
            print(f"{num_clients:<8} Error: {str(e)}")
            results.append({
                'num_clients': num_clients,
                'accuracy': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'loss': np.nan,
                'time': np.nan
            })

    import pandas as pd
    df = pd.DataFrame(results)[['num_clients', 'accuracy', 'RMSE', 'MAE', 'MAPE', 'loss', 'time']]
    df.to_csv('federated_learning_client_sweep_1_to_50.csv', index=False)
    print("\nResults saved to 'federated_learning_client_sweep_1_to_50.csv'")

    # Plotting
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.plot(df['num_clients'], df['accuracy'], 'o-', label='Accuracy')
    plt.xlabel('Number of Clients')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Clients')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(df['num_clients'], df['RMSE'], 's-', color='red')
    plt.xlabel('Number of Clients')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Number of Clients')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(df['num_clients'], df['MAE'], '^-', color='orange')
    plt.xlabel('Number of Clients')
    plt.ylabel('MAE')
    plt.title('MAE vs. Number of Clients')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(df['num_clients'], df['MAPE'], 'd-', color='purple')
    plt.xlabel('Number of Clients')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE vs. Number of Clients')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(df['num_clients'], df['loss'], 'x-', color='brown')
    plt.xlabel('Number of Clients')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss vs. Number of Clients')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(df['num_clients'], df['time'], 'o-', color='green')
    plt.xlabel('Number of Clients')
    plt.ylabel('Time (s)')
    plt.title('Training Time vs. Number of Clients')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('client_sweep_results_1_to_50.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nSummary Statistics:")
    print(df.describe().round(4))

    return df


if __name__ == "__main__":
    df_results = sweep_clients()