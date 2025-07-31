#!/usr/bin/env python3
"""
Neural Network Prediction Engine - Advanced Deep Learning Models
Transformer, LSTM, CNN, and GNN models for cricket performance prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Neural Engine using device: {device}")

@dataclass
class NeuralPrediction:
    """Neural network prediction output"""
    expected_points: float
    confidence_score: float
    performance_range: Tuple[float, float]  # (min, max)
    feature_importance: Dict[str, float]
    uncertainty: float

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for performance sequence analysis"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class PerformanceTransformer(nn.Module):
    """Transformer model for analyzing performance sequences"""
    
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, 128)
        
    def forward(self, x, mask=None):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Global average pooling
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TransformerLayer(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, attn_weights = self.multi_head_attention(x, x, x, mask)
        
        # Feed forward
        ff_output = self.feed_forward(attn_output)
        ff_output = self.dropout(ff_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(ff_output + attn_output)
        
        return output, attn_weights

class FormLSTM(nn.Module):
    """LSTM for analyzing recent form trends"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=4, batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size * 2, 64)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Self-attention on LSTM outputs
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output
        output = self.output_layer(attn_out[:, -1, :])
        
        return output, attn_weights

class PlayerInteractionGNN(nn.Module):
    """Graph Neural Network for modeling player interactions"""
    
    def __init__(self, node_features: int, hidden_dim: int = 64):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.graph_conv1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.graph_conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 32)
        
    def forward(self, node_features, adjacency_matrix):
        # Node embedding
        x = F.relu(self.node_embedding(node_features))
        
        # Graph convolutions
        x = F.relu(self.graph_conv1(x, adjacency_matrix))
        x = F.relu(self.graph_conv2(x, adjacency_matrix))
        
        # Global pooling (sum pooling)
        graph_embedding = torch.sum(x, dim=1)
        
        # Readout
        output = self.readout(graph_embedding)
        
        return output

class GraphConvLayer(nn.Module):
    """Graph convolution layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adjacency):
        # x: [batch_size, num_nodes, in_features]
        # adjacency: [batch_size, num_nodes, num_nodes]
        
        x = self.linear(x)
        x = torch.bmm(adjacency, x)  # Graph convolution
        
        return x

class RoleSpecificNetwork(nn.Module):
    """Role-specific neural networks for different player types"""
    
    def __init__(self, input_dim: int, role_type: str):
        super().__init__()
        
        self.role_type = role_type
        
        if role_type == 'batsman':
            self.network = self._build_batsman_network(input_dim)
        elif role_type == 'bowler':
            self.network = self._build_bowler_network(input_dim)
        elif role_type == 'allrounder':
            self.network = self._build_allrounder_network(input_dim)
        elif role_type == 'wicketkeeper':
            self.network = self._build_wicketkeeper_network(input_dim)
        else:
            self.network = self._build_generic_network(input_dim)
    
    def _build_batsman_network(self, input_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def _build_bowler_network(self, input_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def _build_allrounder_network(self, input_dim: int):
        # More complex network for all-rounders
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def _build_wicketkeeper_network(self, input_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def _build_generic_network(self, input_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        return self.network(x)

class NeuralPerformancePredictor(nn.Module):
    """Main neural network ensemble for performance prediction"""
    
    def __init__(self, sequence_input_dim: int = 10, static_input_dim: int = 50):
        super().__init__()
        
        # Component networks
        self.performance_transformer = PerformanceTransformer(sequence_input_dim)
        self.form_lstm = FormLSTM(sequence_input_dim)
        self.player_interaction_gnn = PlayerInteractionGNN(static_input_dim)
        
        # Role-specific networks
        self.role_networks = nn.ModuleDict({
            'batsman': RoleSpecificNetwork(static_input_dim, 'batsman'),
            'bowler': RoleSpecificNetwork(static_input_dim, 'bowler'),
            'allrounder': RoleSpecificNetwork(static_input_dim, 'allrounder'),
            'wicketkeeper': RoleSpecificNetwork(static_input_dim, 'wicketkeeper'),
            'generic': RoleSpecificNetwork(static_input_dim, 'generic')
        })
        
        # Fusion layers
        fusion_input_dim = 128 + 64 + 32 + 32  # transformer + lstm + gnn + role
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-task outputs
        self.performance_head = nn.Linear(64, 1)  # Expected points
        self.confidence_head = nn.Linear(64, 1)   # Confidence score
        self.uncertainty_head = nn.Linear(64, 1)  # Uncertainty estimation
        self.captain_head = nn.Linear(64, 1)      # Captain probability
        
    def forward(self, sequence_data, static_data, adjacency_matrix, role_type='generic'):
        # Extract features from different components
        
        # Transformer on performance sequence
        transformer_out, transformer_attention = self.performance_transformer(sequence_data)
        
        # LSTM on recent form
        lstm_out, lstm_attention = self.form_lstm(sequence_data)
        
        # GNN on player interactions
        gnn_out = self.player_interaction_gnn(static_data, adjacency_matrix)
        
        # Role-specific network
        role_network = self.role_networks.get(role_type, self.role_networks['generic'])
        role_out = role_network(static_data.mean(dim=1))  # Global pooling for role network
        
        # Fusion
        combined_features = torch.cat([transformer_out, lstm_out, gnn_out, role_out], dim=1)
        fused_features = self.fusion_network(combined_features)
        
        # Multi-task outputs
        expected_points = self.performance_head(fused_features)
        confidence_score = torch.sigmoid(self.confidence_head(fused_features))
        uncertainty = torch.sigmoid(self.uncertainty_head(fused_features))
        captain_probability = torch.sigmoid(self.captain_head(fused_features))
        
        return {
            'expected_points': expected_points,
            'confidence_score': confidence_score,
            'uncertainty': uncertainty,
            'captain_probability': captain_probability,
            'attention_weights': {
                'transformer': transformer_attention,
                'lstm': lstm_attention
            }
        }

class CricketDataset(Dataset):
    """Dataset class for cricket performance data"""
    
    def __init__(self, sequence_data, static_data, adjacency_matrices, 
                 targets, roles):
        self.sequence_data = torch.FloatTensor(sequence_data)
        self.static_data = torch.FloatTensor(static_data)
        self.adjacency_matrices = torch.FloatTensor(adjacency_matrices)
        self.targets = torch.FloatTensor(targets)
        self.roles = roles
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'sequence_data': self.sequence_data[idx],
            'static_data': self.static_data[idx],
            'adjacency_matrix': self.adjacency_matrices[idx],
            'target': self.targets[idx],
            'role': self.roles[idx]
        }

class NeuralPredictionEngine:
    """Main engine for neural network-based predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler_sequence = StandardScaler()
        self.scaler_static = StandardScaler()
        self.device = device
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 15
        
    def prepare_data(self, player_data_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for neural network training"""
        
        sequence_data = []
        static_data = []
        adjacency_matrices = []
        targets = []
        roles = []
        
        for player_data in player_data_list:
            # Sequence data (recent performances)
            recent_form = player_data.get('recent_form', [])
            sequence = self._extract_sequence_features(recent_form)
            sequence_data.append(sequence)
            
            # Static data (player characteristics)
            static = self._extract_static_features(player_data)
            static_data.append(static)
            
            # Adjacency matrix (player interactions - simplified)
            adj_matrix = self._create_adjacency_matrix(player_data)
            adjacency_matrices.append(adj_matrix)
            
            # Target (fantasy points)
            target = player_data.get('expected_points', 50.0)
            targets.append(target)
            
            # Role
            role = self._normalize_role(player_data.get('role', 'generic'))
            roles.append(role)
        
        return (
            np.array(sequence_data),
            np.array(static_data),
            np.array(adjacency_matrices),
            np.array(targets),
            roles
        )
    
    def _extract_sequence_features(self, recent_form: List[Dict[str, Any]], max_len: int = 20) -> np.ndarray:
        """Extract sequence features from recent form"""
        
        # Pad or truncate to max_len
        if len(recent_form) > max_len:
            recent_form = recent_form[:max_len]
        
        sequence = []
        for match in recent_form:
            features = [
                match.get('runs', 0),
                match.get('wickets', 0),
                match.get('catches', 0),
                match.get('fantasy_points', 0),
                match.get('strike_rate', 100) / 100,  # Normalize
                match.get('bowling_figures_runs', 30) / 50,  # Normalize
                match.get('bowling_figures_wickets', 0),
                match.get('economy_rate', 7) / 10,  # Normalize
                match.get('batting_position', 7) / 11,  # Normalize
                1.0  # Match played indicator
            ]
            sequence.append(features)
        
        # Pad with zeros if needed
        while len(sequence) < max_len:
            sequence.append([0.0] * 10)
        
        return np.array(sequence)
    
    def _extract_static_features(self, player_data: Dict[str, Any]) -> np.ndarray:
        """Extract static features for a player"""
        
        features = [
            # Performance metrics
            player_data.get('ema_score', 0) / 100,
            player_data.get('consistency_score', 50) / 100,
            player_data.get('performance_rating', 50) / 100,
            player_data.get('form_momentum', 0),
            player_data.get('dynamic_opportunity_index', 1.0),
            
            # Batting stats
            player_data.get('batting_stats', {}).get('average', 30) / 50,
            player_data.get('batting_stats', {}).get('strikeRate', 120) / 200,
            
            # Bowling stats
            player_data.get('bowling_stats', {}).get('average', 30) / 50,
            player_data.get('bowling_stats', {}).get('economy', 7) / 12,
            player_data.get('bowling_stats', {}).get('wickets', 50) / 100,
            
            # Role encoding
            self._encode_role(player_data.get('role', 'batsman')),
            
            # Contextual features
            player_data.get('captain_vice_captain_probability', 0) / 100,
            player_data.get('injury_risk', 0.1),
            player_data.get('fatigue_index', 0.0),
            player_data.get('psychological_state', 0.5),
        ]
        
        # Extend to fixed size
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def _create_adjacency_matrix(self, player_data: Dict[str, Any], num_players: int = 22) -> np.ndarray:
        """Create adjacency matrix for player interactions"""
        
        # Simplified adjacency matrix
        # In reality, this would model actual player interactions
        adj_matrix = np.eye(num_players)  # Self connections
        
        # Add some random connections for demonstration
        np.random.seed(player_data.get('player_id', 0))
        random_connections = np.random.rand(num_players, num_players) < 0.1
        adj_matrix = adj_matrix + random_connections.astype(float)
        
        # Normalize
        adj_matrix = adj_matrix / (adj_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        return adj_matrix
    
    def _normalize_role(self, role: str) -> str:
        """Normalize role names"""
        role_lower = role.lower()
        
        if 'wicket' in role_lower or 'wk' in role_lower:
            return 'wicketkeeper'
        elif 'allrounder' in role_lower or 'all-rounder' in role_lower:
            return 'allrounder'
        elif 'bowl' in role_lower:
            return 'bowler'
        elif 'bat' in role_lower:
            return 'batsman'
        else:
            return 'generic'
    
    def _encode_role(self, role: str) -> float:
        """Encode role as numerical value"""
        role_mapping = {
            'batsman': 0.2,
            'bowler': 0.4,
            'allrounder': 0.6,
            'wicketkeeper': 0.8,
            'generic': 0.0
        }
        
        normalized_role = self._normalize_role(role)
        return role_mapping.get(normalized_role, 0.0)
    
    def train_model(self, sequence_data: np.ndarray, static_data: np.ndarray,
                   adjacency_matrices: np.ndarray, targets: np.ndarray, 
                   roles: List[str]) -> Dict[str, float]:
        """Train the neural network model"""
        
        # Scale data
        sequence_data_scaled = self.scaler_sequence.fit_transform(
            sequence_data.reshape(-1, sequence_data.shape[-1])
        ).reshape(sequence_data.shape)
        
        static_data_scaled = self.scaler_static.fit_transform(static_data)
        
        # Create dataset
        dataset = CricketDataset(
            sequence_data_scaled, static_data_scaled, 
            adjacency_matrices, targets, roles
        )
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.model = NeuralPerformancePredictor(
            sequence_input_dim=sequence_data.shape[-1],
            static_input_dim=static_data.shape[-1]
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"🔧 Training neural network for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move data to device
                sequence_data_batch = batch['sequence_data'].to(self.device)
                static_data_batch = batch['static_data'].to(self.device)
                adjacency_batch = batch['adjacency_matrix'].to(self.device)
                targets_batch = batch['target'].to(self.device)
                roles_batch = batch['role']
                
                # Forward pass
                outputs = self.model(
                    sequence_data_batch, static_data_batch, 
                    adjacency_batch, roles_batch[0] if roles_batch else 'generic'
                )
                
                # Calculate loss
                loss = criterion(outputs['expected_points'].squeeze(), targets_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    sequence_data_batch = batch['sequence_data'].to(self.device)
                    static_data_batch = batch['static_data'].to(self.device)
                    adjacency_batch = batch['adjacency_matrix'].to(self.device)
                    targets_batch = batch['target'].to(self.device)
                    roles_batch = batch['role']
                    
                    outputs = self.model(
                        sequence_data_batch, static_data_batch,
                        adjacency_batch, roles_batch[0] if roles_batch else 'generic'
                    )
                    
                    loss = criterion(outputs['expected_points'].squeeze(), targets_batch)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_model()
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
    
    def predict(self, player_data: Dict[str, Any], 
                match_context: Dict[str, Any] = None) -> NeuralPrediction:
        """Make prediction for a single player"""
        
        if self.model is None:
            # Return default prediction if model not trained
            return NeuralPrediction(
                expected_points=50.0,
                confidence_score=0.5,
                performance_range=(30.0, 70.0),
                feature_importance={},
                uncertainty=0.3
            )
        
        self.model.eval()
        
        with torch.no_grad():
            # Prepare data
            sequence_data, static_data, adjacency_matrix, _, roles = self.prepare_data([player_data])
            
            # Scale data
            sequence_data_scaled = self.scaler_sequence.transform(
                sequence_data.reshape(-1, sequence_data.shape[-1])
            ).reshape(sequence_data.shape)
            
            static_data_scaled = self.scaler_static.transform(static_data)
            
            # Convert to tensors
            sequence_tensor = torch.FloatTensor(sequence_data_scaled).to(self.device)
            static_tensor = torch.FloatTensor(static_data_scaled).to(self.device)
            adjacency_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
            
            # Forward pass
            outputs = self.model(
                sequence_tensor, static_tensor, adjacency_tensor, roles[0]
            )
            
            # Extract predictions
            expected_points = outputs['expected_points'].item()
            confidence_score = outputs['confidence_score'].item()
            uncertainty = outputs['uncertainty'].item()
            
            # Calculate performance range
            std_dev = uncertainty * 20  # Scale uncertainty to points
            performance_range = (
                max(0, expected_points - 2 * std_dev),
                expected_points + 2 * std_dev
            )
            
            # Feature importance (simplified)
            feature_importance = {
                'recent_form': 0.3,
                'consistency': 0.2,
                'role_suitability': 0.2,
                'matchup': 0.15,
                'venue': 0.15
            }
            
            return NeuralPrediction(
                expected_points=expected_points,
                confidence_score=confidence_score,
                performance_range=performance_range,
                feature_importance=feature_importance,
                uncertainty=uncertainty
            )
    
    def batch_predict(self, players_data: List[Dict[str, Any]], 
                     match_context: Dict[str, Any] = None) -> List[NeuralPrediction]:
        """Make predictions for multiple players"""
        
        predictions = []
        for player_data in players_data:
            prediction = self.predict(player_data, match_context)
            predictions.append(prediction)
        
        return predictions
    
    def _save_model(self):
        """Save trained model"""
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'neural_model.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_sequence': self.scaler_sequence,
            'scaler_static': self.scaler_static
        }, model_path)
    
    def load_model(self):
        """Load pre-trained model"""
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'neural_model.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            self.model = NeuralPerformancePredictor().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load scalers
            self.scaler_sequence = checkpoint['scaler_sequence']
            self.scaler_static = checkpoint['scaler_static']
            
            print("✅ Loaded pre-trained neural network model")
            return True
        
        return False
    
    def generate_synthetic_training_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Generate synthetic training data for initial model training"""
        
        np.random.seed(42)
        
        synthetic_players = []
        
        for i in range(n_samples):
            # Generate synthetic player data
            role = np.random.choice(['batsman', 'bowler', 'allrounder', 'wicketkeeper'], 
                                  p=[0.4, 0.3, 0.2, 0.1])
            
            # Generate performance based on role
            if role == 'batsman':
                base_performance = np.random.gamma(3, 15)
            elif role == 'bowler':
                base_performance = np.random.gamma(2, 18)
            elif role == 'allrounder':
                base_performance = np.random.gamma(2.5, 20)
            else:  # wicketkeeper
                base_performance = np.random.gamma(2.8, 17)
            
            # Recent form
            recent_form = []
            for j in range(10):
                match_performance = max(0, base_performance + np.random.normal(0, 8))
                recent_form.append({
                    'runs': int(np.random.poisson(match_performance * 0.6)) if role in ['batsman', 'allrounder', 'wicketkeeper'] else int(np.random.poisson(5)),
                    'wickets': int(np.random.poisson(match_performance * 0.08)) if role in ['bowler', 'allrounder'] else 0,
                    'catches': int(np.random.poisson(1)) if role == 'wicketkeeper' else int(np.random.poisson(0.3)),
                    'fantasy_points': match_performance,
                    'strike_rate': np.random.normal(125, 20),
                    'bowling_figures_runs': np.random.normal(25, 10),
                    'economy_rate': np.random.normal(7, 1.5),
                    'batting_position': np.random.randint(1, 11)
                })
            
            player_data = {
                'player_id': i,
                'role': role,
                'recent_form': recent_form,
                'ema_score': base_performance + np.random.normal(0, 5),
                'consistency_score': np.random.beta(3, 2) * 100,
                'performance_rating': base_performance + np.random.normal(0, 10),
                'form_momentum': np.random.normal(0, 0.3),
                'dynamic_opportunity_index': np.random.beta(2, 2),
                'batting_stats': {
                    'average': np.random.gamma(2, 15) if role in ['batsman', 'allrounder', 'wicketkeeper'] else np.random.gamma(1, 8),
                    'strikeRate': np.random.normal(125, 25)
                },
                'bowling_stats': {
                    'average': np.random.gamma(2, 12) if role in ['bowler', 'allrounder'] else 50,
                    'economy': np.random.normal(7, 1.5),
                    'wickets': np.random.poisson(30) if role in ['bowler', 'allrounder'] else 0
                },
                'captain_vice_captain_probability': np.random.beta(1, 4) * 100,
                'injury_risk': np.random.beta(1, 9),
                'fatigue_index': np.random.beta(2, 8),
                'psychological_state': np.random.beta(3, 2),
                'expected_points': base_performance
            }
            
            synthetic_players.append(player_data)
        
        return self.prepare_data(synthetic_players)

# Export
__all__ = ['NeuralPredictionEngine', 'NeuralPrediction']