# stocktime/core/stocktime_predictor.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

class StockTimePredictor(nn.Module):
    """
    StockTime: LLM-based stock price prediction model
    Based on research paper architecture
    """
    
    def __init__(self, 
                 llm_model_name: str = "microsoft/DialoGPT-small",
                 lookback_window: int = 32,
                 patch_length: int = 4,
                 hidden_dim: int = 256,
                 num_lstm_layers: int = 2):
        super().__init__()
        
        self.lookback_window = lookback_window
        self.patch_length = patch_length
        self.hidden_dim = hidden_dim
        
        # Load frozen LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModel.from_pretrained(llm_model_name)
        
        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Autoregressive encoder (LSTM)
        self.autoregressive_encoder = nn.LSTM(
            input_size=patch_length,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Embedding projection for price patches
        self.price_projection = nn.Linear(hidden_dim, self.llm.config.hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(self.llm.config.hidden_size, patch_length)
        
        # Instance normalization
        self.instance_norm = nn.InstanceNorm1d(1)
        
    def create_textual_template(self, price_patches: torch.Tensor, timestamps: List[str]) -> List[str]:
        """
        Extract textual information from stock price patches
        Following StockTime methodology
        """
        templates = []
        
        for i, patch in enumerate(price_patches):
            # Calculate statistics
            min_val = torch.min(patch).item()
            max_val = torch.max(patch).item()
            mean_val = torch.mean(patch).item()
            
            # Calculate rate of change
            if len(patch) > 1:
                rate_change = ((patch[-1] - patch[0]) / patch[0] * 100).item()
            else:
                rate_change = 0.0
                
            # Create template
            template = f"###Context: This is a stock price time series. "
            template += f"###Trend analysis: The input data features a minimum of {min_val:.4f}, "
            template += f"a maximum of {max_val:.4f}, moving average of {mean_val:.4f}, "
            template += f"and a change rate of {rate_change:.2f}%. "
            template += f"###Time stamp: {timestamps[i] if i < len(timestamps) else 'N/A'}."
            
            templates.append(template)
            
        return templates
    
    def forward(self, price_sequences: torch.Tensor, timestamps: List[str]) -> torch.Tensor:
        """
        Forward pass through StockTime model
        """
        batch_size, seq_len = price_sequences.shape
        
        # Instance normalization
        normalized_prices = self.instance_norm(price_sequences.unsqueeze(1)).squeeze(1)
        
        # Create patches
        num_patches = seq_len // self.patch_length
        patches = normalized_prices[:, :num_patches * self.patch_length].view(
            batch_size, num_patches, self.patch_length
        )
        
        # Autoregressive encoding
        lstm_out, _ = self.autoregressive_encoder(patches)
        price_embeddings = self.price_projection(lstm_out)
        
        # Create textual information
        textual_templates = []
        for b in range(batch_size):
            batch_templates = self.create_textual_template(
                patches[b], timestamps
            )
            textual_templates.extend(batch_templates)
        
        # Tokenize textual information
        tokenized = self.tokenizer(
            textual_templates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get textual embeddings from frozen LLM
        with torch.no_grad():
            textual_outputs = self.llm(**tokenized)
            textual_embeddings = textual_outputs.last_hidden_state.mean(dim=1)
        
        # Reshape for batch processing
        textual_embeddings = textual_embeddings.view(
            batch_size, num_patches, -1
        )
        
        # Multimodal fusion (addition as in paper)
        fused_embeddings = price_embeddings + textual_embeddings
        
        # Process through final layers (skip LLM processing for now to avoid compatibility issues)
        # In a full implementation, this would involve more sophisticated LLM integration
        processed_embeddings = fused_embeddings
        
        # Generate predictions
        predictions = self.output_projection(processed_embeddings)
        
        return predictions
    
    def predict_next_prices(self, price_history: np.ndarray, timestamps: List[str], 
                           prediction_steps: int = 1) -> np.ndarray:
        """
        Predict future stock prices
        """
        self.eval()
        
        with torch.no_grad():
            price_tensor = torch.FloatTensor(price_history).unsqueeze(0)
            predictions = self.forward(price_tensor, timestamps)
            
            # Get the last prediction for next steps
            last_pred = predictions[0, -1].cpu().numpy()
            
            # For multi-step prediction, we'd need to implement autoregressive generation
            return last_pred[:prediction_steps]

def main():
    """
    Test the StockTime predictor
    """
    # Initialize model
    predictor = StockTimePredictor(
        llm_model_name="microsoft/DialoGPT-small",  # Smaller model for testing
        lookback_window=32,
        patch_length=4
    )
    
    # Generate sample data
    np.random.seed(42)
    sample_prices = np.cumsum(np.random.randn(128) * 0.01) + 100
    sample_timestamps = [f"2024-01-{i+1:02d}" for i in range(32)]
    
    # Test prediction
    predictions = predictor.predict_next_prices(sample_prices, sample_timestamps)
    print(f"Sample predictions: {predictions}")
    
    logging.info("StockTime predictor initialized and tested successfully")

if __name__ == "__main__":
    main()
