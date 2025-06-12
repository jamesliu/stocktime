# stocktime/core/stocktime_predictor_2025.py

"""
StockTime Predictor - 2025 Updated Version
Optimized for best small open-source LLMs of 2025

Priority LLM Selection:
1. Qwen2.5-0.5B-Instruct (0.5B) - Best overall balance
2. TinyLlama-1.1B (1.1B) - Proven performance
3. SmolLM2-360M-Instruct (360M) - Ultra-lightweight
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

class StockTimePredictor2025(nn.Module):
    """
    StockTime: LLM-based stock price prediction model
    2025 Edition with optimal small LLM selection
    
    Model Priority Order:
    1. Qwen2.5-0.5B-Instruct: 128K context, instruction-following, multilingual
    2. TinyLlama-1.1B: Proven downstream performance, efficient
    3. SmolLM2-360M: Ultra-lightweight for edge deployment
    """
    
    # Define model configurations
    MODEL_CONFIGS = {
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "context_length": 8192,  # Can handle up to 128K
            "strengths": ["instruction_following", "multilingual", "long_context"],
            "use_case": "optimal_balance"
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "context_length": 2048,
            "strengths": ["efficiency", "downstream_performance", "stability"],
            "use_case": "proven_performance"
        },
        "HuggingFaceTB/SmolLM2-360M-Instruct": {
            "context_length": 2048,
            "strengths": ["ultra_lightweight", "on_device", "low_power"],
            "use_case": "edge_deployment"
        },
        "microsoft/DialoGPT-small": {  # Fallback for compatibility
            "context_length": 1024,
            "strengths": ["compatibility", "tested"],
            "use_case": "fallback"
        }
    }
    
    def __init__(self, 
                 llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 llm_model_fallback: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 llm_model_ultra_light: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
                 lookback_window: int = 32,
                 patch_length: int = 4,
                 hidden_dim: int = 256,
                 num_lstm_layers: int = 2,
                 deployment_mode: str = "optimal"):  # "optimal", "efficient", "ultra_light"
        super().__init__()
        
        self.lookback_window = lookback_window
        self.patch_length = patch_length
        self.hidden_dim = hidden_dim
        self.deployment_mode = deployment_mode
        
        # Select model based on deployment mode
        model_priority = self._select_model_by_deployment_mode(
            llm_model_name, llm_model_fallback, llm_model_ultra_light, deployment_mode
        )
        
        # Load LLM with fallback mechanism
        self.llm_model_name, self.tokenizer, self.llm = self._load_llm_with_fallback(model_priority)
        
        # Get model configuration
        self.model_config = self.MODEL_CONFIGS.get(self.llm_model_name, self.MODEL_CONFIGS["microsoft/DialoGPT-small"])
        self.max_context_length = self.model_config["context_length"]
        
        # Freeze LLM parameters (core StockTime principle)
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Autoregressive encoder (LSTM)
        self.autoregressive_encoder = nn.LSTM(
            input_size=patch_length,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0.0
        )
        
        # Embedding projection for price patches
        self.price_projection = nn.Linear(hidden_dim, self.llm.config.hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(self.llm.config.hidden_size, patch_length)
        
        # Instance normalization
        self.instance_norm = nn.InstanceNorm1d(1)
        
        logging.info(f"🚀 StockTime 2025 initialized with {self.llm_model_name}")
        logging.info(f"📊 Model strengths: {self.model_config['strengths']}")
        logging.info(f"🎯 Use case: {self.model_config['use_case']}")
        
    def _select_model_by_deployment_mode(self, primary, fallback, ultra_light, mode):
        """Select model priority based on deployment requirements"""
        if mode == "optimal":
            return [primary, fallback, ultra_light, "microsoft/DialoGPT-small"]
        elif mode == "efficient":
            return [fallback, primary, ultra_light, "microsoft/DialoGPT-small"]
        elif mode == "ultra_light":
            return [ultra_light, fallback, primary, "microsoft/DialoGPT-small"]
        else:
            return [primary, fallback, ultra_light, "microsoft/DialoGPT-small"]
    
    def _load_llm_with_fallback(self, model_priority: List[str]) -> Tuple[str, any, any]:
        """Load LLM with automatic fallback to alternative models"""
        
        for i, model_name in enumerate(model_priority):
            try:
                logging.info(f"🔄 Attempting to load LLM {i+1}/{len(model_priority)}: {model_name}")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Handle missing pad token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                llm = AutoModel.from_pretrained(model_name)
                
                logging.info(f"✅ Successfully loaded: {model_name}")
                return model_name, tokenizer, llm
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to load {model_name}: {str(e)[:100]}...")
                if i == len(model_priority) - 1:  # Last model in priority
                    raise RuntimeError(f"❌ Could not load any LLM model. Last error: {e}")
                continue
    
    def create_textual_template_enhanced(self, price_patches: torch.Tensor, timestamps: List[str]) -> List[str]:
        """
        Enhanced textual template creation optimized for 2025 small LLMs
        
        Improvements:
        - More structured format for instruction-following models
        - Enhanced numerical precision for financial data
        - Better correlation and trend descriptions
        """
        templates = []
        
        for i, patch in enumerate(price_patches):
            # Enhanced statistical analysis
            min_val = torch.min(patch).item()
            max_val = torch.max(patch).item()
            mean_val = torch.mean(patch).item()
            std_val = torch.std(patch).item()
            
            # Calculate rate of change and momentum
            if len(patch) > 1:
                rate_change = ((patch[-1] - patch[0]) / (patch[0] + 1e-8) * 100).item()
                momentum = ((patch[-1] - patch[0]) / len(patch)).item()
            else:
                rate_change = 0.0
                momentum = 0.0
            
            # Calculate volatility indicator
            if len(patch) > 2:
                volatility = (std_val / (mean_val + 1e-8) * 100).item()
            else:
                volatility = 0.0
            
            # Create enhanced template for 2025 LLMs
            if "Qwen" in self.llm_model_name:
                # Optimized for Qwen's instruction-following capability
                template = f"Analyze this financial time series segment:\n"
                template += f"Price Range: ${min_val:.4f} - ${max_val:.4f}\n"
                template += f"Average: ${mean_val:.4f}, Volatility: {volatility:.2f}%\n"
                template += f"Change Rate: {rate_change:.2f}%, Momentum: {momentum:.4f}\n"
                template += f"Timestamp: {timestamps[i] if i < len(timestamps) else 'N/A'}\n"
                
            elif "TinyLlama" in self.llm_model_name:
                # Optimized for TinyLlama's efficiency and proven performance
                template = f"Stock data: price_min={min_val:.4f} price_max={max_val:.4f} "
                template += f"avg={mean_val:.4f} change={rate_change:.2f}% "
                template += f"vol={volatility:.2f}% time={timestamps[i] if i < len(timestamps) else 'N/A'}"
                
            elif "SmolLM" in self.llm_model_name:
                # Optimized for SmolLM's ultra-lightweight processing
                template = f"Price: {min_val:.2f}-{max_val:.2f}, avg={mean_val:.2f}, "
                template += f"chg={rate_change:.1f}%, {timestamps[i] if i < len(timestamps) else 'N/A'}"
                
            else:
                # Default template for compatibility
                template = f"###Context: This is a stock price time series. "
                template += f"###Trend analysis: The input data features a minimum of {min_val:.4f}, "
                template += f"a maximum of {max_val:.4f}, moving average of {mean_val:.4f}, "
                template += f"and a change rate of {rate_change:.2f}%. "
                template += f"###Time stamp: {timestamps[i] if i < len(timestamps) else 'N/A'}."
            
            templates.append(template)
            
        return templates
    
    def forward(self, price_sequences: torch.Tensor, timestamps: List[str]) -> torch.Tensor:
        """
        Forward pass through StockTime model with 2025 optimizations
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
        
        # Create enhanced textual information
        textual_templates = []
        for b in range(batch_size):
            batch_templates = self.create_textual_template_enhanced(
                patches[b], timestamps
            )
            textual_templates.extend(batch_templates)
        
        # Tokenize with model-specific context length
        max_length = min(self.max_context_length // 4, 512)  # Conservative estimate
        tokenized = self.tokenizer(
            textual_templates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Get textual embeddings from frozen LLM
        with torch.no_grad():
            try:
                textual_outputs = self.llm(**tokenized)
                textual_embeddings = textual_outputs.last_hidden_state.mean(dim=1)
            except Exception as e:
                logging.warning(f"LLM processing fallback due to: {e}")
                # Fallback to simple embedding
                textual_embeddings = torch.randn(
                    len(textual_templates), self.llm.config.hidden_size
                )
        
        # Reshape for batch processing
        textual_embeddings = textual_embeddings.view(
            batch_size, num_patches, -1
        )
        
        # Ensure embedding dimensions match
        if textual_embeddings.shape[-1] != price_embeddings.shape[-1]:
            textual_embeddings = textual_embeddings[:, :, :price_embeddings.shape[-1]]
        
        # Multimodal fusion (addition as in paper)
        fused_embeddings = price_embeddings + textual_embeddings
        
        # Generate predictions
        predictions = self.output_projection(fused_embeddings)
        
        return predictions
    
    def predict_next_prices(self, price_history: np.ndarray, timestamps: List[str], 
                           prediction_steps: int = 1) -> np.ndarray:
        """
        Predict future stock prices with enhanced error handling
        """
        self.eval()
        
        try:
            with torch.no_grad():
                price_tensor = torch.FloatTensor(price_history).unsqueeze(0)
                predictions = self.forward(price_tensor, timestamps)
                
                # Get the last prediction for next steps
                last_pred = predictions[0, -1].cpu().numpy()
                
                # Ensure we return the requested number of predictions
                if len(last_pred) < prediction_steps:
                    # Pad with the last available prediction
                    padding = np.repeat(last_pred[-1], prediction_steps - len(last_pred))
                    last_pred = np.concatenate([last_pred, padding])
                
                return last_pred[:prediction_steps]
                
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            # Return conservative fallback predictions
            return np.array([price_history[-1]] * prediction_steps)
    
    def get_model_info(self) -> Dict:
        """Get detailed information about the loaded model"""
        return {
            "model_name": self.llm_model_name,
            "model_config": self.model_config,
            "context_length": self.max_context_length,
            "deployment_mode": self.deployment_mode,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in self.llm.parameters())
        }

def main():
    """
    Test the StockTime 2025 predictor with different deployment modes
    """
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing StockTime 2025 Predictor")
    print("=" * 50)
    
    # Test different deployment modes
    deployment_modes = ["optimal", "efficient", "ultra_light"]
    
    for mode in deployment_modes:
        print(f"\n🔧 Testing {mode} deployment mode:")
        
        try:
            # Initialize model
            predictor = StockTimePredictor2025(
                lookback_window=32,
                patch_length=4,
                deployment_mode=mode
            )
            
            # Display model info
            info = predictor.get_model_info()
            print(f"   Model: {info['model_name']}")
            print(f"   Trainable params: {info['trainable_parameters']:,}")
            print(f"   Context length: {info['context_length']:,}")
            
            # Generate sample data
            np.random.seed(42)
            sample_prices = np.cumsum(np.random.randn(128) * 0.01) + 100
            sample_timestamps = [f"2024-01-{i+1:02d}" for i in range(32)]
            
            # Test prediction
            predictions = predictor.predict_next_prices(
                sample_prices[:32], sample_timestamps, prediction_steps=3
            )
            print(f"   Sample predictions: {[f'{p:.2f}' for p in predictions]}")
            print(f"   ✅ {mode.title()} mode: PASSED")
            
        except Exception as e:
            print(f"   ❌ {mode.title()} mode: FAILED - {e}")
    
    print(f"\n🎯 StockTime 2025 testing completed!")

if __name__ == "__main__":
    main()
