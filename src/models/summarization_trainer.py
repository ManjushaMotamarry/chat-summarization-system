"""
BART/T5 Summarization Model Trainer
Handles model training, evaluation, and MLflow tracking
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from rouge_score import rouge_scorer
import os
import sys
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SummarizationTrainer:
    """Trains and evaluates summarization models"""
    
    def __init__(self, config_path='configs/model_config.yaml'):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to model configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['device'])
        
        logger.info(f"Initialized SummarizationTrainer with device: {self.device}")
        logger.info(f"Model: {self.config['model']['name']}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_data(self):
        """Load train and validation datasets"""
        logger.info("Loading datasets...")
        
        train_df = pd.read_csv(self.config['data']['train_path'])
        val_df = pd.read_csv(self.config['data']['val_path'])
        
        logger.info(f"Loaded {len(train_df)} training examples")
        logger.info(f"Loaded {len(val_df)} validation examples")
        
        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(train_df[['dialogue', 'summary']])
        val_dataset = Dataset.from_pandas(val_df[['dialogue', 'summary']])
        
        return train_dataset, val_dataset
    
    def initialize_model(self):
        """Initialize tokenizer and model"""
        logger.info("Initializing model and tokenizer...")
        
        model_name = self.config['model']['name']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
        
    def preprocess_function(self, examples):
        """
        Tokenize inputs and targets for model training
        
        Args:
            examples: Batch of examples with 'dialogue' and 'summary'
            
        Returns:
            Tokenized inputs and labels
        """
        model_inputs = self.tokenizer(
            examples['dialogue'],
            max_length=self.config['model']['max_input_length'],
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            examples['summary'],
            max_length=self.config['model']['max_target_length'],
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        """
        Compute ROUGE metrics for evaluation
        
        Args:
            eval_pred: Predictions and labels from model
            
        Returns:
            Dictionary of metric scores
        """
        predictions, labels = eval_pred
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Ensure predictions are token IDs (convert to int if needed)
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        predictions = predictions.astype(np.int64)
        
        # Decode predictions
        try:
            decoded_preds = self.tokenizer.batch_decode(
                predictions, 
                skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"Error decoding predictions: {e}")
            # Return dummy metrics if decode fails
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = labels.astype(np.int64)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(label, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def train(self):
        """Main training loop with MLflow tracking"""
        logger.info("="*60)
        logger.info("Starting model training")
        logger.info("="*60)
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'model_name': self.config['model']['name'],
                'batch_size': self.config['model']['batch_size'],
                'learning_rate': self.config['model']['learning_rate'],
                'num_epochs': self.config['model']['num_epochs'],
                'max_input_length': self.config['model']['max_input_length'],
                'max_target_length': self.config['model']['max_target_length']
            })
            
            # Load data
            train_dataset, val_dataset = self.load_data()
            
            # Initialize model
            self.initialize_model()
            
            # Tokenize datasets
            logger.info("Tokenizing datasets...")
            tokenized_train = train_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=['dialogue', 'summary']
            )
            tokenized_val = val_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=['dialogue', 'summary']
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model
            )
            
            # Training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.config['model']['output_dir'],
                eval_strategy="steps",
                eval_steps=self.config['model']['eval_steps'],
                save_steps=self.config['model']['save_steps'],
                logging_steps=self.config['model']['logging_steps'],
                learning_rate=self.config['model']['learning_rate'],
                per_device_train_batch_size=self.config['model']['batch_size'],
                per_device_eval_batch_size=self.config['model']['batch_size'],
                num_train_epochs=self.config['model']['num_epochs'],
                weight_decay=self.config['model']['weight_decay'],
                warmup_steps=self.config['model']['warmup_steps'],
                predict_with_generate=True,
                fp16=False,  # Set to True if using GPU with fp16 support
                push_to_hub=False,
                load_best_model_at_end=False,
                # metric_for_best_model='rougeL',
                # greater_is_better=True,
                save_total_limit=2
            )
            
            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )
            
            # Train
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_result.training_loss,
                'train_runtime': train_result.metrics['train_runtime'],
                'train_samples_per_second': train_result.metrics['train_samples_per_second']
            })
            
            # Evaluate
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate()
            
            mlflow.log_metrics({
                'eval_rouge1': eval_results['eval_rouge1'],
                'eval_rouge2': eval_results['eval_rouge2'],
                'eval_rougeL': eval_results['eval_rougeL']
            })
            
            # Save best model
            logger.info(f"Saving best model to {self.config['model']['best_model_dir']}")
            trainer.save_model(self.config['model']['best_model_dir'])
            self.tokenizer.save_pretrained(self.config['model']['best_model_dir'])
            
            # Log model to MLflow
            mlflow.pytorch.log_model(self.model, "model")
            
            logger.info("="*60)
            logger.info("Training complete!")
            logger.info(f"Best ROUGE-L: {eval_results['eval_rougeL']:.4f}")
            logger.info("="*60)
            
            return eval_results


if __name__ == "__main__":
    # Train model
    trainer = SummarizationTrainer()
    results = trainer.train()
    
    print("\n📊 Final Results:")
    print(f"ROUGE-1: {results['eval_rouge1']:.4f}")
    print(f"ROUGE-2: {results['eval_rouge2']:.4f}")
    print(f"ROUGE-L: {results['eval_rougeL']:.4f}")