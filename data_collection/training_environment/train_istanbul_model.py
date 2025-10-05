"""
Istanbul Tourism Model Training Script
"""

import os
import json
import torch
import wandb
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IstanbulModelTrainer:
    def __init__(self, config_dir: str = "../istanbul_model_config"):
        self.config_dir = Path(config_dir)
        self.load_configs()
        
    def load_configs(self):
        """Load configurations"""
        with open(self.config_dir / "training_config.json", 'r') as f:
            self.training_config = json.load(f)
        
        self.model_config = GPT2Config.from_pretrained(self.config_dir)
        logger.info(f"Loaded configurations from {self.config_dir}")
    
    def prepare_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        tokenizer_path = "../istanbul_tokenizer"
        if os.path.exists(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            logger.warning("Custom tokenizer not found, using base GPT-2")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2-medium',
            config=self.model_config
        )
        
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized model embeddings to {len(self.tokenizer)}")
    
    def load_dataset(self, data_path: str = "../data/training"):
        """Load training dataset"""
        data_path = Path(data_path)
        
        if (data_path / "conversation_data.jsonl").exists():
            dataset = load_dataset('json', data_files=str(data_path / "conversation_data.jsonl"))['train']
        else:
            logger.error(f"Training data not found at {data_path}")
            raise FileNotFoundError("Training data not found")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.training_config['max_length']
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval")
    
    def setup_training_arguments(self, output_dir: str = "./istanbul_model_output"):
        """Setup training arguments"""
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            max_grad_norm=self.training_config['max_grad_norm'],
            warmup_steps=self.training_config['warmup_steps'],
            max_steps=self.training_config['max_steps'],
            logging_steps=100,
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.training_config['fp16'],
            gradient_checkpointing=self.training_config['gradient_checkpointing'],
            dataloader_drop_last=self.training_config['dataloader_drop_last'],
            run_name="istanbul-tourism-model",
            report_to="wandb" if wandb.api.api_key else None,
            seed=self.training_config['seed'],
            data_seed=self.training_config['data_seed']
        )
    
    def train(self):
        """Run training"""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting training...")
        train_result = trainer.train()
        
        trainer.save_model()
        trainer.save_state()
        
        logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
        return train_result

def main():
    parser = argparse.ArgumentParser(description="Train Istanbul Tourism Model")
    parser.add_argument("--config_dir", default="../istanbul_model_config")
    parser.add_argument("--data_path", default="../data/training")
    parser.add_argument("--output_dir", default="./istanbul_model_output")
    parser.add_argument("--wandb_project", default="istanbul-tourism-model")
    
    args = parser.parse_args()
    
    if wandb.api.api_key:
        wandb.init(project=args.wandb_project)
    
    trainer = IstanbulModelTrainer(args.config_dir)
    trainer.prepare_model_and_tokenizer()
    trainer.load_dataset(args.data_path)
    trainer.setup_training_arguments(args.output_dir)
    
    result = trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
