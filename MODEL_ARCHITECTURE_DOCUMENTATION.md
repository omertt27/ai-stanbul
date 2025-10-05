# Istanbul Tourism Domain-Specific Model Architecture Documentation

## Overview

This document outlines the complete model architecture, training setup, and deployment configuration for the Istanbul Tourism Domain-Specific AI Assistant.

## Model Architecture

### Base Model: GPT-2 Medium (355M Parameters)

**Key Specifications:**
- **Architecture**: GPT-2 Medium
- **Parameters**: 355 million
- **Context Length**: 2,048 tokens
- **Vocabulary Size**: 50,400 (extended from base 50,257)
- **Hidden Size**: 1,024
- **Layers**: 24
- **Attention Heads**: 16
- **Intermediate Size**: 4,096

### Domain-Specific Optimizations

#### 1. Extended Vocabulary (91 Istanbul-specific terms)

**Districts and Areas:**
- sultanahmet, beyoğlu, galata, karaköy, beşiktaş, ortaköy, taksim, kadıköy, üsküdar, fatih, eminönü, bakırköy, şişli, pendik, maltepe

**Transportation:**
- metro, metrobüs, dolmuş, vapur, tramvay, marmaray, istanbulkart, otobüs, taksi, uber, bilet, aktarma, durak, istasyon, hattı

**Attractions:**
- ayasofya, sultanahmet, topkapı, kapalıçarşı, galata kulesi, boğaz, büyük mecidiye, dolmabahçe, basilica cistern, hagia sophia, blue mosque, grand bazaar, spice bazaar, bosphorus

**Food and Dining:**
- kebap, döner, lahmacun, pide, simit, çay, kahve, baklava, lokum, meze, rakı, balık ekmek, kokoreç, midye dolma, künefe, mantı, börek, çorba, pilav

**Cultural Terms:**
- hamam, cami, müze, saray, köprü, medrese, türbe, han, bedesten, çeşme, Ottoman, Byzantine, Turkish, Islamic

**Tourism Terms:**
- turist, rehber, tur, bilet, müze kart, otel, pansiyon, hostel, rezervasyon, tatil, gezi, fotoğraf, alışveriş, hediyelik

#### 2. Special Tokens for Conversation Flow

```json
{
  "pad_token": "<|pad|>",
  "bos_token": "<|startoftext|>",
  "eos_token": "<|endoftext|>",
  "unk_token": "<|unk|>",
  "user_token": "<|user|>",
  "assistant_token": "<|assistant|>",
  "context_token": "<|context|>",
  "location_token": "<|location|>",
  "time_token": "<|time|>",
  "preference_token": "<|preference|>"
}
```

## Training Configuration

### Hardware Requirements

| Hardware Tier | GPU Memory | Training Mode | Batch Size | Est. Time |
|---------------|------------|---------------|------------|-----------|
| High-end | 24GB+ (RTX 4090, A100) | Full precision + gradient checkpointing | 4-8 | 6-12 hours |
| Mid-range | 12-24GB (RTX 3090, 4080) | Mixed precision (fp16) | 2-4 | 12-24 hours |
| Budget | 8-12GB (RTX 3070, 4060 Ti) | LoRA fine-tuning + quantization | 1-2 | 1-2 days |
| CPU Only | No GPU | CPU training (very slow) | 1-2 | 5-10 days |

### Training Parameters

```json
{
  "model_name": "istanbul-tourism-gpt2-medium",
  "max_length": 2048,
  "learning_rate": 2e-05,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 500,
  "max_steps": 10000,
  "eval_steps": 500,
  "save_steps": 1000,
  "fp16": true,
  "gradient_checkpointing": true,
  "weight_decay": 0.01,
  "lr_scheduler_type": "cosine",
  "optim": "adamw_torch"
}
```

### Memory Requirements

**Model Sizes:**
- **FP16**: 710 MB
- **4-bit Quantized**: 178 MB

**Training Memory:**
- **Minimum**: 12-16 GB GPU memory
- **Recommended**: 24 GB+ for optimal performance

**Inference Memory:**
- **FP16**: 1-2 GB GPU memory
- **4-bit**: 512 MB GPU memory
- **CPU**: 2-4 GB RAM (slower)

## Quantization Configuration

### GPTQ Quantization Settings

```json
{
  "quantization_method": "GPTQ",
  "bits": 4,
  "group_size": 128,
  "desc_act": false,
  "disable_exllama": false,
  "model_seqlen": 2048,
  "cache_examples_on_gpu": true,
  "use_triton": true,
  "warmup_autotune": true,
  "fuse_layers": true
}
```

## Training Data Pipeline

### Data Sources
1. **Curated Istanbul Tourism Content**
2. **Multi-language Support** (Turkish/English)
3. **Conversation-style Training Examples**
4. **Domain-specific Q&A Pairs**

### Data Formats Generated
- **Q&A Format**: 24 examples per source
- **Conversation Format**: 24 examples per source  
- **Instruction Format**: 24 examples per source
- **Total**: 72 training examples per curated source

### Quality Metrics
- **Curation Retention Rate**: 75% (12/16 sample records passed quality filters)
- **Relevance Scoring**: Content scored on Istanbul tourism relevance
- **Language Quality**: Turkish/English text quality validation
- **Duplicate Detection**: Advanced deduplication based on semantic similarity

## Knowledge Distillation Setup

### Configuration

```json
{
  "teacher_model": "gpt-3.5-turbo",
  "student_model": "istanbul-tourism-gpt2-medium",
  "distillation_method": "response_distillation",
  "temperature": 3.0,
  "alpha": 0.7,
  "beta": 0.3,
  "max_examples": 50000,
  "topics": [
    "istanbul_attractions",
    "transportation", 
    "food_dining",
    "hotels_accommodation",
    "cultural_sites",
    "shopping",
    "nightlife",
    "day_trips",
    "practical_info",
    "history_culture"
  ]
}
```

## Deployment Architecture

### Model Serving Options

1. **High-Performance Deployment**
   - FP16 model on GPU
   - 1-2 GB GPU memory
   - ~50-100ms response time

2. **Cost-Optimized Deployment**
   - 4-bit quantized model
   - 512 MB GPU memory  
   - ~100-200ms response time

3. **CPU Deployment**
   - Quantized model on CPU
   - 2-4 GB RAM
   - ~500ms-2s response time

### Integration with Hybrid System

The domain-specific model serves as the **Tier 3** component in the hybrid architecture:

```
User Query → Rule-based Router → Semantic Cache → Domain LLM → Response
```

**Smart Routing Logic:**
- Simple questions (hours, location) → Rule-based
- Cached queries → Semantic cache  
- Complex/personalized queries → Domain LLM
- Fallback queries → External API (if needed)

## Performance Expectations

### Accuracy Targets
- **Domain Coverage**: 90%+ of Istanbul tourism queries
- **Response Quality**: High relevance for Turkish/English tourism content
- **Multi-language**: Seamless Turkish-English conversation support

### Cost Efficiency
- **Training Cost**: One-time setup (~$50-200 depending on hardware)
- **Inference Cost**: ~$0.001-0.01 per query (vs $0.02-0.06 for GPT-3.5/4)
- **Scalability**: 40K+ queries/month with minimal infrastructure

### Response Times  
- **GPU Inference**: 50-200ms
- **CPU Inference**: 500ms-2s
- **With Caching**: 10-50ms for cached responses

## Evaluation Metrics

### Automated Metrics
- **ROUGE-1/2/L**: Text similarity and overlap
- **BLEU Score**: Translation quality (Turkish-English)
- **Perplexity**: Language modeling quality
- **Exact Match**: Factual accuracy for known answers

### Domain-Specific Metrics
- **Istanbul Knowledge Accuracy**: Correct information about attractions, transportation, etc.
- **Cultural Sensitivity**: Appropriate cultural context and recommendations
- **Practical Utility**: Actionable travel advice and information

## Next Steps for Implementation

### Phase 1: Model Training (Week 3)
1. **Environment Setup**: Run `setup_environment.sh`
2. **Data Preparation**: Generate large-scale training dataset  
3. **Model Training**: Execute training pipeline
4. **Initial Evaluation**: Test model performance

### Phase 2: Optimization (Week 4)  
1. **Quantization**: Apply 4-bit quantization for deployment
2. **Performance Tuning**: Optimize inference speed and memory
3. **Integration Testing**: Connect with hybrid routing system
4. **Quality Assurance**: Comprehensive evaluation and testing

### Phase 3: Deployment (Week 5+)
1. **Production Setup**: Deploy model in production environment
2. **Monitoring**: Implement logging and performance monitoring  
3. **Continuous Improvement**: Collect user feedback and iterate
4. **Scaling**: Optimize for 40K+ monthly queries

## Files Created

### Model Configuration
- `istanbul_model_config/model_config.json`: Model architecture settings
- `istanbul_model_config/training_config.json`: Training hyperparameters  
- `istanbul_model_config/quantization_config.json`: Deployment optimization
- `istanbul_model_config/istanbul_vocabulary.json`: Domain-specific vocabulary
- `istanbul_model_config/special_tokens.json`: Conversation flow tokens

### Training Environment
- `training_environment/requirements.txt`: Python dependencies
- `training_environment/train_istanbul_model.py`: Main training script
- `training_environment/setup_environment.sh`: Environment setup
- `training_environment/distillation_config.json`: Knowledge distillation setup
- `training_environment/environment_summary.json`: System requirements summary

### Extended Tokenizer
- `istanbul_tokenizer/`: Extended GPT-2 tokenizer with Istanbul vocabulary

This architecture provides a solid foundation for creating a cost-efficient, accurate, and scalable Istanbul tourism AI assistant that can handle 40K+ queries per month while maintaining high quality and cultural relevance.
