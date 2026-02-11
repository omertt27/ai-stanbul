"""
Phase 2: Deep Learning Models for AI Istanbul

This package contains deep learning implementations for personalized recommendations:
- Neural Collaborative Filtering (NCF)
- Wide & Deep Learning
- BERT4Rec (Transformer-based sequential recommendations)
- Model Ensemble and Smart Routing

Architecture:
    User Interaction → Event Collection → Deep Learning Pipeline → Recommendations
                                          ├─ NCF (user-item pairs)
                                          ├─ Wide & Deep (features)
                                          ├─ BERT4Rec (sequences)
                                          └─ Ensemble Router

Author: AI Istanbul Team
Date: February 10, 2026
"""

__version__ = "2.0.0"
__all__ = [
    "BaseModel",
    "NCF",
    "WideAndDeep",
    "BERT4Rec",
    "ModelEnsemble",
    "ModelRouter",
]
