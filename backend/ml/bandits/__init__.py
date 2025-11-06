"""
Bandit Algorithms for Exploration-Exploitation Trade-off
Implements contextual bandits for personalized recommendations
"""

from .contextual_thompson_sampling import ContextualThompsonSampling, BanditContext

__all__ = [
    'ContextualThompsonSampling',
    'BanditContext'
]
