#!/usr/bin/env python3
"""Analyze current training data distribution"""

import json
from collections import Counter

# Load training data
with open('comprehensive_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total samples: {len(data)}')
print(f'Format: {type(data[0])} - {data[0]}')

# Count intents
intents = Counter([item[1] for item in data])
print('\n=== Intent Distribution ===')
for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
    print(f'{intent:25s}: {count:4d} samples')

# Check for English samples
english_keywords = ['what', 'where', 'how', 'when', 'can', 'is', 'are', 'the', 'best', 'nearest']
english_count = 0
for query, intent in data:
    if any(keyword in query.lower() for keyword in english_keywords):
        english_count += 1

print(f'\n=== Language Distribution ===')
print(f'Estimated Turkish samples: {len(data) - english_count}')
print(f'Estimated English samples: {english_count}')
print(f'Turkish coverage: {((len(data) - english_count) / len(data) * 100):.1f}%')
