from intent_utils import parse_user_input
import json

test_queries = [
    "Find vegan restaurants in Paris tomorrow",
    "Tell me about the Louvre",
    "How much is a ticket to the Colosseum?",
    "What concerts are happening in New York this weekend?"
]

for q in test_queries:
    print("USER:", q)
    parsed = parse_user_input(q)
    print("BOT (parsed):", parsed)
    print("------")
