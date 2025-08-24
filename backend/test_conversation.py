from intent_utils import parse_user_input
import json

# Simple context tracker
def update_context(context, parsed):
    # Update context with any new entities
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            return context
    entities = parsed.get("entities", {})
    for k, v in entities.items():
        context[k] = v
    return context

test_conversation = [
    "Find Italian restaurants in Rome",
    "And what about museums nearby?"
]

context = {}
for q in test_conversation:
    print("USER:", q)
    parsed = parse_user_input(q)
    print("BOT (parsed):", parsed)
    context = update_context(context, parsed)
    print("CONTEXT:", context)
    print("------")

# Simulate using context for follow-up
if "location" in context:
    print(f"Bot would use location '{context['location']}' for follow-up API calls.")
else:
    print("No location found in context.")
