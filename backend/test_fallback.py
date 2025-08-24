from intent_utils import parse_user_input
import json

test_queries = [
    "Blah blah blah",
    "Find a restaurant"
]

for q in test_queries:
    print("USER:", q)
    parsed = parse_user_input(q)
    print("BOT (parsed):", parsed)
    try:
        parsed_json = json.loads(parsed)
        intent = parsed_json.get("intent", "")
        entities = parsed_json.get("entities", {})
        if not intent or intent == "unknown":
            print("BOT: Sorry, I didn’t quite get that. Can you rephrase?")
        elif intent in ["find_restaurants", "restaurant_search"] and "location" not in entities:
            print("BOT: Sure! Which city are you interested in?")
        elif intent in ["inquire_about_museums", "museum_info"] and "location" not in entities:
            print("BOT: Which museum or city do you mean?")
        elif intent in ["find_concerts", "event_lookup"] and ("location" not in entities or "date" not in entities):
            print("BOT: Can you specify the city or date for the event?")
        elif intent in ["inquire_ticket_price", "ticket_prices"] and "attraction" not in entities:
            print("BOT: Which attraction or event do you want the ticket price for?")
    except Exception:
        print("BOT: Sorry, I didn’t quite get that. Can you rephrase?")
    print("------")
