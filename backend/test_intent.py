from intent_utils import parse_user_input

if __name__ == "__main__":
    user_input = input("Enter a user query: ")
    result = parse_user_input(user_input)
    print("Intent and entities:")
    print(result)
