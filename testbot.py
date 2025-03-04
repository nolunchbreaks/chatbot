from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# Load BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

print("Recovery Bot: Hi! How can I support you today? (Type 'exit' to stop)")

while True:
    # Get user input
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Recovery Bot: Take care! Stay strong. ❤️")
        break

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate response
    response_ids = model.generate(**inputs)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    print("Recovery Bot:", response)

