import os
import openai

# Load your API key
openai.api_key = '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'

ip = input('enter your choice:')
if ip == "bot":
    def chatbot():
        # Create a list to store all the messages for context
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        print("Start chatting with the bot (type 'quit' to stop)!")

        while True:
            # user input
            message = input("User: ")

            if message.lower() == "quit":
                break

            messages.append({"role": "user", "content": message})

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            chat_message = response['choices'][0]['message']['content']
            print(f"Bot: {chat_message}")
            messages.append({"role": "assistant", "content": chat_message})

    chatbot()

if ip == "questions":
    import openai

    api_key = "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"

    def chat_with_bot():
        # Create a list to store all the messages for context
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        print("Start chatting with the bot (type 'quit' to stop)")

        job_title = input('Enter job title: ')

        for _ in range(5):
            messages.append({"role": "user", "content": f"User: generate a question on {job_title}."})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                api_key=api_key
            )

            # Get the question from the model and ask the user to answer
            question = response['choices'][0]['message']['content']
            print(f"Bot: {question}")
            messages.append({"role": "assistant", "content": question})

            user_response = input("User: ")
            if user_response.lower() == "quit":
                break
            messages.append({"role": "user", "content": user_response})

            rating_prompt = "Evaluate the user's answer based on your knowledge on a scale from 1 to 5: "
            messages.append({"role": "assistant", "content": rating_prompt})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                api_key=api_key  # Include your API key here
            )

            bot_rating = response['choices'][0]['message']['content']
            print(f"Bot's Rating: {bot_rating}")

        print("Conversation completed")

    chat_with_bot()
