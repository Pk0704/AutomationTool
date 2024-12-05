import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_creative_content(client, prompt, model="gpt-4"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
            generated_content = response.choices[0].message.content.strip()
            print("Generated Content:", generated_content)
            return generated_content
        else:
            print("No content was generated. Response:", response)
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_user_feedback(content):
    print("Please rate the content from 1 (poor) to 5 (excellent):")
    rating = input("Rating: ")
    return int(rating)

def extract_features(vectorizer, data):
    return vectorizer.transform(data)  # CHANGED: We will fit once initially, then transform

def save_feedback_to_csv(prompts, contents, ratings, filename='feedback.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for prompt, content, rating in zip(prompts, contents, ratings):
            writer.writerow([prompt, content, rating])

def calculate_average_rating(filename='feedback.csv'):
    ratings = []
    if not os.path.exists(filename):
        return 0
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                rating = int(row[2])
                ratings.append(rating)
            except ValueError:
                continue
    return sum(ratings) / len(ratings) if ratings else 0

def get_user_preferences():
    while True:
        print("What type of content would you like? (S) Story, (A) Article, (J) Journal, (P) Poem, (O) Other:")
        content_type = input().strip().upper()
        if content_type in ['S', 'A', 'P', 'J', 'O']:
            break
        else:
            print("Invalid input. Please enter S, J, A, P, or O.")

    if content_type in ['S', 'P']:
        style = input("What style would you like it in? You can say anything from Shakespeare to Old American folk! ").strip()
    elif content_type in ['A', 'J', 'O']:
        style = input("What would you like to know more about? ").strip()
    return content_type, style

def create_prompt(content_type, style):
    if content_type == 'S':
        return f"Write a story in the style of {style}."
    elif content_type == 'A':
        return f"Write an article about {style}."
    elif content_type == 'P':
        return f"Write a poem in the style of {style}."
    elif content_type == 'O':
        return f"Write a piece of literature about {style}."
    elif content_type == 'J':
        return f"Write a journal about {style}."
    else:
        return ""

def get_reward(rating):
    if rating == 5:
        return 2
    elif rating == 4:
        return 1
    elif rating == 3:
        return 0
    elif rating == 2:
        return -1
    else:
        return -2

# ADDED: A function to generate candidate prompts and choose the best one
def generate_prompt_variations(content_type, style):
    # Just a few naive variations. You could add more complexity here.
    base_prompt = create_prompt(content_type, style)
    variations = [
        base_prompt,
        base_prompt + " Please provide detailed descriptions.",
        base_prompt + " Make it more whimsical and imaginative.",
        base_prompt + " Focus on character development and setting.",
        base_prompt + " Keep the language simple and accessible."
    ]
    return variations

# ADDED: Function to use Q-network to pick best prompt
def choose_best_prompt(variations, q_network, vectorizer):
    # We assume q_network and vectorizer are already trained/fitted at least once.
    # We'll predict Q-values for each variation and choose the one with the best expected rating.
    highest_q = -float('inf')
    best_prompt = variations[0]

    # Extract features for all variations at once
    X = vectorizer.transform(variations).toarray()
    X = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = q_network(X)
        # The Q-network outputs a vector of 5 values (one for each rating).
        # We'll consider the expected rating as a weighted sum or just the max Q-value.
        # For simplicity, we'll pick the prompt whose max Q-value (i.e., best rating prediction) is highest.
        for i, output in enumerate(outputs):
            max_q = torch.max(output).item()
            if max_q > highest_q:
                highest_q = max_q
                best_prompt = variations[i]

    return best_prompt


def main():
    api_key = os.getenv('OPENAI_API_KEY', 'YourOwnPrivateApi')
    client = OpenAI(api_key=api_key)

    contents = []
    prompts = []
    ratings = []

    vectorizer = TfidfVectorizer()
    
    # Instead of fitting vectorizer every time, we'll do an initial fit with an empty set and update incrementally
    # Actually TF-IDF needs some corpus to fit on. We'll start empty and refit as we go if needed.
    # A simple approach: We'll store all generated texts and refit after each iteration.
    all_generated_texts = []

    # Initialize Q-Network with a placeholder size. We will set this dynamically after first generation.
    input_size = 1000  # arbitrary until we know the real size
    output_size = 5
    q_network = QNetwork(input_size, output_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    first_iteration = True

    while True:
        content_type, style = get_user_preferences()
        
        # If we have trained the Q-network at least once, use it to choose the best prompt
        if not first_iteration and len(all_generated_texts) > 0:
            # Refit vectorizer on all generated texts so it knows all vocabulary
            vectorizer.fit(all_generated_texts)

            prompt_variations = generate_prompt_variations(content_type, style)
            # Choose the best prompt based on Q-network predictions
            best_prompt = choose_best_prompt(prompt_variations, q_network, vectorizer)
            prompt = best_prompt
        else:
            # On the first iteration, we have no Q-network training, just use the basic prompt
            prompt = create_prompt(content_type, style)

        generated_content = generate_creative_content(client, prompt)

        if generated_content:
            contents.append(generated_content)
            prompts.append(prompt)
            
            # Collect user feedback
            rating = get_user_feedback(generated_content)
            ratings.append(rating)

            # Update our corpus and refit vectorizer to include new text
            all_generated_texts.append(generated_content)
            vectorizer.fit(all_generated_texts)  # Update vocabulary with new content
            
            # Extract features from the newly generated content
            X = extract_features(vectorizer, [generated_content]).toarray()
            X = torch.tensor(X, dtype=torch.float32)

            # If first iteration, we might need to re-initialize Q-network with correct input size
            if first_iteration:
                input_size = X.shape[1]
                q_network = QNetwork(input_size, output_size)
                optimizer = optim.Adam(q_network.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                first_iteration = False

            # Prepare target
            target = torch.zeros(output_size)
            reward = get_reward(rating)
            target[rating - 1] = reward

            # Train Q-network
            q_network.train()
            output = q_network(X)
            loss = criterion(output.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        else:
            print("Failed to generate content.")
        
        another = input("Would you like to generate another piece of content? (yes/no): ").strip().lower()
        if another != 'yes':
            break

    # Save feedback to CSV
    save_feedback_to_csv(prompts, contents, ratings)

    # Calculate and display average rating
    average_rating = calculate_average_rating()
    print(f"Current average user satisfaction rating: {average_rating:.2f}")

if __name__ == "__main__":
    main()
