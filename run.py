from flask import Flask, request, render_template
import nltk

app = Flask(__name__)

# Open the text file which is the knowledge base of our chatbot and read each line of text file
with open('data/dataset.txt', 'r') as file:
    f = file.readlines()

# From text file split line into two sides of question and its answers
pairs = [line.strip().split(':') for line in f]

# Create an empty list that will keep question and their answers splitted from pairs
conversation = []
for pair in pairs:
    question = nltk.word_tokenize(pair[0].lower())
    answer = nltk.word_tokenize(pair[1].lower())
    # Append question with their answers to conversation list
    conversation.append((question, answer))

# Create function that takes user input and process it to find the matching answer
def response(user_input):
    user_input = nltk.word_tokenize(user_input.lower())
    best_match = (0, None)

    for i, (question, answer) in enumerate(conversation):
        sim = nltk.jaccard_distance(set(user_input), set(question))

        if sim <= 0.1:  # If similarity is greater than or equal to 80%
            return ' '.join(answer)  # Return the answer
        elif best_match[1] is None or sim < best_match[0]:
            best_match = (sim, i)

    if best_match[1] is not None:
        return ' '.join(conversation[best_match[1]][1])  # Join tokens into a sentence for output
    else:
        return "I don't understand."

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    if user_input.lower() == 'katisha':
        return "Kwaheri. Asante"
    else:
        reply = response(user_input)
        return f"{reply}"

if __name__ == "__main__":
    app.run(debug=True)




# def generate_response(user_input):
#     inputs = tokenizer(user_input, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Extract the answer from the model's output
#     answer_start = torch.argmax(outputs.start_logits)
#     answer_end = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

#     return answer

# @app.route("/")
# def index():
#     return render_template('index.html')

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.form["msg"]
#     if user_input.lower() == 'katisha':
#         return "Kwaheri. Asante"
#     else:
#         reply = generate_response(user_input)
#         return reply

# if __name__ == "__main__":
#     app.run(debug=True)
