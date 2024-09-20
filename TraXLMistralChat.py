import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton
from transformers import GPT2TokenizerFast
from TraXLMistralModel import TraXLMistralForCausalLM, TraXLMistralConfig
from safetensors.torch import load_file

class ChatBox(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("AI Chat Box")
        self.setGeometry(100, 100, 600, 400)

        # Set up the layout
        self.layout = QVBoxLayout()

        # Chat display area (QTextEdit acts as the chat area)
        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        self.layout.addWidget(self.chat_area)

        # Input field and send button layout
        self.input_layout = QHBoxLayout()

        # Input field (QLineEdit for user input)
        self.entry = QLineEdit(self)
        self.entry.setPlaceholderText("Type your message...")
        self.entry.returnPressed.connect(self.send_message)  # Bind Enter key to send_message
        self.input_layout.addWidget(self.entry)

        # Send button (QPushButton)
        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        self.input_layout.addWidget(self.send_button)

        # Add input layout to main layout
        self.layout.addLayout(self.input_layout)

        # Set the window's layout
        self.setLayout(self.layout)

        # Load the tokenizer and model from local paths
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        """Load the tokenizer from local files."""
        tokenizer = GPT2TokenizerFast(
            vocab_file='C:/Users/wonde/Wonder-Griffin/TraXLMistralForCausalLM/checkpoint-200245/vocab.json',
            merges_file='C:/Users/wonde/Wonder-Griffin/TraXLMistralForCausalLM/checkpoint-200245/merges.txt',
            tokenizer_file='C:/Users/wonde/Wonder-Griffin/TraXLMistralForCausalLM/checkpoint-200245/tokenizer.json',
        )
        return tokenizer

    def load_model(self):
        """Load the model from the local configuration and weights."""
        # Load the config manually from the local config.json file
        config = TraXLMistralConfig.from_json_file('C:/Users/wonde/Wonder-Griffin/TraXLMistralForCausalLM/checkpoint-200245/config.json')

        # Load the weights from the safetensors file
        model_weights_path = 'C:/Users/wonde/Wonder-Griffin/TraXLMistralForCausalLM/checkpoint-200245/model.safetensors'
        model = TraXLMistralForCausalLM(config)

        # Load the weights from the safetensors file
        state_dict = load_file(model_weights_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model
        
    def generate_ai_response(self, user_input):
        """Generate a response from the AI model."""
        # Tokenize the user input
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt')

        # Generate the response using the model
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    def send_message(self):
        """Handles sending a message when Enter is pressed or the Send button is clicked."""
        # Get user input
        user_input = self.entry.text()

        if user_input:
            # Display user input in the chat area
            self.chat_area.append(f"You: {user_input}")

            # Generate AI response
            ai_response = self.generate_ai_response(user_input)

            # Display AI response in the chat
            self.chat_area.append(f"AI: {ai_response}")

            # Clear the entry field
            self.entry.clear()


# Running the chatbox application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    chatbox = ChatBox()
    chatbox.show()
    sys.exit(app.exec_())