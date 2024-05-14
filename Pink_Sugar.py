import pandas as pd
from datasets import load_dataset
#dataset = load_dataset("danielpark/MQuAD-v1") # No longer available
dataset=load_dataset('flaviagiammarino/vqa-rad') # Also saved as 'medical_conversation_data.csv'

# Assuming 'dataset' is your DatasetDict and you're interested in the 'train' split
train_dataset = dataset['train']

# Extract 'question' and 'answer' fields
user_text = train_dataset['question']
bot_text = train_dataset['answer']

# Create a pandas DataFrame with 'user_text' and 'bot_text' columns
df = pd.DataFrame({
    'User': user_text,
    'Bot': bot_text
})

# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# Optionally, save the DataFrame to a CSV file
df.to_csv('medical_conversation_data.csv', index=False)

# Display the first few rows of the DataFrame to verify its diversity
print(df.head())

# Assuming 'df' is your Pandas DataFrame with columns 'User' and 'Bot'
df['formatted'] = "User: " + df['User'] + " \nBot: " + df['Bot'] + "\n"

training_text = "\n".join(df['formatted'].tolist())

from transformers import GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  

# Tokenize the training text
inputs = tokenizer(training_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # Set labels to be the same as input_ids
        return item

# Create the dataset
dataset = ConversationDataset(inputs)

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained('gpt2')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

prompt = "What causes nausea?"
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

# Generate a response
output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=1000,
    temperature=0.3,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,  # Prevent repeating n-grams
    num_beams=5,  # Beam search
    length_penalty=0.8,  # Adjust length of responses
    do_sample=True,
    num_return_sequences=1,
)

# Decode the generated sequence to text
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

# Extract the text after the prompt
response_text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]

print(response_text)
