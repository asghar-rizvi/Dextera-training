import json
import random

with open("legal_data.json", "r", encoding="utf-8") as f:
    legal_data = json.load(f)

with open("lima_chat_sample.json", "r", encoding="utf-8") as f:
    chat_data = json.load(f)

with open("refusal_examples.json", "r", encoding="utf-8") as f:
    refusal_data = json.load(f)


merged_data = legal_data + chat_data + refusal_data
random.shuffle(merged_data)

with open("instruction.json", "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"Final dataset 'instruction.json' created with {len(merged_data)} total samples.")