import json
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

ds = load_dataset("GAIR/lima", revision="refs/convert/parquet", split="train")
lima_qa_pairs = []

for example in ds:
    conv = example['conversations']
    if len(conv) >= 2:
        qa_pair = {
            "question": conv[0],
            "answer": conv[1]
        }
        lima_qa_pairs.append(qa_pair)


with open("lima_chat_sample.json", "w") as f:
    json.dump(lima_qa_pairs[:500], f, indent=2)

print(f"Extracted {len(lima_qa_pairs[:500])} chat pairs.")