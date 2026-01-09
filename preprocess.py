import json
from colorama import Fore

instructions = []

with open('combined_data/merged_dataset_no_filenames.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for key, chunk in data.items():
        context = chunk.get('context', '')  
        for pairs in chunk['generated']:
            question = f"Context: {context.strip()}\n\nQuestion: {pairs['question']}"
            answer = pairs['answer']
            context_pair = {
                "question": question,
                "answer": answer
            }
            instructions.append(context_pair)
        print(Fore.YELLOW + f"Processed chunk {key} with {len(chunk['generated'])} Q&A pairs")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

with open('data/instruction.json', 'w', encoding='utf-8') as f:
    json.dump(instructions, f, indent=2, ensure_ascii=False)

with open('data/instruction.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(Fore.LIGHTMAGENTA_EX + json.dumps(data[:3], indent=2, ensure_ascii=False))
