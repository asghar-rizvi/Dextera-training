import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import traceback

model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "pak_legal_llama_adapter"  

print("Initializing Pakistani Law Llama...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Load Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

SYSTEM_PROMPT = """Role: Specialized Expert in Pakistani Criminal Law (PPC & CrPC).
Rules of Engagement:
1. SCOPE: You only answer queries related to Pakistani criminal statutes, precedents, and procedures.
2. REFUSAL: For ANY non-legal or non-Pakistani criminal law question, you MUST respond with: "I'm designed to assist only with questions related to Pakistani criminal law. Please ask a question about criminal law in Pakistan."
3. CITATIONS: Always cite the relevant Section of the Pakistan Penal Code (PPC) or Article of the Code of Criminal Procedure (CrPC) when providing legal information.
4. TONE: Maintain a formal, objective, and analytical tone.
5. CONTEXT: If legal text is provided in the query, prioritize that text for your reasoning.

Constraint: Do not provide personal opinions or general legal advice.
Reminder: Act Normal when greet and If i asked anything other than pakistan criminal law just SAY: 'NOT A PAKISTAN CRIMINAL LAW QUERY '"""

def generate_response(user_query):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            
            {"role": "user", "content": "What is your Name?"},
            {"role": "assistant", "content": "Hi, I am Dextera. I was built to assist you with Pakistani criminal law."},
            
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, I am Dextera. I was built to assist you with Pakistani criminal law."},

            {"role": "user", "content": "What is your purpose"},
            {"role": "assistant", "content": "Hi, I am Dextera. I was built to assist you with Pakistani criminal law."},

            {"role": "user", "content": "Can you write me a poem about the sea?"},
            {"role": "assistant", "content": "NOT A PAKISTAN CRIMINAL LAW QUERY. I'm designed to assist only with questions related to Pakistani criminal law. Please ask a question about criminal law in Pakistan."},
            
            {"role": "user", "content": "How do I cook Biryani?"},
            {"role": "assistant", "content": "NOT A PAKISTAN CRIMINAL LAW QUERY. I'm designed to assist only with questions related to Pakistani criminal law. Please ask a question about criminal law in Pakistan."},
            
            {"role": "user", "content": user_query},
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = tokenizer(formatted_text, return_tensors="pt", padding=True).to(model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.01, 
            top_p=0.9,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id=tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )
        
        new_tokens = output_tokens[0][len(inputs.input_ids[0]):]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
        
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    print("\n" + "="*50)
    print("--- Pakistani Law Llama Bot Active ---")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*50)
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]: 
            break
            
        print("\nAssistant: ", end="", flush=True)
        response = generate_response(query)
        print(response)
        print("\n" + "-"*30)