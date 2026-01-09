from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 

import json
from typing import List 
from pydantic import BaseModel
from langchain_community.llms import Ollama
from prompt_template import prompt_template
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

print_lock = threading.Lock()
thread_local = threading.local()

def get_ollama_model():
    if not hasattr(thread_local, 'ollama_model'):
        thread_local.ollama_model = Ollama(model="gemma3:latest")
    return thread_local.ollama_model

def llm_call(data: str, num_records: int = 5) -> dict:
    llm = get_ollama_model()
    prompt = prompt_template(data, num_records)
    
    response = llm.invoke(prompt)

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        parsed_data = json.loads(response)
        
        if isinstance(parsed_data, list):
            return {"generated": parsed_data}
        elif isinstance(parsed_data, dict):
            return parsed_data
        else:
            # fallback
            return {"generated": []}

    except json.JSONDecodeError as e:
        with print_lock:
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {response}")
        return {"generated": []}


def process_chunk(i, chunk, chunker):
    thread_chunker = HybridChunker()
    with print_lock:
        print(Fore.YELLOW + f"Raw Text (Chunk {i}):\n{chunk.text[:300]}…" + Fore.RESET)
    
    enriched_text = thread_chunker.contextualize(chunk=chunk)
    
    with print_lock:
        print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text (Chunk {i}):\n{enriched_text[:300]}…" + Fore.RESET)
    
    data = llm_call(enriched_text)
    
    return i, {
        "generated": data.get("generated", []),
        "context": enriched_text
    }

if __name__ == "__main__": 

    pdf_file_path = 'law_dataSet/Anti-Rape (lnvestigation and Trial) Act. 2021 & Rules, 2022.pdf'
    converter = DocumentConverter()
    doc = converter.convert(pdf_file_path).document
    chunker = HybridChunker()
    chunks = list(chunker.chunk(dl_doc=doc))
        
    dataset = {}
    
    with ThreadPoolExecutor(max_workers=14) as executor: 
        futures = [executor.submit(process_chunk, i, chunk, chunker) for i, chunk in enumerate(chunks)]
        
        for future in tqdm(futures, total=len(chunks), desc="Processing Chunks"):
            i, result = future.result()
            dataset[i] = result
    
    output_path = 'Anti-Rape (lnvestigation and Trial) Act. 2021 & Rules, 2022.json' 
    with open(output_path, 'w') as f: 
        json.dump(dataset, f, indent=2)
    
    print(f"Test complete! Dataset saved to {output_path}")