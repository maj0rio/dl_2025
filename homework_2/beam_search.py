import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class BeamCandidate:
    sequence: List[int]
    score: float
    length: int

def beam_search_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    num_beams: int = 5,
    length_penalty: float = 1.0,
    max_length: int = 1000,
    early_stopping: bool = True
) -> str:

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    unfinished_candidates: List[BeamCandidate] = []
    finished_candidates: List[BeamCandidate] = []
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs[0], num_beams)
        
        for prob, token_id in zip(top_probs, top_indices):
            candidate = BeamCandidate(
                sequence=input_ids[0].tolist() + [token_id.item()],
                score=float(prob),
                length=len(input_ids[0]) + 1
            )
            
            if token_id.item() == tokenizer.eos_token_id:
                finished_candidates.append(candidate)
            else:
                unfinished_candidates.append(candidate)

    while len(unfinished_candidates) > 0 and len(finished_candidates) < num_beams:
        all_candidates: List[BeamCandidate] = []

        for candidate in unfinished_candidates:
            with torch.no_grad():
                input_ids = torch.tensor([candidate.sequence])
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
  
                top_probs, top_indices = torch.topk(probs[0], num_beams)
  
                for prob, token_id in zip(top_probs, top_indices):
                    new_candidate = BeamCandidate(
                        sequence=candidate.sequence + [token_id.item()],
                        score=candidate.score * float(prob),
                        length=candidate.length + 1
                    )
                    all_candidates.append(new_candidate)

        all_candidates.sort(
            key=lambda x: x.score / (x.length ** length_penalty),
            reverse=True
        )

        unfinished_candidates = []
        for candidate in all_candidates[:num_beams]:
            if candidate.sequence[-1] == tokenizer.eos_token_id:
                finished_candidates.append(candidate)
            else:
                unfinished_candidates.append(candidate)

        if len(finished_candidates) >= num_beams and early_stopping:
            break
        if any(c.length >= max_length for c in unfinished_candidates):
            break

    if finished_candidates:
        best_candidate = max(
            finished_candidates,
            key=lambda x: x.score / (x.length ** length_penalty)
        )
    else:
        best_candidate = max(
            unfinished_candidates,
            key=lambda x: x.score / (x.length ** length_penalty)
        )

    return tokenizer.decode(best_candidate.sequence)

def main():

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    hedgehog_prompt = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    json_prompt = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    num_beanses = [1, 4, 4, 4, 8]
    lenght_penalties = [1, 1, 0.5, 2, 1]

    for num_beans, lenght_penalty in zip(num_beanses, lenght_penalties):
        print(f"Beam Search (num_beams={num_beans}, length_penalty={lenght_penalty}):")
        story = beam_search_decode(
            model, tokenizer, hedgehog_prompt,
            num_beams=num_beans,
            length_penalty=lenght_penalty
        )
        print(story)
        print("\n" + "="*80 + "\n")

    for num_beans, lenght_penalty in zip(num_beanses, lenght_penalties):
        print(f"Beam Search (num_beams={num_beans}, length_penalty={lenght_penalty}):")
        story = beam_search_decode(
            model, tokenizer, json_prompt,
            num_beams=num_beans,
            length_penalty=lenght_penalty
        )
        print(story)


if __name__ == "__main__":
    main() 