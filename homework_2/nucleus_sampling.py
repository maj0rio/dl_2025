import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def nucleus_sampling(probs, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=0)

    nucleus_idx = torch.where(cumsum_probs >= top_p)[0]

    if len(nucleus_idx) == 0:
        nucleus_idx = torch.tensor([0])

    nucleus_probs = sorted_probs[:nucleus_idx[0] + 1]
    nucleus_tokens = sorted_indices[:nucleus_idx[0] + 1]

    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    chosen_idx = torch.multinomial(nucleus_probs, num_samples=1)

    return nucleus_tokens[chosen_idx].item()

def nucleus_sampling_decode(model, tokenizer, input_text, top_p=0.9, temperature=1.0, max_length=1000):
    encoding = tokenizer(input_text).encodings[0]
    input_ids = torch.tensor([encoding.ids])
    attention_mask = torch.tensor([encoding.attention_mask])

    generated_ids = input_ids.clone()
    current_length = input_ids.shape[1]

    while current_length < max_length:
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1]

        next_token_logits = next_token_logits / temperature

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        next_token = nucleus_sampling(probs, top_p)

        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
        current_length += 1

        if next_token == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    return generated_text

def main():
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    hedgehog_prompt = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    json_prompt = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    top_p_values = [0.9, 0.15]
    temperature = [0.5, 1.0]

    print("Generating hedgehog stories with different top_p values:")
    for temp in temperature:
        for top_p in top_p_values:
            print(f"\nTemperature: {temp}, Top-p: {top_p}")
            hedgehog_story = nucleus_sampling_decode(model, tokenizer, hedgehog_prompt, top_p=top_p, temperature=temp)
            print(hedgehog_story)

    print("\n" + "="*50 + "\n")

    print("Generating JSON with different top_p values:")
    for temp in temperature:
        for top_p in top_p_values:
            print(f"\nTemperature: {temp}, Top-p: {top_p}")
            json_output = nucleus_sampling_decode(model, tokenizer, json_prompt, top_p=top_p, temperature=temp)
            print(json_output)

if __name__ == "__main__":
    main()
