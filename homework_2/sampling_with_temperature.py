import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def sampling_with_temperature_decode(model, tokenizer, input_text, temperature=1.0, max_length=1000):
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

        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
        current_length += 1

        if next_token.item() == 151645:
            break

    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    return generated_text

def main():
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    hedgehog_prompt = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    json_prompt = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    temperatures = [0.001, 0.1, 0.5, 1.0, 10.0]

    print("Generating hedgehog stories with different temperatures:")
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        hedgehog_story = sampling_with_temperature_decode(model, tokenizer, hedgehog_prompt, temperature=temp)
        print(hedgehog_story)

    print("\n" + "="*50 + "\n")

    print("Generating JSON with different temperatures:")
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        json_output = sampling_with_temperature_decode(model, tokenizer, json_prompt, temperature=temp)
        print(json_output)


if __name__ == "__main__":
    main() 