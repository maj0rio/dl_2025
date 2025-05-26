import json
from pathlib import Path
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_dataset(output_dir: str = "data"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("sentence-transformers/natural-questions")

    data = []
    for item in dataset['train']:
        data.append({
            'question': item['query'],
            'passage': item['answer'],
            'label': 1
        })

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(output_path / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
   
    print(f"Dataset prepared and saved to {output_dir}")
    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

if __name__ == "__main__":
    prepare_dataset()
