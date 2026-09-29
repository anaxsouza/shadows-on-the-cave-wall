
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.loaders.conll.loader import CONLLLoader



def test_conll_loader():
    print("Testing CONLLLoader...")
    loader = CONLLLoader()
    # Load raw dataset first
    try:
        raw_dataset = loader.load_raw_dataset(split='test')
        print(f"Loaded raw dataset with {len(raw_dataset)} examples.")
        
        # Convert to examples (first 500 to catch Nadim Ladki)
        examples = loader.convert_to_examples(raw_dataset, max_examples=500)
        
        print(f"Converted {len(examples)} examples.")
        
        target_text = "Nadim Ladki"
        found = False
        for ex in examples:
            if target_text in ex.text:
                print(f"Found '{target_text}': {ex.text}")
                print(f"Entities: {ex.entities}")
                found = True
                break
        
        if not found:
            print(f"'{target_text}' not found in first 500 examples.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_conll_loader()
