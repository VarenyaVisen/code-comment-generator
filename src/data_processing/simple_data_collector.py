"""
Simple data collector using pre-existing code-comment datasets.
"""

import json
import requests
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd


class SimpleDataCollector:
    """Downloads and processes existing code-comment datasets."""
    
    def __init__(self):
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_dataset(self) -> List[Dict]:
        """Create a small sample dataset manually for testing."""
        
        print("📝 Creating sample dataset...")
        
        # Manually created sample data (real Python functions with docstrings)
        sample_data = [
            {
                "function_code": """def add_numbers(a, b):
    result = a + b
    return result""",
                "docstring": "Add two numbers together and return the result."
            },
            {
                "function_code": """def calculate_area(radius):
    pi = 3.14159
    area = pi * radius * radius
    return area""",
                "docstring": "Calculate the area of a circle given its radius. Returns the area as a float."
            },
            {
                "function_code": """def find_maximum(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers[1:]:
        if num > max_num:
            max_num = num
    return max_num""",
                "docstring": "Find the maximum value in a list of numbers. Returns None if the list is empty."
            },
            {
                "function_code": """def reverse_string(text):
    reversed_text = ""
    for char in text:
        reversed_text = char + reversed_text
    return reversed_text""",
                "docstring": "Reverse a string by iterating through characters backwards."
            },
            {
                "function_code": """def count_words(sentence):
    words = sentence.split()
    return len(words)""",
                "docstring": "Count the number of words in a sentence by splitting on whitespace."
            },
            {
                "function_code": """def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True""",
                "docstring": "Check if a number is prime. Returns True if prime, False otherwise."
            },
            {
                "function_code": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
                "docstring": "Calculate the factorial of a number using iterative approach."
            },
            {
                "function_code": """def merge_lists(list1, list2):
    merged = []
    merged.extend(list1)
    merged.extend(list2)
    return merged""",
                "docstring": "Merge two lists into a single list containing all elements."
            },
            {
                "function_code": """def filter_even_numbers(numbers):
    even_numbers = []
    for num in numbers:
        if num % 2 == 0:
            even_numbers.append(num)
    return even_numbers""",
                "docstring": "Filter a list to return only even numbers."
            },
            {
                "function_code": """def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    return total / len(numbers)""",
                "docstring": "Calculate the average of a list of numbers. Returns 0 if list is empty."
            }
        ]
        
        print(f"✅ Created {len(sample_data)} sample function-comment pairs")
        return sample_data
    
    def download_huggingface_dataset(self) -> List[Dict]:
        """Download a real code-comment dataset from Hugging Face."""
        
        print("🤗 Downloading dataset from Hugging Face...")
        
        try:
            # Using CodeSearchNet dataset (simplified)
            # This is a real dataset used for code-comment research
            url = "https://huggingface.co/datasets/code_search_net/resolve/main/data/python_train_0.jsonl"
            
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                print("✅ Successfully downloaded CodeSearchNet data")
                
                # Parse JSONL (JSON Lines format)
                lines = response.text.strip().split('\n')
                dataset = []
                
                for line in lines[:50]:  # Take first 50 examples
                    try:
                        data = json.loads(line)
                        if 'code' in data and 'docstring' in data:
                            # Clean and format the data
                            code = data['code'].strip()
                            docstring = data['docstring'].strip()
                            
                            if len(code) > 20 and len(docstring) > 10:  # Basic filtering
                                dataset.append({
                                    "function_code": code,
                                    "docstring": docstring
                                })
                    except json.JSONDecodeError:
                        continue
                
                print(f"✅ Processed {len(dataset)} examples from CodeSearchNet")
                return dataset
            
        except Exception as e:
            print(f"❌ Error downloading HuggingFace dataset: {e}")
        
        # Fallback to sample data
        print("📝 Falling back to sample dataset...")
        return self.create_sample_dataset()
    
    def expand_sample_dataset(self, base_data: List[Dict]) -> List[Dict]:
        """Expand the sample dataset by creating variations."""
        
        print("🔄 Expanding dataset with variations...")
        
        expanded_data = base_data.copy()
        
        # Add more sample functions
        additional_samples = [
            {
                "function_code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
                "docstring": "Perform binary search on a sorted array to find target element. Returns index if found, -1 otherwise."
            },
            {
                "function_code": """def remove_duplicates(items):
    unique_items = []
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
    return unique_items""",
                "docstring": "Remove duplicate items from a list while preserving order."
            },
            {
                "function_code": """def validate_email(email):
    if '@' not in email:
        return False
    parts = email.split('@')
    if len(parts) != 2:
        return False
    return '.' in parts[1]""",
                "docstring": "Validate email format by checking for @ symbol and domain structure."
            },
            {
                "function_code": """def convert_temperature(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit""",
                "docstring": "Convert temperature from Celsius to Fahrenheit."
            },
            {
                "function_code": """def find_common_elements(list1, list2):
    common = []
    for item in list1:
        if item in list2 and item not in common:
            common.append(item)
    return common""",
                "docstring": "Find common elements between two lists without duplicates."
            }
        ]
        
        expanded_data.extend(additional_samples)
        
        print(f"✅ Expanded dataset to {len(expanded_data)} examples")
        return expanded_data
    
    def save_dataset(self, data: List[Dict], filename: str):
        """Save dataset to JSON file."""
        
        filepath = self.raw_data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(data)} examples to {filepath}")
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame(data)
        csv_path = filepath.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Also saved as CSV: {csv_path}")
    
    def collect_all_data(self):
        """Main method to collect all training data."""
        
        print("🚀 Starting simple data collection...\n")
        
        # Try to get real data, fallback to sample
        dataset = self.download_huggingface_dataset()
        
        # Expand with additional samples
        expanded_dataset = self.expand_sample_dataset(dataset)
        
        # Save the final dataset
        self.save_dataset(expanded_dataset, 'code_comments_dataset.json')
        
        # Create summary
        summary = {
            "total_examples": len(expanded_dataset),
            "average_code_length": sum(len(item["function_code"]) for item in expanded_dataset) / len(expanded_dataset),
            "average_comment_length": sum(len(item["docstring"]) for item in expanded_dataset) / len(expanded_dataset),
            "sample_example": expanded_dataset[0] if expanded_dataset else None
        }
        
        summary_path = self.raw_data_dir / 'dataset_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Data collection complete!")
        print(f"📊 Collected {len(expanded_dataset)} function-comment pairs")
        print(f"📋 Summary saved to {summary_path}")
        
        return expanded_dataset


if __name__ == "__main__":
    collector = SimpleDataCollector()
    data = collector.collect_all_data()
    
    # Show a sample
    if data:
        print(f"\n📝 Sample function-comment pair:")
        print("="*50)
        sample = data[0]
        print("Function Code:")
        print(sample["function_code"])
        print("\nDocstring:")
        print(sample["docstring"])
        print("="*50)