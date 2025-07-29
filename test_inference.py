"""
Test the inference pipeline without interactive mode.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.inference.generator import CodeCommentGenerator

def test_generator():
    """Test the generator on simple examples."""
    
    print("🧪 Testing Code Comment Generator...\n")
    
    try:
        # Initialize generator
        generator = CodeCommentGenerator()
        
        # Simple test
        test_code = """def add_numbers(a, b):
    result = a + b
    return result"""
        
        print("📝 Test Code:")
        print(test_code)
        
        comment = generator.generate_comment(
            test_code, 
            max_length=20,
            temperature=0.5
        )
        
        print(f"\n🎉 Generated Comment: {comment}")
        print("\n✅ Inference test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generator()