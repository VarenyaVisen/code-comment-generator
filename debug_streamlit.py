"""
Debug the Streamlit generation issue.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def test_basic_generation():
    """Test basic generation outside Streamlit."""
    
    print("🔍 Testing basic generation...")
    
    try:
        from src.inference.generator import CodeCommentGenerator
        
        # Initialize generator
        print("📥 Loading generator...")
        generator = CodeCommentGenerator()
        print("✅ Generator loaded successfully")
        
        # Test simple code
        test_code = """def add_numbers(a, b):
    result = a + b
    return result"""
        
        print(f"📝 Testing with code:")
        print(test_code)
        
        # Generate comment
        print("🤖 Generating comment...")
        comment = generator.generate_comment(
            test_code,
            max_length=30,
            temperature=0.3,
            top_k=20
        )
        
        print(f"✅ Generated: {comment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_generation()