"""
Streamlit web app for Code Comment Generator - Debug Version.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import traceback

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure page
st.set_page_config(
    page_title="Code Comment Generator",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Code Comment Generator")
st.markdown("AI-powered Transformer model that generates meaningful comments for Python code")

@st.cache_resource
def load_generator():
    """Load the generator with detailed error reporting."""
    try:
        st.write("🔄 Loading model...")
        
        from src.inference.generator import CodeCommentGenerator
        
        generator = CodeCommentGenerator()
        st.write("✅ Model loaded successfully!")
        return generator, None
        
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {e}"
        st.write(f"❌ {error_msg}")
        return None, error_msg
        
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        st.write(f"❌ {error_msg}")
        st.write("Full error:")
        st.code(traceback.format_exc())
        return None, error_msg

def clean_comment(comment):
    """Clean up generated comment."""
    if not comment:
        return "No comment generated."
    
    # Remove special tokens
    comment = comment.replace('<START>', '').replace('<END>', '')
    comment = comment.replace('<PAD>', '').replace('<UNK>', '')
    comment = comment.replace('<CODE>', '').replace('<COMMENT>', '')
    comment = comment.replace('<INDENT>', '').replace('<NEWLINE>', '')
    
    # Clean up extra spaces
    comment = ' '.join(comment.split())
    
    # Fallback for poor outputs
    if len(comment.strip()) < 3:
        return "Function that performs a specific operation."
    
    return comment.strip()

def main():
    """Main app function."""
    
    # Load generator
    with st.spinner("Loading AI model..."):
        generator, error = load_generator()
    
    if error:
        st.error(f"❌ Failed to load model: {error}")
        
        # Show file check
        st.subheader("🔍 Debug Information")
        
        model_path = "checkpoints/best_model.pt"
        tokenizer_path = "data/processed/tokenizer.pkl"
        
        st.write("**Checking required files:**")
        st.write(f"Model file exists: {Path(model_path).exists()}")
        st.write(f"Tokenizer file exists: {Path(tokenizer_path).exists()}")
        
        if Path(model_path).exists():
            st.write(f"Model file size: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        st.info("💡 Make sure you've trained the model first!")
        return
    
    # Input section
    st.subheader("📝 Enter Your Python Code")
    
    # Example codes
    examples = {
        "Add Numbers": '''def add_numbers(a, b):
    result = a + b
    return result''',
        
        "Calculate Area": '''def calculate_area(radius):
    pi = 3.14159
    area = pi * radius * radius
    return area''',
        
        "Find Maximum": '''def find_maximum(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num'''
    }
    
    # Example selector
    selected_example = st.selectbox("Choose an example:", ["Custom"] + list(examples.keys()))
    
    if selected_example != "Custom":
        default_code = examples[selected_example]
    else:
        default_code = '''def your_function():
    # Enter your Python code here...
    pass'''
    
    # Code input
    code_input = st.text_area(
        "Python Code:",
        value=default_code,
        height=200,
        help="Enter a complete Python function"
    )
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 1.0, 0.3, 0.1)
    with col2:
        max_length = st.slider("Max Length", 10, 50, 20, 5)
    
    # Generate button
    if st.button("🚀 Generate Comment", type="primary"):
        if not code_input.strip():
            st.error("❌ Please enter some Python code!")
            return
        
        try:
            with st.spinner("🤖 Generating comment..."):
                st.write("🔄 Processing code...")
                
                # Show what we're processing
                st.code(code_input[:100] + "..." if len(code_input) > 100 else code_input)
                
                # Generate comment with debug info
                raw_comment = generator.generate_comment(
                    code_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=20,
                    top_p=0.8
                )
                
                st.write(f"🔍 Raw output: {raw_comment}")
                
                # Clean comment
                cleaned_comment = clean_comment(raw_comment)
                
                # Display results
                st.success("✅ Comment generated!")
                st.markdown(f"**Generated Comment:** {cleaned_comment}")
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            
            # Show detailed error
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()