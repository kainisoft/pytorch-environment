#!/usr/bin/env python3
"""
Validation script for the Chatbot-Qoder tutorial series.
Tests all components to ensure proper integration and functionality.
"""

import os
import json
import sys
from pathlib import Path

def test_directory_structure():
    """Test that all required directories and files exist."""
    print("🔍 Testing directory structure...")
    
    base_dir = Path(".")
    required_dirs = [
        "notebooks",
        "data/conversations", 
        "data/corpora",
        "data/embeddings",
        "models",
        "utils",
        "configs"
    ]
    
    required_files = [
        "README.md",
        "notebooks/01_pytorch_fundamentals.ipynb",
        "notebooks/02_tensor_operations.ipynb", 
        "notebooks/03_text_preprocessing.ipynb",
        "notebooks/04_neural_networks_basics.ipynb",
        "notebooks/05_language_modeling.ipynb",
        "notebooks/06_rule_based_chatbot.ipynb",
        "notebooks/07_retrieval_based_chatbot.ipynb",
        "notebooks/08_sequence_models.ipynb",
        "notebooks/09_attention_mechanisms.ipynb",
        "notebooks/10_transformer_basics.ipynb",
        "notebooks/11_generative_chatbot.ipynb",
        "notebooks/12_fine_tuning_deployment.ipynb",
        "data/conversations/simple_qa_pairs.json",
        "data/conversations/faq_knowledge.json",
        "data/corpora/ml_text_corpus.txt",
        "utils/__init__.py",
        "utils/text_utils.py",
        "utils/model_helpers.py",
        "utils/training_helpers.py",
        "utils/evaluation_helpers.py",
        "utils/chatbot_helpers.py",
        "configs/__init__.py",
        "configs/model_configs.py",
        "configs/training_configs.py",
        "configs/data_configs.py"
    ]
    
    # Check directories
    for dir_path in required_dirs:
        if not (base_dir / dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Check files
    for file_path in required_files:
        if not (base_dir / file_path).exists():
            print(f"❌ Missing file: {file_path}")
            return False
        else:
            print(f"✅ File exists: {file_path}")
    
    return True

def test_data_files():
    """Test that data files contain valid content."""
    print("\n📊 Testing data files...")
    
    # Test JSON files
    json_files = [
        "data/conversations/simple_qa_pairs.json",
        "data/conversations/faq_knowledge.json"
    ]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Valid JSON: {json_file} ({len(data)} entries)")
        except Exception as e:
            print(f"❌ Invalid JSON {json_file}: {e}")
            return False
    
    # Test text corpus
    try:
        with open("data/corpora/ml_text_corpus.txt", 'r') as f:
            content = f.read()
        print(f"✅ Text corpus: {len(content)} characters")
    except Exception as e:
        print(f"❌ Text corpus error: {e}")
        return False
    
    return True

def test_notebook_structure():
    """Test that notebooks have proper structure."""
    print("\n📓 Testing notebook structure...")
    
    notebook_dir = Path("notebooks")
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    
    expected_count = 12
    if len(notebooks) != expected_count:
        print(f"❌ Expected {expected_count} notebooks, found {len(notebooks)}")
        return False
    
    for i, notebook in enumerate(notebooks, 1):
        expected_name = f"{i:02d}_"
        if not notebook.name.startswith(expected_name):
            print(f"❌ Notebook naming issue: {notebook.name}")
            return False
        
        # Check notebook content
        try:
            with open(notebook, 'r') as f:
                content = json.load(f)
            
            if "cells" not in content:
                print(f"❌ Invalid notebook structure: {notebook.name}")
                return False
            
            print(f"✅ Notebook {i:02d}: {notebook.name} ({len(content['cells'])} cells)")
            
        except Exception as e:
            print(f"❌ Notebook error {notebook.name}: {e}")
            return False
    
    return True

def test_python_files():
    """Test that Python files have valid syntax."""
    print("\n🐍 Testing Python files...")
    
    python_files = [
        "utils/text_utils.py",
        "utils/model_helpers.py", 
        "utils/training_helpers.py",
        "utils/evaluation_helpers.py",
        "utils/chatbot_helpers.py",
        "configs/model_configs.py",
        "configs/training_configs.py",
        "configs/data_configs.py"
    ]
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                code = f.read()
            
            # Basic syntax check
            compile(code, py_file, 'exec')
            print(f"✅ Valid Python: {py_file}")
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {py_file}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading {py_file}: {e}")
            return False
    
    return True

def test_readme_content():
    """Test README content for completeness."""
    print("\n📖 Testing README content...")
    
    try:
        with open("README.md", 'r') as f:
            content = f.read()
        
        required_sections = [
            "# Chatbot-Qoder: Comprehensive Chatbot Tutorial Series",
            "## 📚 Learning Path",
            "## 🛠 Technical Requirements", 
            "## 🚀 Quick Start",
            "## 📁 Project Structure"
        ]
        
        for section in required_sections:
            if section not in content:
                print(f"❌ Missing README section: {section}")
                return False
            else:
                print(f"✅ README section: {section}")
        
        # Check for all 12 notebooks mentioned
        for i in range(1, 13):
            notebook_ref = f"{i:02d}_"
            if notebook_ref not in content:
                print(f"❌ Missing notebook reference: {notebook_ref}")
                return False
        
        print("✅ All notebook references found in README")
        
    except Exception as e:
        print(f"❌ README error: {e}")
        return False
    
    return True

def generate_summary():
    """Generate a summary of the tutorial series."""
    print("\n📋 Tutorial Series Summary:")
    print("=" * 50)
    
    # Count files and directories
    notebook_count = len(list(Path("notebooks").glob("*.ipynb")))
    util_files = len(list(Path("utils").glob("*.py")))
    config_files = len(list(Path("configs").glob("*.py")))
    data_files = len(list(Path("data").rglob("*.*")))
    
    print(f"📚 Notebooks: {notebook_count}")
    print(f"🛠️  Utility modules: {util_files}")
    print(f"⚙️  Configuration files: {config_files}")
    print(f"💾 Data files: {data_files}")
    
    # Estimate total lines of code
    total_lines = 0
    for py_file in Path(".").rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                total_lines += len(f.readlines())
        except:
            pass
    
    print(f"📝 Total Python code lines: {total_lines:,}")
    
    # List learning objectives
    print("\n🎯 Learning Progression:")
    topics = [
        "1. PyTorch Fundamentals",
        "2. Tensor Operations", 
        "3. Text Preprocessing",
        "4. Neural Network Basics",
        "5. Language Modeling",
        "6. Rule-based Chatbots",
        "7. Retrieval-based Chatbots", 
        "8. Sequence Models",
        "9. Attention Mechanisms",
        "10. Transformer Basics",
        "11. Generative Chatbots",
        "12. Fine-tuning & Deployment"
    ]
    
    for topic in topics:
        print(f"   {topic}")

def main():
    """Run all validation tests."""
    print("🚀 Chatbot-Qoder Tutorial Validation")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_data_files,
        test_notebook_structure, 
        test_python_files,
        test_readme_content
    ]
    
    all_passed = True
    for test_func in tests:
        if not test_func():
            all_passed = False
            break
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✨ The Chatbot-Qoder tutorial series is ready for use!")
        generate_summary()
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues above before using the tutorial.")
        sys.exit(1)

if __name__ == "__main__":
    main()