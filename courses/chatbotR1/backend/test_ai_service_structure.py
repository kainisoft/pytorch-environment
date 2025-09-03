#!/usr/bin/env python3
"""
Test script for AI Service structure and imports (without PyTorch dependencies).

This script tests the AI service structure and basic functionality
without requiring PyTorch to be installed.
"""

import sys
import os
import ast
import inspect

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))


def test_file_structure():
    """Test that all required files exist."""
    print("\n=== Testing File Structure ===")
    
    required_files = [
        'app/services/__init__.py',
        'app/services/ai_service.py',
        'app/core/config.py',
        'app/routers/health.py',
        'app/main.py'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True


def test_ai_service_structure():
    """Test AI service file structure and class definitions."""
    print("\n=== Testing AI Service Structure ===")
    
    ai_service_path = os.path.join(os.path.dirname(__file__), 'app/services/ai_service.py')
    
    with open(ai_service_path, 'r') as f:
        content = f.read()
    
    # Parse the AST to check for required classes and functions
    tree = ast.parse(content)
    
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    
    required_classes = ['ModelConfig', 'AIService']
    required_functions = [
        'get_ai_service', 
        'initialize_ai_service', 
        'cleanup_ai_service'
    ]
    
    print("Classes found:", classes)
    print("Functions found:", [f for f in functions if not f.startswith('_')])
    
    # Check required classes
    for cls in required_classes:
        if cls in classes:
            print(f"✓ Class {cls} found")
        else:
            print(f"✗ Class {cls} missing")
            return False
    
    # Check required functions
    for func in required_functions:
        if func in functions:
            print(f"✓ Function {func} found")
        else:
            print(f"✗ Function {func} missing")
            return False
    
    return True


def test_config_integration():
    """Test configuration integration."""
    print("\n=== Testing Configuration Integration ===")
    
    try:
        from app.core.config import settings
        
        # Check AI-related settings
        ai_settings = [
            'model_path',
            'model_name', 
            'device',
            'max_context_length',
            'max_response_length'
        ]
        
        for setting in ai_settings:
            if hasattr(settings, setting):
                value = getattr(settings, setting)
                print(f"✓ {setting}: {value}")
            else:
                print(f"✗ {setting} missing from config")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False


def test_services_init():
    """Test services package initialization."""
    print("\n=== Testing Services Package ===")
    
    services_init_path = os.path.join(os.path.dirname(__file__), 'app/services/__init__.py')
    
    with open(services_init_path, 'r') as f:
        content = f.read()
    
    required_exports = [
        'AIService',
        'ModelConfig',
        'get_ai_service',
        'initialize_ai_service',
        'cleanup_ai_service'
    ]
    
    for export in required_exports:
        if export in content:
            print(f"✓ {export} exported")
        else:
            print(f"✗ {export} not exported")
            return False
    
    return True


def test_health_router_integration():
    """Test health router integration with AI service."""
    print("\n=== Testing Health Router Integration ===")
    
    health_router_path = os.path.join(os.path.dirname(__file__), 'app/routers/health.py')
    
    with open(health_router_path, 'r') as f:
        content = f.read()
    
    # Check for AI service integration
    checks = [
        'from app.services import get_ai_service',
        'ai_service = get_ai_service()',
        'ai_health = ai_service.health_check()',
        '/model/info',
        '/model/load',
        '/model/unload'
    ]
    
    for check in checks:
        if check in content:
            print(f"✓ Found: {check}")
        else:
            print(f"✗ Missing: {check}")
            return False
    
    return True


def test_main_app_integration():
    """Test main application integration."""
    print("\n=== Testing Main App Integration ===")
    
    main_app_path = os.path.join(os.path.dirname(__file__), 'app/main.py')
    
    with open(main_app_path, 'r') as f:
        content = f.read()
    
    # Check for AI service integration
    checks = [
        'from app.services import get_ai_service, cleanup_ai_service',
        'ai_service = get_ai_service()',
        'cleanup_ai_service()'
    ]
    
    for check in checks:
        if check in content:
            print(f"✓ Found: {check}")
        else:
            print(f"✗ Missing: {check}")
            return False
    
    return True


def test_requirements_coverage():
    """Test that implementation covers the task requirements."""
    print("\n=== Testing Requirements Coverage ===")
    
    requirements = [
        "AIService class in app/services/ai_service.py",
        "Model loading system with device selection",
        "Model configuration management", 
        "Health checking functionality",
        "Model cleanup and memory management utilities"
    ]
    
    # Check AIService class exists
    ai_service_path = os.path.join(os.path.dirname(__file__), 'app/services/ai_service.py')
    with open(ai_service_path, 'r') as f:
        content = f.read()
    
    coverage_checks = [
        ('AIService class', 'class AIService'),
        ('Device selection', '_setup_device'),
        ('Model loading', 'load_model'),
        ('Configuration management', 'class ModelConfig'),
        ('Health checking', 'health_check'),
        ('Memory management', 'cleanup'),
        ('Model info', 'get_model_info'),
        ('Memory usage tracking', '_get_memory_usage')
    ]
    
    for desc, check in coverage_checks:
        if check in content:
            print(f"✓ {desc}: implemented")
        else:
            print(f"✗ {desc}: missing")
            return False
    
    return True


def main():
    """Run all structure tests."""
    print("AI Service Structure Test Suite")
    print("===============================")
    
    tests = [
        test_file_structure,
        test_ai_service_structure,
        test_config_integration,
        test_services_init,
        test_health_router_integration,
        test_main_app_integration,
        test_requirements_coverage
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            all_passed = False
    
    print("\n=== Test Results ===")
    if all_passed:
        print("✓ All structure tests passed!")
        print("AI Service foundation is properly implemented.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())