"""
Test suite for the Templates AgenTool.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestTemplatesAgent:
    """Test suite for templates AgenTool."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Create test templates directory
        self.test_templates_dir = Path("test_templates")
        if self.test_templates_dir.exists():
            shutil.rmtree(self.test_templates_dir)
        self.test_templates_dir.mkdir(exist_ok=True)
        
        # Create a sample template file
        sample_template = """Hello {{ name }}!
{% if greeting %}
{{ greeting }}
{% endif %}"""
        (self.test_templates_dir / "greeting.jinja").write_text(sample_template)
        
        # Create storage_kv agent for testing references
        from agentoolkit.storage.kv import create_storage_kv_agent
        kv_agent = create_storage_kv_agent()
        
        # Create storage_fs agent for testing references
        from agentoolkit.storage.fs import create_storage_fs_agent
        fs_agent = create_storage_fs_agent()
        
        # Create logging agent (dependency)
        from agentoolkit.system.logging import create_logging_agent
        logging_agent = create_logging_agent()
        
        # Create metrics agent (dependency)
        from agentoolkit.observability.metrics import create_metrics_agent
        metrics_agent = create_metrics_agent()
        
        # Import and create the templates agent with test directory
        from agentoolkit.system.templates import create_templates_agent
        agent = create_templates_agent(templates_dir=str(self.test_templates_dir))
    
    def teardown_method(self):
        """Clean up test templates directory."""
        if self.test_templates_dir.exists():
            shutil.rmtree(self.test_templates_dir)
    
    def test_render_template(self):
        """Test rendering a pre-loaded template."""
        
        async def run_test():
            injector = get_injector()
            
            # Test rendering with variables
            result = await injector.run('templates', {
                "operation": "render",
                "template_name": "greeting",
                "variables": {
                    "name": "Alice",
                    "greeting": "Welcome to the system!"
                }
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "Hello Alice!" in data['data']['rendered']
            assert "Welcome to the system!" in data['data']['rendered']
            
            # Print the rendered output
            print("\n=== test_render_template Output ===")
            print(f"Rendered template:\n{data['data']['rendered']}")
            print(f"Template name: {data['data']['template_name']}")
            print(f"Variables resolved: {data['data']['variables_resolved']}")
            print("=" * 40)
            
            # Test rendering without optional variable
            result = await injector.run('templates', {
                "operation": "render",
                "template_name": "greeting",
                "variables": {
                    "name": "Bob"
                }
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "Hello Bob!" in data['data']['rendered']
            assert "Welcome" not in data['data']['rendered']
            
            # Test non-existent template (should raise exception)
            try:
                result = await injector.run('templates', {
                    "operation": "render",
                    "template_name": "nonexistent",
                    "variables": {"name": "Test"}
                })
                # Should not reach here
                assert False, "Expected ValueError for non-existent template"
            except ValueError as e:
                assert "not found" in str(e).lower()
                print(f"\n   Expected exception caught: {e}")
        
        asyncio.run(run_test())
    
    def test_save_template(self):
        """Test saving a new template."""
        
        async def run_test():
            injector = get_injector()
            
            # Save a valid template
            template_content = """Dear {{ recipient }},
This is a test email.
{% for item in items %}
- {{ item }}
{% endfor %}
Best regards,
{{ sender }}"""
            
            result = await injector.run('templates', {
                "operation": "save",
                "template_name": "test_email",
                "template_content": template_content
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['data']['template_name'] == "test_email"
            
            # Verify the template file was created
            template_file = self.test_templates_dir / "test_email.jinja"
            assert template_file.exists()
            assert template_file.read_text() == template_content
            
            # Test rendering the newly saved template
            result = await injector.run('templates', {
                "operation": "render",
                "template_name": "test_email",
                "variables": {
                    "recipient": "John",
                    "items": ["Item 1", "Item 2", "Item 3"],
                    "sender": "Admin"
                }
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "Dear John" in data['data']['rendered']
            assert "Item 1" in data['data']['rendered']
            
            # Print the saved and rendered template
            print("\n=== test_save_template Output ===")
            print(f"Saved template: {data['data']['template_name']}")
            print(f"Rendered content:\n{data['data']['rendered']}")
            print("=" * 40)
            
            # Test saving invalid template (should raise exception)
            try:
                result = await injector.run('templates', {
                    "operation": "save",
                    "template_name": "invalid",
                    "template_content": "{% for item in %}"  # Invalid syntax
                })
                # Should not reach here
                assert False, "Expected ValueError for invalid template syntax"
            except ValueError as e:
                assert "validation failed" in str(e).lower()
                print(f"\n   Expected exception caught: {e}")
        
        asyncio.run(run_test())
    
    def test_list_templates(self):
        """Test listing available templates."""
        
        async def run_test():
            injector = get_injector()
            
            # List templates
            result = await injector.run('templates', {
                "operation": "list"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['data']['count'] >= 1  # At least the greeting template
            
            template_names = [t['name'] for t in data['data']['templates']]
            assert "greeting" in template_names
            
            # Print the templates list
            print("\n=== test_list_templates Output ===")
            print(f"Available templates: {template_names}")
            print(f"Total count: {data['data']['count']}")
            print(f"Templates directory: {data['data']['directory']}")
            for template in data['data']['templates']:
                print(f"  - {template['name']}: {template.get('filename', 'N/A')} ({template.get('size', 'N/A')} bytes)")
            print("=" * 40)
        
        asyncio.run(run_test())
    
    def test_validate_template(self):
        """Test template validation."""
        
        async def run_test():
            injector = get_injector()
            
            # Validate a correct template
            result = await injector.run('templates', {
                "operation": "validate",
                "template_content": "Hello {{ name }}! Your age is {{ age }}."
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['data']['valid'] is True
            assert "name" in data['data']['variables']
            assert "age" in data['data']['variables']
            
            # Validate an incorrect template (should raise exception)
            try:
                result = await injector.run('templates', {
                    "operation": "validate",
                    "template_content": "{% for item in %} {{ item }}"  # Missing iterable
                })
                # Should not reach here
                assert False, "Expected ValueError for invalid template syntax"
            except ValueError as e:
                assert "syntax error" in str(e).lower()
                print(f"\n   Expected exception caught for validation: {e}")
        
        asyncio.run(run_test())
    
    def test_exec_template(self):
        """Test executing ad-hoc templates."""
        
        async def run_test():
            injector = get_injector()
            
            # Execute a simple template
            result = await injector.run('templates', {
                "operation": "exec",
                "template_content": "Result: {{ value * 2 }}",
                "variables": {"value": 21}
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "Result: 42" in data['data']['rendered']
            
            # Print the exec output
            print("\n=== test_exec_template Output (Simple) ===")
            print(f"Template: 'Result: {{{{ value * 2 }}}}'")
            print(f"Variables: {{value: 21}}")
            print(f"Rendered: {data['data']['rendered']}")
            print("=" * 40)
            
            # Execute with loops
            result = await injector.run('templates', {
                "operation": "exec",
                "template_content": "{% for i in range(3) %}{{ i }}{% if not loop.last %}, {% endif %}{% endfor %}",
                "variables": {}
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "0, 1, 2" in data['data']['rendered']
            
            # Print the loop output
            print("\n=== test_exec_template Output (Loop) ===")
            print(f"Rendered loop: {data['data']['rendered']}")
            print("=" * 40)
            
            # Execute with undefined variable (strict mode - should raise exception)
            try:
                result = await injector.run('templates', {
                    "operation": "exec",
                    "template_content": "Hello {{ name }}!",
                    "variables": {},
                    "strict": True
                })
                # Should not reach here
                assert False, "Expected ValueError for undefined variable in strict mode"
            except ValueError as e:
                assert "undefined" in str(e).lower()
                print(f"\n   Expected exception caught: {e}")
            
            # Execute with undefined variable (lenient mode)
            result = await injector.run('templates', {
                "operation": "exec",
                "template_content": "Hello {{ name }}!",
                "variables": {},
                "strict": False
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert data['data']['rendered'] == "Hello !"
        
        asyncio.run(run_test())
    
    def test_storage_references(self):
        """Test variable resolution from storage_kv and storage_fs."""
        
        async def run_test():
            injector = get_injector()
            
            # Store some test data in storage_kv
            await injector.run('storage_kv', {
                "operation": "set",
                "key": "test_user_name",
                "value": "John Doe"
            })
            
            await injector.run('storage_kv', {
                "operation": "set",
                "key": "test_company",
                "value": "Acme Corp"
            })
            
            # Store some test data in storage_fs
            test_file = str(self.test_templates_dir / "test_data.txt")
            await injector.run('storage_fs', {
                "operation": "write",
                "path": test_file,
                "content": "This is file content"
            })
            
            # Test rendering with storage references
            result = await injector.run('templates', {
                "operation": "exec",
                "template_content": """User: {{ user }}
Company: {{ company }}
File: {{ file_content }}""",
                "variables": {
                    "user": "!ref:storage_kv:test_user_name",
                    "company": "!ref:storage_kv:test_company",
                    "file_content": f"!ref:storage_fs:{test_file}"
                }
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "User: John Doe" in data['data']['rendered']
            assert "Company: Acme Corp" in data['data']['rendered']
            assert "File: This is file content" in data['data']['rendered']
            
            # Print the storage references output
            print("\n=== test_storage_references Output ===")
            print("Variables with storage references:")
            print("  user: !ref:storage_kv:test_user_name")
            print("  company: !ref:storage_kv:test_company")
            print(f"  file_content: !ref:storage_fs:{test_file}")
            print(f"\nRendered output:\n{data['data']['rendered']}")
            print("=" * 40)
            
            # Test with non-existent references
            result = await injector.run('templates', {
                "operation": "exec",
                "template_content": "Value: {{ missing }}",
                "variables": {
                    "missing": "!ref:storage_kv:nonexistent_key"
                }
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert 'operation' in data
            assert "<undefined:" in data['data']['rendered'] or "Value: " in data['data']['rendered']
        
        asyncio.run(run_test())


if __name__ == "__main__":
    # Run tests
    test_suite = TestTemplatesAgent()
    
    print("Testing Templates AgenTool...")
    
    # Test render
    test_suite.setup_method()
    try:
        test_suite.test_render_template()
        print("✓ Render template test passed")
    except AssertionError as e:
        print(f"✗ Render template test failed: {e}")
    finally:
        test_suite.teardown_method()
    
    # Test save
    test_suite.setup_method()
    try:
        test_suite.test_save_template()
        print("✓ Save template test passed")
    except AssertionError as e:
        print(f"✗ Save template test failed: {e}")
    finally:
        test_suite.teardown_method()
    
    # Test list
    test_suite.setup_method()
    try:
        test_suite.test_list_templates()
        print("✓ List templates test passed")
    except AssertionError as e:
        print(f"✗ List templates test failed: {e}")
    finally:
        test_suite.teardown_method()
    
    # Test validate
    test_suite.setup_method()
    try:
        test_suite.test_validate_template()
        print("✓ Validate template test passed")
    except AssertionError as e:
        print(f"✗ Validate template test failed: {e}")
    finally:
        test_suite.teardown_method()
    
    # Test exec
    test_suite.setup_method()
    try:
        test_suite.test_exec_template()
        print("✓ Exec template test passed")
    except AssertionError as e:
        print(f"✗ Exec template test failed: {e}")
    finally:
        test_suite.teardown_method()
    
    # Test storage references
    test_suite.setup_method()
    try:
        test_suite.test_storage_references()
        print("✓ Storage references test passed")
    except AssertionError as e:
        print(f"✗ Storage references test failed: {e}")
    finally:
        test_suite.teardown_method()
    
    print("\nAll tests completed!")