"""
Basic tests for Black LangCube library structure and imports.
Updated for the new src layout structure.
"""

import sys
import unittest
from pathlib import Path

# Add the src directory to the path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


class TestLibraryStructure(unittest.TestCase):
    """Test basic library structure and imports."""
    
    def test_main_package_import(self):
        """Test that the main package can be imported."""
        try:
            import black_langcube
            self.assertTrue(hasattr(black_langcube, '__version__'))
            self.assertTrue(hasattr(black_langcube, '__description__'))
            self.assertEqual(black_langcube.__version__, "0.1.0")
        except ImportError as e:
            self.fail(f"Failed to import main package: {e}")
    
    def test_core_components_available(self):
        """Test that core components are available from main package."""
        try:
            from black_langcube import BaseGraph, GraphState, LLMNode
            from black_langcube import get_basegraph_classes
            
            # Check that these are callable/classes
            self.assertTrue(callable(BaseGraph))
            self.assertTrue(callable(LLMNode))
            self.assertTrue(callable(get_basegraph_classes))
            
            # Test that get_basegraph_classes returns the expected types
            bg_class, gs_class = get_basegraph_classes()
            self.assertEqual(bg_class, BaseGraph)
            self.assertEqual(gs_class, GraphState)
            
        except ImportError as e:
            self.fail(f"Failed to import core components: {e}")
    
    def test_data_structures_import(self):
        """Test that data structures can be imported."""
        try:
            from black_langcube.data_structures import Strategies, Article, Outline, OutlineItem
            
            # Test basic instantiation
            strategies = Strategies(strategy1="test", strategy2="test2") 
            article = Article(topic="test", language="English")
            
            self.assertEqual(strategies.strategy1, "test")
            self.assertEqual(article.topic, "test")
            
        except ImportError as e:
            self.fail(f"Failed to import data structures: {e}")
    
    def test_process_module_import(self):
        """Test that process module can be imported."""
        try:
            from black_langcube import run_workflow_by_id, run_complete_pipeline
            
            self.assertTrue(callable(run_workflow_by_id))
            self.assertTrue(callable(run_complete_pipeline))
            
        except ImportError as e:
            self.fail(f"Failed to import process module: {e}")
    
    def test_helper_modules_import(self):
        """Test that helper modules can be imported."""
        try:
            from black_langcube.helper_modules.get_basegraph_classes import get_basegraph_classes
            
            # Test that it returns the expected classes
            BaseGraph, GraphState = get_basegraph_classes()
            self.assertTrue(callable(BaseGraph))
            
        except ImportError as e:
            self.fail(f"Failed to import helper modules: {e}")
    
    def test_required_directories_exist(self):
        """Test that required directories exist in the src layout."""
        src_path = Path(__file__).parent.parent / "src" / "black_langcube"
        
        required_dirs = [
            "graf",
            "data_structures", 
            "llm_modules",
            "helper_modules",
            "messages",
            "prompts",
            "examples"
        ]
        
        for dir_name in required_dirs:
            dir_path = src_path / dir_name
            self.assertTrue(dir_path.exists(), f"Required directory {dir_name} does not exist at {dir_path}")
            self.assertTrue(dir_path.is_dir(), f"{dir_name} is not a directory")
    
    def test_required_files_exist(self):
        """Test that required files exist in the project root and src directory."""
        project_root = Path(__file__).parent.parent
        src_path = project_root / "src" / "black_langcube"
        
        # Files in project root
        root_files = [
            "pyproject.toml",
            "README.md",
            "LICENSE",
        ]
        
        for file_name in root_files:
            file_path = project_root / file_name
            self.assertTrue(file_path.exists(), f"Required file {file_name} does not exist at {file_path}")
            self.assertTrue(file_path.is_file(), f"{file_name} is not a file")
        
        # Files in src directory
        src_files = [
            "__init__.py",
            "process.py"
        ]
        
        for file_name in src_files:
            file_path = src_path / file_name
            self.assertTrue(file_path.exists(), f"Required file {file_name} does not exist at {file_path}")
            self.assertTrue(file_path.is_file(), f"{file_name} is not a file")


class TestDataStructures(unittest.TestCase):
    """Test data structure functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from black_langcube.data_structures import Strategies, Article, Outline, OutlineItem
            self.Strategies = Strategies
            self.Article = Article
            self.Outline = Outline
            self.OutlineItem = OutlineItem
        except ImportError:
            self.skipTest("Data structures not available for testing")
    
    def test_strategies_creation(self):
        """Test creating Strategies objects."""
        strategies = self.Strategies(
            strategy1="Search academic papers",
            strategy2="Analyze recent publications"
        )
        
        self.assertEqual(strategies.strategy1, "Search academic papers")
        self.assertEqual(strategies.strategy2, "Analyze recent publications")
    
    def test_article_creation(self):
        """Test creating Article objects.""" 
        article = self.Article(
            topic="AI in Healthcare",
            language="English"
        )
        
        self.assertEqual(article.topic, "AI in Healthcare")
        self.assertEqual(article.language, "English")
    
    def test_outline_creation(self):
        """Test creating Outline and OutlineItem objects."""
        items = [
            self.OutlineItem(foo="Introduction", baz="1", bar="Overview"),
            self.OutlineItem(foo="Methods", baz="2", bar="Methodology")
        ]
        
        outline = self.Outline(items=items)
        
        self.assertEqual(len(outline.items), 2)
        self.assertEqual(outline.items[0].foo, "Introduction")
        self.assertEqual(outline.items[1].foo, "Methods")


class TestCoreComponents(unittest.TestCase):
    """Test core component functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from black_langcube import BaseGraph, GraphState, LLMNode
            self.BaseGraph = BaseGraph
            self.GraphState = GraphState
            self.LLMNode = LLMNode
        except ImportError:
            self.skipTest("Core components not available for testing")
    
    def test_basegraph_instantiation(self):
        """Test that BaseGraph can be instantiated."""
        # Create a simple state class for testing
        class TestState(self.GraphState):
            test_field: str = "test"
        
        # Test BaseGraph instantiation
        graph = self.BaseGraph(
            state_cls=TestState,
            user_message="test message",
            folder_name="test_folder",
            language="English"
        )
        
        self.assertEqual(graph.user_message, "test message")
        self.assertEqual(graph.folder_name, "test_folder")
        self.assertEqual(graph.language, "English")
        self.assertEqual(graph.workflow_name, "base_graph")
    
    def test_graphstate_type_checking(self):
        """Test that GraphState is a valid TypedDict type."""
        # GraphState is a TypedDict, so we test its type properties
        self.assertTrue(hasattr(self.GraphState, '__annotations__'))
        
        # Check that it has the expected fields
        annotations = self.GraphState.__annotations__
        expected_fields = ['messages', 'question_translation', 'folder_name', 'language']
        
        for field in expected_fields:
            self.assertIn(field, annotations)
    
    def test_llmnode_abstract_methods(self):
        """Test that LLMNode requires implementation of abstract methods."""
        # LLMNode should raise NotImplementedError for generate_messages
        node = self.LLMNode(state={}, config={})
        
        with self.assertRaises(NotImplementedError):
            node.generate_messages()
        
        # Test that we can access the basic attributes
        self.assertEqual(node.state, {})
        self.assertEqual(node.config, {})
        self.assertTrue(hasattr(node, 'logger'))


if __name__ == '__main__':
    unittest.main()
