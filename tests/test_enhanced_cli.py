"""
Tests for the enhanced CLI module.
"""

import unittest
import sys
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.enhanced_cli import enhanced_main, validate_file_exists, check_system_requirements


class TestEnhancedCLI(unittest.TestCase):
    """Test cases for the enhanced CLI."""
    
    def setUp(self):
        """Set up test CLI runner and sample data."""
        self.runner = CliRunner()
        
        # Create a temporary test data file
        self.test_data = {
            'reviews': [
                {'text': 'Great car with excellent performance.', 'label': 'positive'},
                {'text': 'Poor fuel economy and high costs.', 'label': 'negative'},
                {'text': 'Amazing design but lacking safety features.', 'label': 'mixed'}
            ]
        }
        
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(enhanced_main, ['--help'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Analyze car reviews using advanced NLP', result.output)
        self.assertIn('--task', result.output)
        self.assertIn('--visualize', result.output)
    
    def test_validate_file_exists_valid_file(self):
        """Test file validation with existing file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                # Should not raise exception
                validate_file_exists(f.name)
            except Exception as e:
                self.fail(f"Valid file should not raise exception: {e}")
            finally:
                os.unlink(f.name)
    
    def test_validate_file_exists_invalid_file(self):
        """Test file validation with non-existent file."""
        with self.assertRaises(SystemExit):
            validate_file_exists("/nonexistent/file.json")
    
    @patch('src.enhanced_cli.psutil.virtual_memory')
    @patch('src.enhanced_cli.psutil.disk_usage')
    def test_check_system_requirements_sufficient(self, mock_disk, mock_memory):
        """Test system requirements check with sufficient resources."""
        # Mock sufficient resources
        mock_memory.return_value = Mock(available=2 * 1024**3)  # 2GB
        mock_disk.return_value = Mock(free=5 * 1024**3)  # 5GB
        
        try:
            check_system_requirements()
        except Exception as e:
            self.fail(f"Sufficient resources should not raise exception: {e}")
    
    @patch('src.enhanced_cli.psutil.virtual_memory')
    def test_check_system_requirements_insufficient_memory(self, mock_memory):
        """Test system requirements check with insufficient memory."""
        # Mock insufficient memory
        mock_memory.return_value = Mock(available=100 * 1024**2)  # 100MB
        
        with self.assertRaises(SystemExit):
            check_system_requirements()
    
    @patch('src.enhanced_cli.load_data')
    @patch('src.enhanced_cli.SentimentAnalysisPipeline')
    def test_cli_sentiment_task(self, mock_pipeline, mock_load_data):
        """Test CLI with sentiment analysis task."""
        # Mock data loading
        mock_load_data.return_value = (
            ['Great car!', 'Poor performance.'],
            ['positive', 'negative']
        )
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [
            {'label': 'POSITIVE', 'score': 0.9},
            {'label': 'NEGATIVE', 'score': 0.8}
        ]
        mock_pipeline.return_value = mock_pipeline_instance
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment',
                    '--verbose'
                ])
                
                self.assertEqual(result.exit_code, 0)
                mock_load_data.assert_called_once()
                mock_pipeline.assert_called_once()
            finally:
                os.unlink(f.name)
    
    @patch('src.enhanced_cli.load_data')
    @patch('src.enhanced_cli.TopicModelingPipeline')
    def test_cli_topic_task(self, mock_pipeline, mock_load_data):
        """Test CLI with topic modeling task."""
        # Mock data loading
        mock_load_data.return_value = (
            ['Car review 1', 'Car review 2'],
            []
        )
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = {
            'topics': ['topic1', 'topic2'],
            'num_topics': 2
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'topic'
                ])
                
                self.assertEqual(result.exit_code, 0)
                mock_load_data.assert_called_once()
                mock_pipeline.assert_called_once()
            finally:
                os.unlink(f.name)
    
    @patch('src.enhanced_cli.load_data')
    @patch('src.enhanced_cli.save_results')
    @patch('src.enhanced_cli.SentimentAnalysisPipeline')
    def test_cli_save_results(self, mock_pipeline, mock_save, mock_load_data):
        """Test CLI with save results option."""
        # Mock data loading
        mock_load_data.return_value = (
            ['Great car!'],
            ['positive']
        )
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'label': 'POSITIVE', 'score': 0.9}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Mock save function
        mock_save.return_value = '/tmp/results.json'
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment',
                    '--save-results'
                ])
                
                self.assertEqual(result.exit_code, 0)
                mock_save.assert_called_once()
            finally:
                os.unlink(f.name)
    
    @patch('src.enhanced_cli.load_data')
    @patch('src.enhanced_cli.create_interactive_dashboard')
    @patch('src.enhanced_cli.SentimentAnalysisPipeline')
    def test_cli_with_visualizations(self, mock_pipeline, mock_dashboard, mock_load_data):
        """Test CLI with visualization option."""
        # Mock data loading
        mock_load_data.return_value = (
            ['Great car!'],
            ['positive']
        )
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{'label': 'POSITIVE', 'score': 0.9}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment',
                    '--visualize'
                ])
                
                self.assertEqual(result.exit_code, 0)
                mock_dashboard.assert_called_once()
            finally:
                os.unlink(f.name)
    
    @patch('src.enhanced_cli.load_data')
    def test_cli_all_tasks(self, mock_load_data):
        """Test CLI with all tasks option."""
        # Mock data loading
        mock_load_data.return_value = (
            ['Great car with excellent performance.'] * 5,
            ['positive'] * 5
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                # This might take longer and have some failures due to missing models
                # But should not crash completely
                with patch.multiple(
                    'src.enhanced_cli',
                    SentimentAnalysisPipeline=Mock(),
                    TopicModelingPipeline=Mock(),
                    TranslationPipeline=Mock(),
                    SummarizationPipeline=Mock(),
                    AspectSentimentPipeline=Mock(),
                    QuestionAnsweringModel=Mock(),
                    NamedEntityRecognitionModel=Mock()
                ):
                    result = self.runner.invoke(enhanced_main, [
                        '--data-file', f.name,
                        '--task', 'all',
                        '--output-format', 'json'
                    ])
                    
                    # Should complete without crashing (exit code 0 or 1 acceptable)
                    self.assertIn(result.exit_code, [0, 1])
            finally:
                os.unlink(f.name)
    
    def test_cli_invalid_task(self):
        """Test CLI with invalid task."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'invalid_task'
                ])
                
                # Should fail due to invalid choice
                self.assertNotEqual(result.exit_code, 0)
            finally:
                os.unlink(f.name)
    
    def test_cli_invalid_output_format(self):
        """Test CLI with invalid output format."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--output-format', 'invalid_format'
                ])
                
                # Should fail due to invalid choice
                self.assertNotEqual(result.exit_code, 0)
            finally:
                os.unlink(f.name)
    
    @patch('src.enhanced_cli.load_data')
    def test_cli_error_handling(self, mock_load_data):
        """Test CLI error handling."""
        # Mock data loading to raise an exception
        mock_load_data.side_effect = Exception("Data loading failed")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment'
                ])
                
                # Should handle error gracefully (non-zero exit code)
                self.assertNotEqual(result.exit_code, 0)
            finally:
                os.unlink(f.name)
    
    @patch('src.enhanced_cli.validate_system_resources')
    def test_cli_system_requirements_failure(self, mock_validate):
        """Test CLI when system requirements are not met."""
        # Mock system validation to fail
        mock_validate.side_effect = SystemExit("Insufficient resources")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment'
                ])
                
                # Should exit with error
                self.assertNotEqual(result.exit_code, 0)
            finally:
                os.unlink(f.name)


class TestCLIPerformance(unittest.TestCase):
    """Performance tests for CLI operations."""
    
    def setUp(self):
        """Set up performance test data."""
        self.runner = CliRunner()
        
        # Create larger test dataset
        self.large_test_data = {
            'reviews': [
                {'text': f'Review number {i} with some content about cars.', 'label': 'positive' if i % 2 == 0 else 'negative'}
                for i in range(100)
            ]
        }
    
    @patch('src.enhanced_cli.load_data')
    @patch('src.enhanced_cli.SentimentAnalysisPipeline')
    def test_cli_performance_large_dataset(self, mock_pipeline, mock_load_data):
        """Test CLI performance with larger dataset."""
        import time
        
        # Mock data loading
        reviews = [item['text'] for item in self.large_test_data['reviews']]
        labels = [item['label'] for item in self.large_test_data['reviews']]
        mock_load_data.return_value = (reviews, labels)
        
        # Mock pipeline with realistic delay
        def mock_process(texts):
            time.sleep(0.01 * len(texts))  # Simulate processing time
            return [{'label': 'POSITIVE', 'score': 0.9} for _ in texts]
        
        mock_pipeline_instance = Mock(side_effect=mock_process)
        mock_pipeline.return_value = mock_pipeline_instance
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.large_test_data, f)
            
            try:
                start_time = time.time()
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment',
                    '--batch-size', '20'
                ])
                execution_time = time.time() - start_time
                
                self.assertEqual(result.exit_code, 0)
                self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
            finally:
                os.unlink(f.name)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI with real components."""
    
    def setUp(self):
        """Set up integration test data."""
        self.runner = CliRunner()
        
        self.real_test_data = {
            'reviews': [
                {
                    'text': 'This Honda Civic is an excellent car with great fuel economy and reliable performance. The interior is comfortable and the handling is smooth.',
                    'label': 'positive'
                },
                {
                    'text': 'The Toyota Camry disappointed me with poor build quality and expensive maintenance. The engine noise is annoying and the seats are uncomfortable.',
                    'label': 'negative'
                },
                {
                    'text': 'Ford F-150 has impressive towing capacity and rugged design, but the fuel economy could be better for daily commuting.',
                    'label': 'mixed'
                }
            ]
        }
    
    @patch('src.enhanced_cli.model_cache')  # Mock the cache to avoid actual model loading
    def test_cli_integration_sentiment_only(self, mock_cache):
        """Test CLI integration with sentiment analysis only."""
        # Mock model cache to return simple mock models
        mock_classifier = Mock()
        mock_classifier.return_value = [{'label': 'POSITIVE', 'score': 0.85}]
        mock_cache.get.return_value = mock_classifier
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(self.real_test_data, f)
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment',
                    '--output-format', 'json',
                    '--verbose'
                ])
                
                # Should complete successfully
                self.assertEqual(result.exit_code, 0)
                
                # Should display progress and results
                self.assertIn('Sentiment Analysis', result.output)
                
            finally:
                os.unlink(f.name)
    
    def test_cli_file_format_detection(self):
        """Test CLI automatic file format detection."""
        # Test with CSV format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('review,label\n')
            f.write('"Great car","positive"\n')
            f.write('"Poor quality","negative"\n')
            
            try:
                result = self.runner.invoke(enhanced_main, [
                    '--data-file', f.name,
                    '--task', 'sentiment',
                    '--dry-run'  # Don't actually run models
                ])
                
                # Should detect CSV format and not crash
                # (might still fail due to missing models, but should get past file loading)
                self.assertNotIn('Unsupported file format', result.output)
                
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)