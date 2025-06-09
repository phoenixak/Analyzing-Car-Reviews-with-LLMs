"""
Tests for the data processor module.
"""

import unittest
import sys
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class."""
    
    def setUp(self):
        """Set up test data processor and sample data."""
        self.processor = DataProcessor()
        self.sample_reviews = [
            "Great car with excellent performance.",
            "Poor fuel economy and high costs.",
            "Amazing design but lacking safety features."
        ]
        self.sample_labels = ["positive", "negative", "mixed"]
        
    def test_processor_initialization(self):
        """Test data processor initialization."""
        self.assertIsInstance(self.processor, DataProcessor)
    
    def test_load_csv_data(self):
        """Test loading CSV data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create sample CSV
            df = pd.DataFrame({
                'review': self.sample_reviews,
                'label': self.sample_labels
            })
            df.to_csv(f.name, index=False)
            
            try:
                reviews, labels = self.processor.load_data(Path(f.name))
                
                self.assertEqual(len(reviews), 3)
                self.assertEqual(len(labels), 3)
                self.assertEqual(reviews[0], "Great car with excellent performance.")
                self.assertEqual(labels[0], "positive")
            finally:
                os.unlink(f.name)
    
    def test_load_json_data(self):
        """Test loading JSON data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create sample JSON
            data = {
                'reviews': [
                    {'text': rev, 'label': label} 
                    for rev, label in zip(self.sample_reviews, self.sample_labels)
                ]
            }
            json.dump(data, f)
            
            try:
                reviews, labels = self.processor.load_data(Path(f.name))
                
                self.assertEqual(len(reviews), 3)
                self.assertEqual(len(labels), 3)
                self.assertEqual(reviews[0], "Great car with excellent performance.")
                self.assertEqual(labels[0], "positive")
            finally:
                os.unlink(f.name)
    
    def test_load_txt_data(self):
        """Test loading TXT data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create sample TXT
            f.write('\n'.join(self.sample_reviews))
            
            try:
                reviews, labels = self.processor.load_data(Path(f.name))
                
                self.assertEqual(len(reviews), 3)
                self.assertEqual(len(labels), 0)  # No labels in txt files
                self.assertEqual(reviews[0], "Great car with excellent performance.")
            finally:
                os.unlink(f.name)
    
    def test_load_excel_data(self):
        """Test loading Excel data."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            # Create sample Excel
            df = pd.DataFrame({
                'review': self.sample_reviews,
                'label': self.sample_labels
            })
            df.to_excel(f.name, index=False)
            
            try:
                reviews, labels = self.processor.load_data(Path(f.name))
                
                self.assertEqual(len(reviews), 3)
                self.assertEqual(len(labels), 3)
                self.assertEqual(reviews[0], "Great car with excellent performance.")
                self.assertEqual(labels[0], "positive")
            finally:
                os.unlink(f.name)
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write("test content")
            
            try:
                with self.assertRaises(ValueError):
                    self.processor.load_data(Path(f.name))
            finally:
                os.unlink(f.name)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_data(Path("/nonexistent/file.csv"))
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        dirty_text = "  This is a GREAT car!!!  It's amazing... 😀 "
        processed = self.processor.preprocess_text(dirty_text)
        
        # Should remove extra whitespace, handle punctuation, etc.
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) <= len(dirty_text))
        self.assertNotEqual(processed, dirty_text)  # Should be different after processing
    
    def test_preprocess_empty_text(self):
        """Test preprocessing empty text."""
        processed = self.processor.preprocess_text("")
        self.assertEqual(processed, "")
        
        processed = self.processor.preprocess_text(None)
        self.assertEqual(processed, "")
    
    def test_validate_data_format(self):
        """Test data format validation."""
        # Valid data
        valid_reviews = ["Review 1", "Review 2"]
        valid_labels = ["positive", "negative"]
        
        try:
            self.processor.validate_data_format(valid_reviews, valid_labels)
        except Exception:
            self.fail("Valid data should not raise exception")
        
        # Invalid data - mismatched lengths
        invalid_labels = ["positive"]
        with self.assertRaises(ValueError):
            self.processor.validate_data_format(valid_reviews, invalid_labels)
        
        # Invalid data - empty reviews
        with self.assertRaises(ValueError):
            self.processor.validate_data_format([], [])
    
    def test_export_results_json(self):
        """Test exporting results to JSON."""
        results = {
            "task": "test",
            "data": {"key": "value"},
            "metrics": {"accuracy": 0.85}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            try:
                export_path = self.processor.export_results(results, Path(f.name))
                
                # Verify file exists and contains correct data
                self.assertTrue(os.path.exists(export_path))
                with open(export_path, 'r') as exported_file:
                    loaded_data = json.load(exported_file)
                    self.assertEqual(loaded_data, results)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)
    
    def test_export_results_csv(self):
        """Test exporting results to CSV."""
        results = {
            "predictions": [
                {"text": "review 1", "label": "positive", "score": 0.9},
                {"text": "review 2", "label": "negative", "score": 0.8}
            ]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            try:
                export_path = self.processor.export_results(results, Path(f.name))
                
                # Verify file exists and can be read
                self.assertTrue(os.path.exists(export_path))
                df = pd.read_csv(export_path)
                self.assertEqual(len(df), 2)
                self.assertIn("text", df.columns)
                self.assertIn("label", df.columns)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)
    
    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.processor.get_supported_formats()
        
        self.assertIsInstance(formats, list)
        self.assertIn('.csv', formats)
        self.assertIn('.json', formats)
        self.assertIn('.txt', formats)
        self.assertIn('.xlsx', formats)
    
    def test_batch_process_reviews(self):
        """Test batch processing of reviews."""
        def mock_processor(text):
            return text.upper()
        
        results = self.processor.batch_process(self.sample_reviews, mock_processor)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "GREAT CAR WITH EXCELLENT PERFORMANCE.")
    
    def test_batch_process_with_progress(self):
        """Test batch processing with progress tracking."""
        def slow_processor(text):
            import time
            time.sleep(0.01)  # Simulate processing time
            return len(text)
        
        results = self.processor.batch_process(
            self.sample_reviews, 
            slow_processor, 
            show_progress=True,
            chunk_size=2
        )
        
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], int)
    
    def test_filter_by_length(self):
        """Test filtering reviews by length."""
        mixed_reviews = [
            "Short",
            "This is a medium length review with some details.",
            "Very very very long review with lots of details and information about the car experience that goes on and on and on."
        ]
        
        # Filter by minimum length
        filtered = self.processor.filter_by_length(mixed_reviews, min_length=20)
        self.assertEqual(len(filtered), 2)  # Should exclude "Short"
        
        # Filter by maximum length
        filtered = self.processor.filter_by_length(mixed_reviews, max_length=50)
        self.assertEqual(len(filtered), 2)  # Should exclude very long review
        
        # Filter by both
        filtered = self.processor.filter_by_length(mixed_reviews, min_length=20, max_length=60)
        self.assertEqual(len(filtered), 1)  # Should only include medium review
    
    def test_detect_language(self):
        """Test language detection."""
        english_text = "This is a great car with excellent performance."
        spanish_text = "Este es un gran coche con excelente rendimiento."
        
        eng_lang = self.processor.detect_language(english_text)
        spa_lang = self.processor.detect_language(spanish_text)
        
        # Should detect different languages (exact language codes may vary)
        self.assertNotEqual(eng_lang, spa_lang)
        self.assertIsInstance(eng_lang, str)
        self.assertIsInstance(spa_lang, str)
    
    def test_error_handling_corrupted_data(self):
        """Test error handling with corrupted data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create invalid JSON
            f.write('{"invalid": json content}')
            
            try:
                with self.assertRaises((json.JSONDecodeError, ValueError)):
                    self.processor.load_data(Path(f.name))
            finally:
                os.unlink(f.name)


class TestDataProcessorPerformance(unittest.TestCase):
    """Performance tests for data processor."""
    
    def setUp(self):
        """Set up performance test data."""
        self.processor = DataProcessor()
        self.large_dataset = [f"Review number {i} with some content." for i in range(1000)]
    
    def test_large_dataset_processing(self):
        """Test processing large datasets."""
        import time
        
        def simple_processor(text):
            return len(text)
        
        start_time = time.time()
        results = self.processor.batch_process(
            self.large_dataset, 
            simple_processor,
            chunk_size=100
        )
        processing_time = time.time() - start_time
        
        self.assertEqual(len(results), 1000)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # This test ensures processing doesn't consume excessive memory
        def memory_intensive_processor(text):
            # Create a moderately sized object
            return [text] * 10
        
        try:
            results = self.processor.batch_process(
                self.large_dataset[:100],  # Use smaller subset for memory test
                memory_intensive_processor,
                chunk_size=10
            )
            self.assertEqual(len(results), 100)
        except MemoryError:
            self.fail("Processing should not cause memory errors")


if __name__ == "__main__":
    unittest.main(verbosity=2)