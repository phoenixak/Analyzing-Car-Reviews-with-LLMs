"""
Tests for the model classes.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import QuestionAnsweringModel


class TestModels(unittest.TestCase):
    """Test cases for the model classes."""

    def setUp(self):
        """Set up test data."""
        self.test_context = "The 2023 Tesla Model 3 has excellent acceleration and handling. The battery range is about 350 miles on a full charge. However, some users complained about the build quality and the minimalist interior design."
        self.test_question = "What is the battery range of the Tesla Model 3?"

    def test_question_answering_model(self):
        """Test the question answering model."""
        try:
            model = QuestionAnsweringModel()
            answer = model(self.test_context, self.test_question)

            # Check that the answer is a non-empty string
            self.assertIsInstance(answer, str)
            self.assertTrue(len(answer) > 0)

            # Check that the answer contains relevant information
            self.assertTrue(
                any(word in answer.lower() for word in ["350", "miles", "range"])
            )
        except Exception as e:
            self.fail(f"Question answering model raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
