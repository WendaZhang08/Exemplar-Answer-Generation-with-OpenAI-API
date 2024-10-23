
import pytest
import numpy as np
from typing import List, Dict
from Section2_OpenAI_Integration import OpenAIHandler

class TestOpenAIHandler:
    @pytest.fixture
    def handler(self):
        """Create a handler instance for testing"""
        return OpenAIHandler("test_api_key")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return [
            {
                "context": {
                    "question": "What is the test question?",
                    "task_content": "This is a test content.",
                    "rubric": {
                        "items": ["High quality", "Medium quality", "Low quality"],
                        "criteria": "Test criteria",
                        "total_score": "3"
                    }
                },
                "exemplar_answer": "This is a test answer."
            }
        ]

    def test_evaluate_example_quality(self, handler):
        """Test the answer quality evaluation functionality"""
        test_example = {
            'answer': 'This is a test answer.',
            'rubric': '{"items": ["High quality", "Medium quality", "Low quality"], "criteria": "Test criteria", "total_score": "3"}',
            'question': 'Test question?'
        }
        quality_score = handler._evaluate_example_quality(test_example)
        assert 0 <= quality_score <= 1
        
    def test_calculate_rubric_alignment(self, handler):
        """Test rubric alignment calculation"""
        test_answer = "This is a high quality answer."
        test_rubric = {
            "items": ["High quality", "Medium quality", "Low quality"],
            "criteria": "Test criteria",
            "total_score": "3"
        }
        alignment_score = handler._calculate_rubric_alignment(test_answer, test_rubric)
        assert 0 <= alignment_score <= 1

    def test_calculate_semantic_similarity(self, handler):
        """Test semantic similarity calculation"""
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        similarity = handler._calculate_semantic_similarity(text1, text2)
        assert 0 <= similarity <= 1

    def test_extract_keywords(self, handler):
        """Test keyword extraction"""
        test_text = "This is a test sentence with important keywords."
        keywords = handler._extract_keywords(test_text)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(k, str) for k in keywords)

    def test_compare_answer_structure(self, handler):
        """Test answer structure comparison"""
        answer1 = "First paragraph.\n\nSecond paragraph."
        answer2 = "First test paragraph.\n\nSecond test paragraph."
        similarity = handler._compare_answer_structure(answer1, answer2)
        assert 0 <= similarity <= 1

    def test_evaluate_reasoning_depth(self, handler):
        """Test reasoning depth evaluation"""
        test_text = "This is because of that reason. Therefore, we can conclude that..."
        depth_score = handler._evaluate_reasoning_depth(test_text)
        assert 0 <= depth_score <= 1

    def test_edge_cases(self, handler):
        """Test edge cases"""
        # Test empty input
        assert handler._extract_keywords("") == []
        assert handler._evaluate_reasoning_depth("") == 0.0
        
        # Test extremely long input
        long_text = "test " * 1000
        assert handler._calculate_rubric_alignment(long_text, {
            "items": ["Test"],
            "criteria": "Test",
            "total_score": "1"
        }) is not None

    def test_cross_validation(self, handler, sample_data):
        """Test cross-validation functionality"""
        results = handler.evaluate_model_performance(sample_data, k_folds=2)
        assert 'mean_metrics' in results
        assert 'std_metrics' in results
        assert 'fold_results' in results

    def test_metric_calculations(self, handler, sample_data):
        """Test metric calculations"""
        metrics = handler._evaluate_fold(sample_data)
        assert 'summary' in metrics
        assert all(k in metrics['summary'] for k in [
            'avg_quality',
            'avg_rubric_alignment',
            'avg_similarity',
            'avg_length_ratio',
            'avg_keyword_coverage',
            'avg_structure_similarity',
            'avg_reasoning_depth'
        ])

    @pytest.mark.parametrize("invalid_input", [None, "", [], {}])
    def test_invalid_inputs(self, handler, invalid_input):
        """Test invalid input handling"""
        with pytest.raises(Exception):
            handler._evaluate_fold(invalid_input)

    def test_api_errors(self, handler):
        """Test API error handling"""
        # Simulate API error
        handler.api_key = "invalid_key"
        with pytest.raises(Exception):
            handler.generate_answer({"context": "test"})

    def test_data_validation(self, handler):
        """Test data validation"""
        invalid_data = [{"missing_key": "value"}]
        with pytest.raises(ValueError):
            handler.evaluate_model_performance(invalid_data)