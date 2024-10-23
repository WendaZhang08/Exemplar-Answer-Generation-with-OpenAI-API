import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from Section2_OpenAI_Integration import OpenAIHandler

class TestOpenAIHandler:
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test generated answer"))
        ]
        mock_response.usage = MagicMock(total_tokens=100)
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)
        return mock_client

    @pytest.fixture
    def handler(self, mock_openai_client):
        """Create a handler instance for testing"""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            handler = OpenAIHandler("test_api_key")
            return handler
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing with multiple examples"""
        return [
            {
                "context": {
                    "question": "What is the first test question?",
                    "task_content": "This is test content 1.",
                    "rubric": {
                        "items": ["High quality", "Medium quality", "Low quality"],
                        "criteria": "Test criteria 1",
                        "total_score": "3"
                    }
                },
                "exemplar_answer": "This is test answer 1."
            },
            {
                "context": {
                    "question": "What is the second test question?",
                    "task_content": "This is test content 2.",
                    "rubric": {
                        "items": ["High quality", "Medium quality", "Low quality"],
                        "criteria": "Test criteria 2",
                        "total_score": "3"
                    }
                },
                "exemplar_answer": "This is test answer 2."
            },
            {
                "context": {
                    "question": "What is the third test question?",
                    "task_content": "This is test content 3.",
                    "rubric": {
                        "items": ["High quality", "Medium quality", "Low quality"],
                        "criteria": "Test criteria 3",
                        "total_score": "3"
                    }
                },
                "exemplar_answer": "This is test answer 3."
            }
        ]

    def test_evaluate_example_quality(self, handler):
        """Test the answer quality evaluation functionality"""
        test_example = {
            'answer': 'This is a comprehensive test answer that meets all criteria.',
            'rubric': json.dumps({
                "items": ["High quality", "Medium quality", "Low quality"],
                "criteria": "Test criteria",
                "total_score": "3"
            }),
            'question': 'Test question?'
        }
        quality_score = handler._evaluate_example_quality(test_example)
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1

    def test_calculate_rubric_alignment(self, handler):
        """Test rubric alignment calculation"""
        test_answer = "This is a high quality answer demonstrating all required criteria."
        test_rubric = {
            "items": ["High quality", "Medium quality", "Low quality"],
            "criteria": "Test criteria",
            "total_score": "3"
        }
        alignment_score = handler._calculate_rubric_alignment(test_answer, test_rubric)
        assert isinstance(alignment_score, float)
        assert 0 <= alignment_score <= 1

    def test_calculate_semantic_similarity(self, handler):
        """Test semantic similarity calculation"""
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        similarity = handler._calculate_semantic_similarity(text1, text2)
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_extract_keywords(self, handler):
        """Test keyword extraction"""
        test_text = "This is a test sentence with important keywords."
        keywords = handler._extract_keywords(test_text)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(k, str) for k in keywords)

    def test_edge_cases(self, handler):
        """Test edge cases"""
        assert handler._extract_keywords("") == []
        assert handler._evaluate_reasoning_depth("") == 0.0
        
        long_text = "test " * 1000
        result = handler._calculate_rubric_alignment(long_text, {
            "items": ["Test"],
            "criteria": "Test",
            "total_score": "1"
        })
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_invalid_input_none(self, handler):
        """Test None input"""
        with pytest.raises((ValueError, TypeError, Exception)):
            handler._evaluate_fold(None)

    def test_invalid_input_empty_string(self, handler):
        """Test empty string input"""
        with pytest.raises((ValueError, TypeError, Exception)):
            handler._evaluate_fold("")

    def test_invalid_input_empty_list(self, handler):
        """Test empty list input"""
        with pytest.raises((ValueError, TypeError, Exception)):
            handler._evaluate_fold([])

    def test_invalid_input_empty_dict(self, handler):
        """Test empty dict input"""
        with pytest.raises((ValueError, TypeError, Exception)):
            handler._evaluate_fold({})

    def test_api_errors(self, handler):
        """Test API error handling"""
        with patch.object(handler, 'client', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                handler.generate_answer({"context": "test"})