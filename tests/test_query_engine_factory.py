"""
Tests for the query engine factory and service.
"""
import pytest
from unittest.mock import Mock, patch

from app.shared.query_engine_factory import (
    QueryEngineFactory,
    QueryEngineService,
    QueryResult,
    QA_TEMPLATE
)
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


class TestQueryEngineFactory:
    """Test the QueryEngineFactory class."""

    @patch("app.shared.query_engine_factory.HuggingFaceEmbedding")
    @patch("app.shared.query_engine_factory.LlamaCPP")
    @patch("app.shared.query_engine_factory.LanceDBVectorStore")
    @patch("app.shared.query_engine_factory.load_index_from_storage")
    def test_create_query_engine(self, mock_load_index, mock_vector_store, mock_llm, mock_embed):
        """Test that the factory creates a query engine with the correct components."""
        factory = QueryEngineFactory()
        query_engine = factory.create_query_engine()

        assert query_engine is not None
        mock_embed.assert_called_once()
        mock_llm.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_load_index.assert_called_once()


class TestQueryEngineService:
    """Test the QueryEngineService class."""

    @patch("app.shared.query_engine_factory.QueryEngineFactory")
    def test_process_query_success(self, mock_factory):
        """Test a successful query."""
        mock_query_engine = Mock()
        mock_query_engine.query.return_value.response = "Test response"
        mock_query_engine.query.return_value.source_nodes = []
        mock_factory.return_value.create_query_engine.return_value = mock_query_engine

        service = QueryEngineService()
        result = service.process_query("test query", "user1", ["group1"])

        assert result.response == "Test response"
        assert result.error is None

    @patch("app.shared.query_engine_factory.QueryEngineFactory")
    def test_process_query_empty(self, mock_factory):
        """Test an empty query."""
        service = QueryEngineService()
        result = service.process_query(" ", "user1", ["group1"])

        assert result.response == "Please enter a question."
        assert result.error == "Empty query text"

    @patch("app.shared.query_engine_factory.QueryEngineFactory")
    def test_process_query_exception(self, mock_factory):
        """Test a query that raises an exception."""
        mock_factory.return_value.create_query_engine.side_effect = Exception("Test error")

        service = QueryEngineService()
        result = service.process_query("test query", "user1", ["group1"])

        assert "I apologize" in result.response
        assert "Test error" in result.error


class TestSecurityFilters:
    """Test the security filter creation."""

    def test_create_user_filters(self):
        """Test the creation of user-specific metadata filters."""
        filters = create_user_filters("user1", ["group1", "group2"])

        assert isinstance(filters, MetadataFilters)
        assert len(filters.filters) == 3
        assert filters.condition == "or"

        user_filter_found = False
        group1_filter_found = False
        group2_filter_found = False

        for f in filters.filters:
            if isinstance(f, ExactMatchFilter) and f.key == "user_id" and f.value == "user1":
                user_filter_found = True
            if isinstance(f, ExactMatchFilter) and f.key == "group_id" and f.value == "group1":
                group1_filter_found = True
            if isinstance(f, ExactMatchFilter) and f.key == "group_id" and f.value == "group2":
                group2_filter_found = True

        assert user_filter_found
        assert group1_filter_found
        assert group2_filter_found

def create_user_filters(user_id: str, group_ids: list[str]) -> MetadataFilters:
    user_filter = ExactMatchFilter(key="user_id", value=user_id)
    group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
    all_filters = [user_filter] + group_filters
    return MetadataFilters(filters=all_filters, condition="or")
