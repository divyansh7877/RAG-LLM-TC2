#!/usr/bin/env python3
"""
Test script to verify that the reranker and query system can:
1. Retrieve chunks from multiple documents
2. Properly cite multiple sources in the response
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.shared.query_engine_factory import QueryEngineFactory
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def test_multi_document_retrieval():
    """Test if the system retrieves and cites multiple documents."""
    
    print("=" * 60)
    print("Multi-Document Retrieval Test")
    print("=" * 60)
    
    # Initialize the query engine factory
    factory = QueryEngineFactory()
    
    # Test configuration
    test_user_id = "assistant1"  # Using the existing user from the data
    test_group_ids = ["common_rules"]  # Using the existing group
    
    # Create a query that should pull from multiple documents
    test_queries = [
        "What information is available about Divyansh Agarwal's experience and skills?",
        "Tell me about all the documents you have access to",
        "Summarize the key points from all available documents"
    ]
    
    for query_idx, query_text in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query #{query_idx}: {query_text}")
        print("=" * 60)
        
        try:
            # Create user filters
            user_filter = ExactMatchFilter(key="user_id", value=test_user_id)
            group_filters = [ExactMatchFilter(key="group_id", value=gid) for gid in test_group_ids]
            filters = MetadataFilters(filters=[user_filter] + group_filters, condition="or")
            
            # Create query engine
            print("\nCreating query engine...")
            engine = factory.create_query_engine(
                user_filters=filters,
                user_id=test_user_id,
                group_ids=test_group_ids
            )
            
            # Execute query
            print("Executing query...")
            response = engine.query(query_text)
            
            # Analyze the response
            print("\n" + "-" * 40)
            print("RESPONSE ANALYSIS")
            print("-" * 40)
            
            # Check source nodes
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\nNumber of source nodes retrieved: {len(response.source_nodes)}")
                
                # Collect unique documents
                unique_docs = set()
                doc_page_map = {}
                
                for i, node in enumerate(response.source_nodes, 1):
                    if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                        metadata = node.node.metadata
                        doc_name = metadata.get('document_name', 'Unknown')
                        page_num = metadata.get('page_number', 'Unknown')
                        score = node.score if hasattr(node, 'score') else 0
                        
                        unique_docs.add(doc_name)
                        
                        if doc_name not in doc_page_map:
                            doc_page_map[doc_name] = []
                        doc_page_map[doc_name].append((page_num, score))
                        
                        print(f"\n  Source #{i}:")
                        print(f"    Document: {doc_name}")
                        print(f"    Page: {page_num}")
                        print(f"    Score: {score:.4f}" if score else "    Score: N/A")
                        print(f"    Text snippet: {node.node.text[:100]}...")
                
                print(f"\n{'='*40}")
                print(f"UNIQUE DOCUMENTS RETRIEVED: {len(unique_docs)}")
                print(f"{'='*40}")
                for doc in unique_docs:
                    pages = doc_page_map[doc]
                    print(f"  - {doc}")
                    print(f"    Pages: {[p[0] for p in pages]}")
                    print(f"    Scores: {[f'{p[1]:.3f}' for p in pages]}")
                
                # Check if multiple documents were retrieved
                if len(unique_docs) > 1:
                    print(f"\n✓ SUCCESS: Retrieved chunks from {len(unique_docs)} different documents!")
                else:
                    print(f"\n⚠ WARNING: Only retrieved chunks from 1 document")
            else:
                print("\n❌ ERROR: No source nodes found in response")
            
            # Check the actual response text for citations
            print("\n" + "-" * 40)
            print("RESPONSE TEXT ANALYSIS")
            print("-" * 40)
            
            response_text = str(response.response) if hasattr(response, 'response') else str(response)
            print(f"\nResponse length: {len(response_text)} characters")
            
            # Look for citations in the response
            import re
            citation_pattern = r'\(Source:.*?(?:,\s*Page:.*?)?\)'
            citations = re.findall(citation_pattern, response_text)
            
            if citations:
                print(f"\nFound {len(citations)} citations in the response:")
                unique_cited_docs = set()
                for citation in citations:
                    print(f"  - {citation}")
                    # Extract document name from citation
                    doc_match = re.search(r'Source:\s*([^,\)]+)', citation)
                    if doc_match:
                        unique_cited_docs.add(doc_match.group(1).strip())
                
                print(f"\nUnique documents cited: {len(unique_cited_docs)}")
                for doc in unique_cited_docs:
                    print(f"  - {doc}")
                    
                if len(unique_cited_docs) > 1:
                    print(f"\n✓ SUCCESS: Response cites {len(unique_cited_docs)} different documents!")
                else:
                    print(f"\n⚠ WARNING: Response only cites 1 document")
            else:
                print("\n⚠ WARNING: No citations found in the response text")
                print("The response might not be following the citation format.")
            
            # Show first part of the actual response
            print("\n" + "-" * 40)
            print("RESPONSE TEXT (first 500 chars):")
            print("-" * 40)
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    # Summary of configuration
    print("\nCurrent Configuration:")
    print(f"  - SIM_TOP_K (initial retrieval): {factory.logger.info if hasattr(factory, '_config_logged') else 'Check logs'}")
    print(f"  - FINAL_K (after reranking): Check logs")
    print(f"  - RERANK_ENABLED: Check logs")
    
    from app.shared.openai_config import get_retrieval_config
    retrieval_config = get_retrieval_config()
    print("\nRetrieval Configuration from OpenAI config:")
    for key, value in retrieval_config.items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    test_multi_document_retrieval()