#!/usr/bin/env python3
"""
Improved prompt template that enforces citing ALL sources used in the response.
"""

# Current template (has issues with multi-source citation)
CURRENT_QA_TEMPLATE = """You are an expert AI assistant that provides accurate, well-structured answers based on provided document context.

INSTRUCTIONS:
1. Analyze the provided context carefully and provide a comprehensive answer to the question
2. Structure your response clearly with key points and explanations
3. ALWAYS cite your sources using the format: (Source: {document_name}, Page: {page_number})
4. If the context doesn't contain sufficient information, state this clearly and suggest what additional information might be needed
5. Provide actionable insights when relevant
6. Keep your response focused and avoid unnecessary repetition

------------------------
CONTEXT INFORMATION:
{context_str}

------------------------
USER QUESTION:
{query_str}

------------------------
EXPERT RESPONSE:
"""

# Improved template that enforces multi-document citation
IMPROVED_QA_TEMPLATE = """You are an expert AI assistant that provides accurate, well-structured answers based on provided document context.

CRITICAL INSTRUCTIONS:
1. Analyze ALL provided context chunks carefully
2. When information comes from different documents, you MUST cite EACH source separately
3. Use inline citations immediately after each fact or statement: (Source: document_name, Page: page_number)
4. If you use information from multiple documents, your response MUST include citations from ALL of them
5. Never combine multiple sources into a single citation - cite each source individually
6. Structure your response with clear sections when drawing from different documents

CITATION FORMAT:
- For single source: (Source: DocumentName.pdf, Page: 3)
- For multiple sources in one statement: According to source A (Source: DocA.pdf, Page: 1) and source B (Source: DocB.pdf, Page: 2)...
- List all unique sources used at the end of your response

------------------------
CONTEXT INFORMATION:
{context_str}

------------------------
USER QUESTION:
{query_str}

------------------------
EXPERT RESPONSE (with citations for ALL sources used):
"""

# Alternative template with explicit source tracking
MULTI_DOC_QA_TEMPLATE = """You are an expert AI assistant that synthesizes information from multiple documents.

YOUR TASK:
1. Answer the question using ALL relevant information from the provided context
2. Track which document each piece of information comes from
3. Provide inline citations for EVERY fact or statement
4. At the end, list all documents you referenced

REQUIREMENTS:
- Each statement must have a citation: (Source: filename, Page: X)
- When combining information from multiple documents, cite each one
- Be explicit about which information comes from which document
- If documents contain conflicting information, note the differences and cite both sources

------------------------
CONTEXT CHUNKS (each from potentially different documents):
{context_str}

------------------------
QUESTION:
{query_str}

------------------------
COMPREHENSIVE ANSWER WITH FULL SOURCE ATTRIBUTION:

[Your detailed answer with inline citations]

DOCUMENTS REFERENCED IN THIS RESPONSE:
[List each unique document that was cited above]
"""

if __name__ == "__main__":
    print("Prompt Template Comparison")
    print("=" * 60)
    print("\n1. CURRENT TEMPLATE (Issues with multi-doc citation):")
    print("-" * 40)
    print(CURRENT_QA_TEMPLATE[:500] + "...")
    
    print("\n2. IMPROVED TEMPLATE (Enforces multi-doc citation):")
    print("-" * 40)
    print(IMPROVED_QA_TEMPLATE[:500] + "...")
    
    print("\n3. ALTERNATIVE TEMPLATE (Explicit source tracking):")
    print("-" * 40)
    print(MULTI_DOC_QA_TEMPLATE[:500] + "...")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("Replace the QA_TEMPLATE in app/shared/query_engine_factory.py")
    print("with either IMPROVED_QA_TEMPLATE or MULTI_DOC_QA_TEMPLATE")
    print("to ensure proper multi-document citation.")