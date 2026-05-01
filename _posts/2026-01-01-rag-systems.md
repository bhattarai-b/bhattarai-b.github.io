---
title: Designing Retrieval Augmented Generation (RAG) Systems
author: Bibek Bhattarai
date: 2026-01-01 1:42:00 +0800
categories: [Blogging, Tutorial]
tags: [writing]
render_with_liquid: false
---
Retrieval Augmented Generation (RAG) is one of the most popular techniques to enhance the performance of Large Language Models (LLMs) models by injecting external data in the context. Like the name suggests, it involves **retrieving** the data from external source outside of LLMs pre-training data, **augmenting** the prompt with retrieved information, and using this information-rich prompt to enhance the quality of response **generated** by LLM. Here are the typical scenarios where RAG is a right choice to enhance the performance of your models.

1. **Accessing Dynamic or Private Knowledge**: LLMs training data have freshness cutoff. All the informations that became available after the pre-training is done, or the private/proprietery informations that wasn't available for training will leave LLMs responses to become sub-par for some problems. In order to fix that, we can access the missing knowledge from external storage, and supply to LLM in prompt. This allows LLM to generate better responses. 

2. **Role-Based Access Control (RBAC)**: For almost every organizations that want to make use of AI in their workflow, access control is a major headache. If we have one model, but 1,000 users with different permission levels (e.g., the CEO can see "Q3 Layoff Plans," the intern cannot). If you use techniques like fine-tuning the model on private data, that data is "baked" into the weights. You cannot tell a neural network: "Don't remember this specific weight if the user is an intern."

You fix this by handling the permissions at the Retrieval Layer. The Retriever checks the user's ID, filters the documents they are allowed to see, and only then passes them to the LLM. 

3. **Data Freshness & Deletion (The "Right to be Forgotten")**: Imagine a scenario where the user updates a document or if customer delete their information. Data protection policies like GDPR requires the data to be deleted. But if we fine-tune the model on this data, the only way of forgetting the deleted entry is to re-run the costly fine-tuning process( catastrophic-forgetting). With RAG, you simply delete the vector from the database. The next time the system queries, the data is gone.

4. **Hallucination & Auditability** In a lot of areas, e.g., finance, or law, it is critical to provide the justification for each decision made by AI. If we make decision simply from the pre-trained/fine-tuned model, it will generate the justification as well(better yet will make up the citations). With RAGs, you have receipts. You can cite the chunks of the informations used to make certain decisions, thereby making it easier to examine why each decision was made.


RAG System Engineering Concept Map
Phase 1: Indexing Pipeline (Data Prep)

Preparing the "Knowledge Base" before any user asks a question.

    1. Data Ingestion

        Loaders: Handling PDFs, HTML, JSON, Markdown.

        Cleaning: Removing headers/footers, normalizing text (Unicode, whitespace).

        Metadata Extraction: Tagging year, author, source (crucial for filtering later).

    2. Chunking Strategy (The Input Control)

        Fixed-Size: Token-based vs. Character-based.

        Content-Aware: Sentence splitting, recursive splitting, Markdown headers.

        Semantic Chunking: Splitting based on embedding similarity (topic shifts).

        Small-to-Big (Parent-Child): Indexing sentences but retrieving paragraphs.

    3. The Embedding "Black Box"

        Selection Criteria: MTEB Leaderboard, Context Length (512 vs 8k), Dimensions.

        Domain Specifics: Instruction-tuned (BGE) vs. General (OpenAI).

        Fine-tuning (Optional): Adapting to niche vocabulary (Medical/Legal).

    4. Vector Database (The Storage)

        Index Types: HNSW (Speed) vs. IVFFlat (Memory).

        Hybrid Storage: Storing Sparse Vectors (Keywords) alongside Dense Vectors (Embeddings).

Phase 2: Retrieval Pipeline (The Engine)

What happens when the user hits "Enter".

    5. Query Pre-Processing

        Query Rewriting: Clarifying vague queries.

        HyDE (Hypothetical Document Embeddings): Generating a fake answer to search with.

        Query Decomposition: Breaking complex questions into sub-queries.

        Multi-Query: Generating synonyms to cast a wider net.

    6. Search Execution

        Dense Retrieval: Vector similarity search (Semantic match).

        Sparse Retrieval: BM25/Splade (Keyword match).

        Hybrid Search: Combining Dense + Sparse (The industry standard).

        Metadata Filtering: Pre-filtering by date/author before searching.

    7. Reranking (The Precision Layer)

        Cross-Encoders: Scoring the top 50 results with a high-precision model (e.g., Cohere/BGE-Reranker).

        Deduplication: Removing near-identical chunks.

Phase 3: Generation Pipeline (The Output)

Synthesizing the answer.

    8. Context Construction

        Prompt Engineering: The System Prompt ("You are a helpful assistant...").

        Context Window Management: Fitting retrieved chunks into the limit.

        Citation Injection: inserting [1] references back to source documents.

    9. LLM Inference

        Model Selection: Reasoning capability vs. Cost/Speed.

        Output Structuring: Forcing JSON output (if needed).

Phase 4: Advanced Orchestration (The Brain)

Moving beyond linear chains to dynamic loops.

    10. Iterative & Agentic RAG

        Router/Classifier: Deciding if retrieval is even needed.

        Self-Correction: LLM checking if the retrieved data actually answers the question.

        Tool Use: Calling a SQL DB or API instead of a Vector DB.

        GraphRAG: Using Knowledge Graphs to find relationships across documents.

Phase 5: Evaluation & Ops (The Ruler)

How do we know it's working?

    11. RAG Metrics

        Retrieval Metrics: Precision@K, Recall@K, MMR (Maximal Marginal Relevance).

        Generation Metrics: Faithfulness (did it hallucinate?), Answer Relevance.

        Frameworks: RAGAS, TruLens, Arize Phoenix.

<https://www.reddit.com/r/LangChain/comments/1bqn1sj/comment/kx4qvmd/?share_id=-NIMzng1Ksu5Fc2-1p1VY&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1>
<https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents>
<https://arxiv.org/pdf/2406.04369>
<https://arxiv.org/pdf/2312.10997v1>

