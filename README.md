# Local RAG

A simple experiment running a RAG (Retrieval-Augmented Generation) system locally using LlamaIndex.

## Overview

This project demonstrates a basic RAG implementation that:
- Uses Ollama with the phi4 model for text generation
- Uses HuggingFace's Snowflake/snowflake-arctic-embed-l-v2.0 for embeddings
- Stores embeddings locally for persistence
- Provides timing metrics for embedding and query operations
- Shows relevance scores for retrieved documents

## Prerequisites

- [Bun](https://bun.sh/) JavaScript runtime
- [Ollama](https://ollama.ai/) with the phi4 model installed