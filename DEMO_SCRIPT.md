# RAG Pipeline Demo Script

This document contains hand-crafted test queries to demonstrate the core capabilities of the RAG pipeline. Keep this file open on a second monitor or side-by-side to easily copy-paste these queries while recording your demo GIF/Video.

---

## 1. Factoid Retrieval (Basic Test)
*Demonstrates the base capability of the system to accurately extract information.*

**Copy this:**
```text
In what year did the Normans conquer England?
```
**Expected Answer:** `1066`

---

## 2. Extreme Typo Robustness (The "Fat Finger" Test)
*Demonstrates the `Auto-Correct Typos (LLM)` feature. Make sure this toggle is turned **ON** in the sidebar.*

**Copy this:**
```text
Wht ctiy srvd az Polnads cpatial n 1313?
```
**What happens under the hood:** LLM rewrites it to *"What city served as Poland's capital in 1313?"* before searching.
**Expected Answer:** `Kraków`

---

## 3. Semantic Search (The BM25 Killer)
*Demonstrates why Dense Vector Search is superior to Keyword Search. This query contains absolutely zero matching keywords with the source text.*

**Copy this:**
```text
During which century did the Scandinavian descendants seize control of the British Isles?
```
**What happens under the hood:** The Dense retriever maps "Scandinavian descendants" -> "Normans" and "British Isles" -> "England".
**Expected Answer:** `11th century`

---

## 4. Multi-hop Reasoning (Complex Logic)
*Demonstrates the LLM's ability to read multiple chunks, find two separate entities, and combine them to form a logical Yes/No conclusion.*

**Copy this:**
```text
Are the director of "Inception" and the director of "Pulp Fiction" both born in the United States?
```
**Expected Answer:** No (Nolan is British/English, Tarantino is American).

---

## 5. Zero-Hallucination Test (The Trick Question)
*Demonstrates the strictness of the prompt and the Cross-Encoder Reranker. This question is designed to trick the AI into hallucinating.*

**Copy this:**
```text
How many metric tons of carbon are believed to be released from the Amazon rain forest each year?
```
**What happens under the hood:** The context mentions the Amazon rainforest, but does NOT contain the exact carbon metric ton numbers. Instead of guessing, the AI refuses to answer.
**Expected Answer:** `I don't have information.`

---

## Bonus Typo Cases (More copy-paste options)

**Test A: Slang & Numbers**
```text
Wht wuz Jcksonvile refered 2 az afr th consoldation?
```
> *Expected: "Bold New City of the South"*

**Test B: Missing Vowels**
```text
Drin wht tim did civlzatin in th Amazn wuz florshng wen Orellana md his obzervatin?
```
> *Expected: 1540s*

---
**Tips for recording:**
1. Turn on **Auto-Correct Typos** and **Use Cross-Encoder Reranker** in the left sidebar.
2. After submitting a typo query, always open the **"View Details & Retrieved Documents"** expander to show the viewer the `LLM Corrected Query` magic!
