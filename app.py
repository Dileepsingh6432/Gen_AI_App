import os
import streamlit as st
import arxiv
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Customized prompt template
prompt_template = """
You are an expert in summarizing research papers. Write a detailed and structured summary of the following research paper:

**Title:** {title}

**Abstract:**
{text}

Your summary should include the following sections:
1. **Introduction**: Briefly describe the problem or research question addressed in the paper.
2. **Key Contributions**: Highlight the main contributions or innovations of the paper.
3. **Methodology**: Explain the methods, techniques, or approaches used in the research.
4. **Results**: Summarize the key findings or results of the study.
5. **Conclusion**: Provide a brief conclusion and discuss the implications or future work.

**Detailed Summary:**
"""

prompt = PromptTemplate.from_template(prompt_template)

# Initialize LLM
llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# Function to fetch papers from arXiv
def fetch_papers(query, max_results=1):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'summary': result.summary,
            'pdf_url': result.pdf_url,
            'published': result.published
        })
    return papers

# Function to summarize a paper
def summarize_paper(paper):
    # Create a LangChain Document object
    doc = Document(page_content=paper['summary'], metadata={"title": paper['title']})
    # Prepare input for the prompt
    input_data = {"title": paper['title'], "text": paper['summary']}
    # Generate summary
    summary = llm_chain.run(input_data)  # Pass the input data to the LLMChain
    return summary

# Streamlit app
def main():
    st.title("AI Research Paper Summarizer")
    
    # Input query
    query = st.text_input("Enter your search query for arXiv (e.g., 'AI', 'Machine Learning'):", "AI")
    
    if st.button("Fetch and Summarize Papers"):
        # Fetch papers
        papers = fetch_papers(query)
        
        # Display and summarize each paper
        for i, paper in enumerate(papers):
            st.subheader(f"Paper {i+1}: {paper['title']}")
            st.write(f"**Published on:** {paper['published']}")
            st.write(f"**PDF URL:** {paper['pdf_url']}")
            
            # Summarize the paper
            summary = summarize_paper(paper)
            st.write("**Detailed Summary:**")
            st.write(summary)
            st.write("---")

if __name__ == "__main__":
    main()
