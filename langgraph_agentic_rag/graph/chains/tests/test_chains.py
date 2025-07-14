"""
Important thing to remember: To run this go to the root directory and run the command:
pytest -s -v
"""
from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
load_dotenv()
import pprint

# Define test for yes answer
def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content # Get the most relevant document

    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_text})

    assert res.binary_score == "yes"

# Define test for no answer
def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content # Get the most relevant document

    res: GradeDocuments = retrieval_grader.invoke({"question": "how to make a pizza", "document": doc_text})

    assert res.binary_score == "no"


# Define test for generation chain
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res: str = generation_chain.invoke({"question": question, "context": docs})
    pprint.pprint(res)

