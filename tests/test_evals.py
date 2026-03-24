import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision
from src.chain import get_rag_chain

def test_rag_pipeline():
    chain = get_rag_chain(user_role="c-level")
    
    questions = ["What were the Q3 marketing expenses?", "Who is on the payroll?"]
    ground_truths = [["Q3 marketing expenses were $50,000."], ["John Doe, Jane Smith."]]
    
    answers = []
    contexts = []
    
    for q in questions:
        res = chain.invoke({"input": q})
        answers.append(res["answer"])
        contexts.append([doc.page_content for doc in res["context"]])

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[answer_relevancy, context_precision])
    
    # Fail the CI/CD pipeline if metrics drop below 0.8
    assert result["answer_relevancy"] > 0.8
    assert result["context_precision"] > 0.8