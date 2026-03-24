from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> str:
    """Scrub sensitive data (SSN, Emails, Phone numbers)"""
    results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"], language='en')
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text.text

def is_out_of_scope(query: str, llm) -> bool:
    """Use a lightweight LLM call to detect out-of-scope questions"""
    prompt = f"""
    You are a security classifier. The user is asking an internal company chatbot a question.
    If the question is about sports, politics, coding help, or general knowledge outside of company HR/Finance, reply 'YES'.
    If it is a valid company question, reply 'NO'.
    Question: {query}
    """
    response = llm.invoke(prompt)
    return "YES" in response.content.upper()