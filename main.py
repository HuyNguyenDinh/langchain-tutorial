import os
from langchain.llms import VertexAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from google.oauth2 import service_account
from langchain.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import PlaywrightURLLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List

# Define the DesignPattern model
class DesignPattern(BaseModel):
    pattern: str = Field(description="name of design pattern")
    reason: str = Field(description="reason to choose this design pattern")

# Define the Response model
class Response(BaseModel):
    answer: List[DesignPattern]

output_parser = PydanticOutputParser(pydantic_object=Response)

# Define the prompt template for the retrievalQA
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
prompt_template = """
    You are a helpful assistant that can answer questions about cloud design pattern.
    
    Use the following pieces of context to answer the question.
    {context}
    
    Answer the following question: {question}
    
    List out name of top 3 suitable design patterns and brief explain reason
    {format_instructions}
"""
# Create a prompt instance with the defined template
PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions}
    )

credentials = service_account.Credentials.from_service_account_file("./credentials.json")
# Create RetrievalQA
urls = ["https://learn.microsoft.com/en-us/azure/architecture/patterns/", "https://microservices.io/patterns/"]
loader = PlaywrightURLLoader(urls=urls)
data = loader.load()

llm = VertexAI(project="canvas-primacy-404501", credentials=credentials, max_output_tokens=1000)
embeddings = VertexAIEmbeddings(project="canvas-primacy-404501", credentials=credentials)

reference_data = FAISS.from_documents(data, embeddings)

chain = LLMChain(llm=llm, prompt=PROMPT)

question = "Which design pattern should I use when design a system with long-time-processing backend but it must be correct at all steps, if failed it need to be rollbacks all the steps that success"
print(f"Question: {question}")

# Run the chain to get the response
docs = reference_data.similarity_search(question, k=4)
docs_page_content = " ".join([d.page_content for d in docs])
response = chain.run(question=question, docs=docs_page_content)
response_data = output_parser.parse(response)

# Print the design patterns and their reasons
print("Answer: ")
for design_pattern in response_data.answer:
    print(f"- {design_pattern.pattern}: {design_pattern.reason}")