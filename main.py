import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Initializing components")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs = {"k":3})

prompt_template = ChatPromptTemplate.from_template("""Answer the question based on the following context:
{context}
Question: {question}
Provide a detailed answer:
""")

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    docs= retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context,question=query)
    response = llm.invoke(messages)
    return response

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain=(
        RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)|prompt_template|llm|StrOutputParser()
    )
    return retrieval_chain

if __name__=="__main__":
    print("Retrieving...")
    query = "What is Pinecone in machine learning"
    result_without_lcel = retrieval_chain_without_lcel(query)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print(result_without_lcel)
    print(result_with_lcel)
