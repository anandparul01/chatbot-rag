import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

def load_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "cpu",low_cpu_mem_usage = True,torch_dtype=torch.float32)

    hf_pipeline =  pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens =256, temperature = 0)
    return HuggingFacePipeline(pipeline=hf_pipeline)

# PDF uploading and Processing
uploaded_file= st.file_uploader("Upload your pdf", type = "pdf")
if uploaded_file:
    reader = PdfReader(uploaded_file)
    all_text = [page.extract_text() for page in reader.pages if page.extract_text()]
    full_text = "\n".join(all_text)

# Splitting the text and creating chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    st.success(f"Processed PDF : Total chunks are: {len(chunks)}")

# Creating vector store
    embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks,embed_model)
    st.success("PDF processed successfully")

# Choose model
    st.divider()
    llm_model_name = st.text_input("LLM model path", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    if "qa" not in st.session_state:
        st.session_state.qa = None

    if st.button("Connect AI"):
        with st.spinner("Connecting AI ..."):
            llm = load_llm(llm_model_name)
            st.session_state.qa = RetrievalQA.from_chain_type(llm = llm, retriever = db.as_retriever(search_kwargs ={"k":3}),
                                                              return_source_documents = True)
            st.success("AI is connected")

#Question and answer
    if st.session_state.qa:
        question = st.chat_input("Ask a question")

        if question:
            response = st.session_state.qa.invoke({"query": question})

            with st.chat_message("assistant"):
                st.write(response["result"])

