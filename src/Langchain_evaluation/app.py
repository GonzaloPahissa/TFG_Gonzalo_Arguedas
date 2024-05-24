import os
from io import StringIO

import bert_score
import streamlit as st
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se ha encontrado la API Key en el archivo .env")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize GPT-2 model and tokenizer for perplexity calculation
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    else:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        return stringio.read()


def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()


def main():
    st.set_page_config(page_title="Chat Interface", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        .stTextInput, .stTextArea, .stButton {
            margin-top: 20px;
        }
        .stTextInput input {
            font-size: 18px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
        }
        .stAlert {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-top: 20px;
        }
        .stAlert h4 {
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Chat Interface")

    source = st.file_uploader("Cargar archivo", type=["pdf", "txt"])

    if source is not None:
        document_text = extract_text_from_file(source)

        st.success("Texto extraido correctamente del archivo.")

        try:
            # Load the text document
            document = Document(page_content=document_text)
            documents = [document]

            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(documents, embeddings)
            retriever = vector_store.as_retriever()
            llm = ChatOpenAI(api_key=api_key, model_name="gpt-4-turbo")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever
            )

            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": "¿En qué podría ayudarte?"}
                ]

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            def generate_gpt_response(query):
                respuesta = qa_chain.run({"query": query})
                return respuesta

            def clear_chat_history():
                st.session_state.messages = [
                    {"role": "system", "content": "¿En qué podría ayudarte?"}
                ]

            st.sidebar.button("Borrar historial del chat", on_click=clear_chat_history)

            if prompt := st.chat_input("Your question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("system"):
                    with st.spinner("Thinking..."):
                        response = generate_gpt_response(prompt)
                        st.write(response)
                        st.session_state.messages.append(
                            {"role": "system", "content": response}
                        )

                        # Evaluate the response
                        bert_p, bert_r, bert_f1 = bert_score.score(
                            [response], [document_text], lang="en"
                        )
                        respuesta_embedding = embeddings.embed_query(response)
                        documento_embedding = embeddings.embed_query(document_text)
                        cos_sim = cosine_similarity(
                            [respuesta_embedding], [documento_embedding]
                        )[0][0]
                        perplexity = calculate_perplexity(response)

                        st.markdown(f"**Similitud Coseno:** {cos_sim:.2f}")
                        st.markdown(
                            f"**BERTScore Precision:** {bert_p.mean().item():.2f}"
                        )
                        st.markdown(f"**BERTScore Recall:** {bert_r.mean().item():.2f}")
                        st.markdown(f"**BERTScore F1:** {bert_f1.mean().item():.2f}")
                        st.markdown(f"**Perplexity:** {perplexity:.2f}")
        except Exception as e:
            st.error(f"Error al cargar el documento: {str(e)}")


if __name__ == "__main__":
    main()
