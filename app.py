import streamlit as st
import trankit
import pandas as pd

# 1. Carregamento do modelo com cache para não recarregar a cada interação
@st.cache_resource
def load_trankit_pipeline(treebank_model: str = "ancient_greek-perseus"):
    """
    Carrega o pipeline do Trankit.
    Opções recomendadas para grego antigo:
    - 'ancient_greek-perseus' (Alinhado ao AGDT/Perseids)
    - 'ancient_greek-proiel'
    """
    # Desative gpu=True se não houver suporte a CUDA no ambiente de execução
    return trankit.Pipeline(lang=treebank_model, gpu=False)

def trankit_to_agdt_dataframe(doc: dict) -> pd.DataFrame:
    """
    Converte o dicionário de saída do Trankit em um DataFrame no padrão CoNLL-U/AGDT.
    """
    rows = []
    for sent_idx, sent in enumerate(doc.get("sentences", []), start=1):
        for token in sent.get("tokens", []):
            # No Trankit, 'tokens' contém as anotações de morfossintaxe e dependência
            rows.append({
                "sent_id": sent_idx,
                "id": token.get("id"),
                "form": token.get("text"),
                "lemma": token.get("lemma", "_"),
                "upostag": token.get("upos", "_"),
                "xpostag": token.get("xpos", "_"),
                "feats": token.get("feats", "_"),
                "head": token.get("head", 0),
                "deprel": token.get("deprel", "_")
            })
    return pd.DataFrame(rows)

# 2. Interface Streamlit
st.title("Anotador AGDT (Trankit Backend)")

# Seleção do modelo do Treebank
model_choice = st.sidebar.selectbox(
    "Modelo de Treebank Grego:",
    ["ancient_greek-perseus", "ancient_greek-proiel"]
)

nlp = load_trankit_pipeline(model_choice)

input_text = st.text_area(
    "Texto em Grego Antigo:",
    "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος",
    height=150
)

if st.button("Analisar Dependências"):
    if input_text.strip():
        with st.spinner("Processando anotação morfossintática e dependências..."):
            # O Trankit executa tokenização, POS, lematização e depparse no parâmetro padrão
            doc = nlp(input_text)
            
            # Converte saída em tabela/dataframe
            df_agdt = trankit_to_agdt_dataframe(doc)
            
            st.subheader("Resultado do Parse (AGDT/CoNLL-U)")
            st.dataframe(df_agdt, use_container_width=True)
            
            # Download em formato CSV/TSV
            csv_data = df_agdt.to_csv(sep="\t", index=False)
            st.download_button(
                label="Baixar Anotações (TSV/CoNLL-U)",
                data=csv_data,
                file_name="agdt_trankit_parsed.tsv",
                mime="text/tab-separated-values"
            )
    else:
        st.warning("Insira um texto válido.")
