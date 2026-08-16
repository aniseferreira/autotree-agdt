import xml.etree.ElementTree as ET
from xml.dom import minidom
import streamlit as st
import stanza

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ============================================================
st.set_page_config(
    page_title="AutoTree AGDT",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ AutoTree AGDT — UD to Arethusa XML")
st.write(
    "Converta texto em Grego Antigo para o formato XML de dependências do **AGDT** "
    "(compatível com o Arethusa) utilizando o Stanza."
)

# ============================================================
# 2. INICIALIZAÇÃO DO STANZA COM CACHE
# ============================================================
@st.cache_resource
def load_stanza_pipeline():
    # Faz o download do modelo perseus do Stanza apenas uma vez
    stanza.download("grc", package="perseus", verbose=False)
    return stanza.Pipeline(
        lang="grc",
        package="perseus",
        processors="tokenize,lemma,pos,depparse",
        tokenize_no_ssplit=True,
        verbose=False
    )

with st.spinner("Carregando o modelo Stanza (Perseus)..."):
    nlp = load_stanza_pipeline()

# ============================================================
# 3. REGRAS E CONSTANTES AGDT
# ============================================================
NEGACOES = {"οὐ", "οὐκ", "οὐχ", "οὐχι", "μή"}
CONJUNCOES_SUBORDINATIVAS = {
    "ὅτι", "οτι", "διότι", "διοτι", "ὡς", "ως", "ἐπειδή", "επειδη",
    "ἐπεί", "επει", "ἐπειδήπερ", "επειδηπερ", "ἐάν", "εαν", "ἄν", "αν",
    "εἰ", "ει", "ἐάνπερ", "εανπερ"
}
PREPOSICOES_GREGAS = {
    "ἀμφί", "ἀμφὶ", "ἀνά", "ἀνὰ", "ἀντί", "ἀντὶ", "ἀπό", "ἀπὸ",
    "διά", "διὰ", "εἰς", "ἐκ", "ἐξ", "ἐν", "ἐπί", "ἐπὶ", "κατά", "κατὰ",
    "μετά", "μετὰ", "παρά", "παρὰ", "περί", "περὶ", "πρό", "πρὸ",
    "πρός", "πρὸς", "σύν", "σὺν", "ὑπέρ", "ὑπὲρ", "ὑπό", "ὑπὸ"
}
PONTUACAO_AUXG = {"ʼ", "῾", "'", "’", "“", "”", "‘", "’", '"'}
PONTUACAO_AUXK = {".", "·", "·", ";", ":", ";"}
PONTUACAO_AUXX = {","}

def normalizar(s):
    return s.lower().strip() if s else ""

def construir_words(sent):
    words = []
    for token in sent.words:
        words.append({
            "id": int(token.id),
            "text": token.text or "_",
            "lemma": token.lemma or "_",
            "upos": token.upos or "_",
            "xpos": token.xpos or "_",
            "feats": token.feats or "_",
            "head": int(token.head),
            "deprel": token.deprel or "_",
            "new_rel": None,
            "new_head": None
        })
    return words

def mapear_relacao_basica(w):
    text = normalizar(w["text"])
    rel = w["deprel"]

    if text in PONTUACAO_AUXX: return "AuxX"
    if text in PONTUACAO_AUXK: return "AuxK"
    if text in PONTUACAO_AUXG: return "AuxG"

    if text in {"ἂν", "ἄν", "αν"} and rel == "advmod": return "AuxY"
    if text in {"γὰρ", "γαρ", "μὲν", "μεν", "δέ", "δε"} and rel == "advmod": return "AuxY"
    if text == "καί" and rel == "advmod": return "AuxZ"

    if text in NEGACOES:
        return "AuxC" if rel in {"mark", "sconj"} else "AuxZ"

    if text in CONJUNCOES_SUBORDINATIVAS: return "AuxC"

    if text in PREPOSICOES_GREGAS:
        w["upos"] = "ADP"
        w["xpos"] = "r--------"
        return "AuxP"

    if rel == "vocative": return "ExD"
    if rel == "cop": return "cop"
    if rel == "aux": return "AuxV"

    if rel == "punct":
        if w["text"] in PONTUACAO_AUXX: return "AuxX"
        if w["text"] in PONTUACAO_AUXK: return "AuxK"
        return "AuxG"

    if w["upos"] == "ADJ" and rel in {"obj", "iobj", "obl", "obl:arg", "nsubj", "nsubj:pass", "csubj"}:
        if rel in {"nsubj", "nsubj:pass", "csubj"}: return "SBJ"
        if rel in {"obj", "iobj", "obl:arg"}: return "OBJ"
        if rel == "obl": return "ADV"

    mapa = {
        "nsubj": "SBJ", "nsubj:pass": "SBJ", "obj": "OBJ", "iobj": "OBJ",
        "xcomp": "OBJ", "ccomp": "OBJ", "csubj": "SBJ", "csubj:pass": "SBJ",
        "root": "PRED", "advmod": "ADV", "advcl": "ADV", "obl": "ADV",
        "obl:arg": "OBJ", "amod": "ATR", "det": "ATR", "nmod": "ATR",
        "acl": "ATR", "acl:relcl": "ATR", "case": "AuxP", "mark": "AuxC",
        "sconj": "AuxC", "cc": "COORD",
    }
    return mapa.get(rel, rel)

def inicializar_agdt(words):
    for w in words:
        w["new_rel"] = mapear_relacao_basica(w)
        w["new_head"] = w["head"]

def get_word(words, word_id):
    for w in words:
        if w["id"] == int(word_id): return w
    return None

def aplicar_auxp(words):
    for prep in words:
        if prep["new_rel"] != "AuxP": continue
        governed = get_word(words, prep["head"])
        if governed is None: continue
        original_relation = governed["new_rel"]
        prep["new_head"] = governed["new_head"]
        governed["new_head"] = prep["id"]
        if original_relation in {"OBJ", "ADV", "ATR", "SBJ", "PRED"}:
            governed["new_rel"] = original_relation
        if prep["new_head"] == prep["id"]: prep["new_head"] = 0

def aplicar_auxc(words):
    for conj in words:
        if conj["new_rel"] != "AuxC": continue
        if normalizar(conj["text"]) in NEGACOES and conj["deprel"] not in {"mark", "sconj"}:
            continue
        subordinate = get_word(words, conj["head"])
        if subordinate is None: continue
        subordinate_relation = subordinate["new_rel"] or "OBJ"
        conj["new_head"] = subordinate["new_head"]
        subordinate["new_head"] = conj["id"]
        if subordinate_relation in {"ADV", "OBJ", "ATR", "SBJ"}:
            subordinate["new_rel"] = subordinate_relation

def aplicar_copula(words):
    for cop in words:
        if cop["new_rel"] != "cop": continue
        predicative = get_word(words, cop["head"])
        if predicative is None: continue
        cop_old_head = predicative["new_head"]
        cop["new_rel"] = "PRED"
        cop["new_head"] = cop_old_head
        predicative["new_rel"] = "PNOM"
        predicative["new_head"] = cop["id"]
        for child in words:
            if child["new_head"] == predicative["id"] and child["new_rel"] == "SBJ":
                child["new_head"] = cop["id"]

def aplicar_auxv(words):
    for w in words:
        if w["lemma"] == "εἰμί" and w["deprel"] == "aux":
            w["new_rel"] = "AuxV"

def aplicar_coordenacao(words):
    elementos_conj = [w for w in words if w["deprel"].split(":")[0] == "conj"]
    if not elementos_conj: return

    grupos = {}
    for segundo in elementos_conj:
        grupos.setdefault(str(segundo["head"]), []).append(segundo)

    for primeiro_id, segundos in grupos.items():
        primeiro = get_word(words, primeiro_id)
        if primeiro is None: continue
        elementos = sorted([primeiro] + segundos, key=lambda w: int(w["id"]))
        funcao_base = primeiro.get("new_rel") or mapear_relacao_basica(primeiro)
        if funcao_base in {"conj", "cc", "COORD"}: funcao_base = "PRED"
        if funcao_base.endswith("_CO"): funcao_base = funcao_base[:-3]

        ids_elementos = {str(e["id"]) for e in elementos}
        conjuncoes = [w for w in words if w["deprel"].split(":")[0] == "cc" and str(w["head"]) in ids_elementos]
        if not conjuncoes:
            for elemento in elementos:
                elemento["new_rel"] = f"{funcao_base}_CO"
                if elemento is not primeiro: elemento["new_head"] = primeiro["new_head"]
            continue

        coord_real = conjuncoes[-1]
        antigo_head = primeiro["new_head"]
        coord_real["new_rel"] = "COORD"
        coord_real["new_head"] = antigo_head

        for elemento in elementos:
            elemento["new_rel"] = f"{funcao_base}_CO"
            elemento["new_head"] = coord_real["id"]

def aplicar_artigos_repetidos(words):
    for idx, w in enumerate(words):
        if w["upos"] != "DET" or idx < 2: continue
        for idx_ant in range(idx - 1, -1, -1):
            anterior = words[idx_ant]
            if anterior["upos"] != "DET": continue
            if len(w["xpos"]) >= 9 and len(anterior["xpos"]) >= 9 and w["xpos"][6:9] == anterior["xpos"][6:9] and w["xpos"][6:9] != "---":
                w["new_head"] = anterior["head"]
                w["new_rel"] = "ATR"
                break

def aplicar_auxiliares_especiais(words):
    for w in words:
        text = normalizar(w["text"])
        if text in {"ἄν", "ἂν", "αν"} and w["deprel"] == "advmod": w["new_rel"] = "AuxY"
        elif text == "καί" and w["deprel"] == "advmod": w["new_rel"] = "AuxZ"
        elif text in {"γὰρ", "γαρ", "μὲν", "μεν", "δέ", "δε"} and w["deprel"] == "advmod": w["new_rel"] = "AuxY"

def aplicar_participios_substantivados(words):
    for w in words:
        feats = w.get("feats") or ""
        # Verifica se é particípio (seja por feats UD ou pela postag/xpos)
        is_part = "VerbForm=Part" in feats or (len(w["xpos"]) > 2 and w["xpos"][2] == "p")
        if not is_part:
            continue
        
        # Verifica se o particípio tem artigo dependente
        artigos = [
            art for art in words 
            if art["upos"] == "DET" and art["head"] == w["id"] and normalizar(art["lemma"]) in {"ὁ", "ο"}
        ]
        
        if artigos:
            # Se estiver no nominativo (feats UD ou xpos), assume a função de Sujeito
            is_nom = "Case=Nom" in feats or (len(w["xpos"]) > 7 and w["xpos"][7] == "n")
            if is_nom or w["deprel"] in {"nsubj", "nsubj:pass", "csubj"}:
                w["new_rel"] = "SBJ"
            elif w["deprel"] in {"obj", "iobj", "obl:arg"}:
                w["new_rel"] = "OBJ"
                
            for artigo in artigos:
                artigo["new_rel"] = "ATR"
                artigo["new_head"] = w["id"]


def aplicar_infinitivo_sujeito(words):
    for infinitivo in list(words):
        feats = infinitivo.get("feats") or ""
        # Corrigido 'w["xpos"]' para 'infinitivo["xpos"]'
        is_inf = "VerbForm=Inf" in feats or (len(infinitivo["xpos"]) > 2 and infinitivo["xpos"][2] == "n")
        if not is_inf:
            continue

        # Caso 1: Infinitivo na posição inicial da sentença funcionando como PRED/SBJ sem verbo principal
        if infinitivo["head"] == 0 or infinitivo["deprel"] == "root":
            # Procura um adjetivo/substantivo que funcione como predicativo (ex: ἀγαθόν)
            pred = None
            for w in words:
                if w["id"] != infinitivo["id"] and w["head"] == infinitivo["id"]:
                    if w["upos"] in {"ADJ", "NOUN"} or (len(w["xpos"]) > 0 and w["xpos"][0] in {"a", "n"}):
                        pred = w
                        break
            
            # Cria o nó artificial elíptico [0] para a cópula omissa
            artificial_id = max(int(w["id"]) for w in words) + 1
            artificial = {
                "id": artificial_id,
                "text": "[0]",
                "lemma": "_",
                "upos": "_",
                "xpos": "_",
                "feats": "_",
                "head": 0,
                "deprel": "root",
                "new_rel": "PRED",
                "new_head": 0,
                "artificial": True,
                "insertion_id": "0001e",
            }
            
            infinitivo["new_rel"] = "SBJ"
            infinitivo["new_head"] = artificial_id
            
            if pred:
                pred["new_rel"] = "PNOM"
                pred["new_head"] = artificial_id
                
            words.append(artificial)
            break
            
        # Caso 2: Infinitivo explicitamente marcado como sujeito de oração copular
        elif infinitivo["deprel"] in {"csubj", "csubj:pass"}:
            infinitivo["new_rel"] = "SBJ"
            
def converter_sentenca(sent):
    words = construir_words(sent)
    inicializar_agdt(words)
    aplicar_auxiliares_especiais(words)
    aplicar_infinitivo_sujeito(words)
    aplicar_participios_substantivados(words)
    aplicar_auxv(words)
    aplicar_copula(words)
    aplicar_auxc(words)
    aplicar_coordenacao(words)
    aplicar_auxp(words)
    aplicar_artigos_repetidos(words)

    for w in words:
        if w["new_head"] is None: w["new_head"] = w["head"]
        if w["new_rel"] is None: w["new_rel"] = w["deprel"]
        if w["new_rel"] == "AuxK": w["new_head"] = 0

    return {"text": sent.text, "words": words}

def gerar_agdt_xml(sentences, nome_base="arethusa_agdt"):
    root = ET.Element("treebank", {"xml:lang": "grc", "format": "aldt", "version": "1.5"})
    for i, sent in enumerate(sentences, start=1):
        sentence = ET.SubElement(root, "sentence", {"id": str(i), "document_id": nome_base, "subdoc": "", "span": ""})
        for w in sent["words"]:
            if w.get("artificial"):
                attrs = {
                    "id": str(w["id"]),
                    "insertion_id": w.get("insertion_id", "0003e"),
                    "artificial": "elliptic",
                    "relation": w["new_rel"],
                    "form": w["text"],
                    "head": str(w["new_head"]),
                }
            else:
                attrs = {
                    "id": str(w["id"]),
                    "form": w["text"],
                    "lemma": w["lemma"],
                    "postag": w["xpos"],
                    "head": str(w["new_head"]),
                    "relation": w["new_rel"]
                }
            ET.SubElement(sentence, "word", attrs)

    xml_bytes = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

# ============================================================
# 4. INTERFACE DO USUÁRIO (STREAMLIT)
# ============================================================
texto_exemplo = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
entrada_texto = st.text_area(
    "Digite ou cole a frase/texto em Grego Antigo:",
    value=texto_exemplo,
    height=120
)

if st.button("Converter para AGDT XML", type="primary"):
    if not entrada_texto.strip():
        st.warning("Por favor, insira um texto para conversão.")
    else:
        with st.spinner("Processando anotação sintática..."):
            doc = nlp(entrada_texto)
            sentences_convertidas = [converter_sentenca(sent) for sent in doc.sentences]
            xml_str = gerar_agdt_xml(sentences_convertidas)

        st.success("Conversão concluída!")
        
        # Botão para Download do XML
        st.download_button(
            label="💾 Baixar XML Arethusa",
            data=xml_str,
            file_name="autotree_agdt.xml",
            mime="application/xml"
        )

        # Exibição do resultado
        st.subheader("Resultado XML (Preview):")
        st.code(xml_str, language="xml")
