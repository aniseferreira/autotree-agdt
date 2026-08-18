import re
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

st.title("🏛️ AutoTree AGDT — UD to Arethusa XML / CoNLL-U")
st.write(
    "Converta texto em Grego Antigo para o formato XML de dependências do **AGDT** "
    "(compatível com o Arethusa) ou exporte em **CoNLL-U** utilizando o Stanza."
)

# ============================================================
# 2. INICIALIZAÇÃO DO STANZA COM CACHE
# ============================================================
@st.cache_resource
def load_stanza_pipeline():
    stanza.download("grc", package="perseus", verbose=False)
    return stanza.Pipeline(
        lang="grc",
        package="perseus",
        processors="tokenize,lemma,pos,depparse",
        verbose=False
    )

with st.spinner("Carregando o modelo Stanza (Perseus)..."):
    nlp = load_stanza_pipeline()

# ============================================================
# 3. DIVISÃO E PRÉ-PROCESSAMENTO DE SENTENÇAS
# ============================================================
def pre_processar_sentencas(texto):
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    texto_limpo = " ".join(linhas)
    
    sentencas = re.split(r'([.;;·!?])', texto_limpo)
    
    frases_finais = []
    curr = ""
    for pedaco in sentencas:
        curr += pedaco
        if re.match(r'[.;;·!?]', pedaco):
            if curr.strip():
                frases_finais.append(curr.strip())
            curr = ""
    if curr.strip():
        frases_finais.append(curr.strip())
        
    return "\n\n".join(frases_finais) if frases_finais else texto

# ============================================================
# 4. REGRAS E CONSTANTES AGDT
# ============================================================

PARTICULAS_OUTE = {
    "οὔτε", "ουτε", "οὔτ", "ουτ", "οὔθ", "ουθ",
    "μήτε", "μητε", "μήτ", "μητ", "μήθ", "μηθ"
}

def e_particula_oute(word):
    if not word:
        return False
    text = normalizar(word.get("text", "")).strip("’'")
    lemma = normalizar(word.get("lemma", "")).strip("’'")
    return text in PARTICULAS_OUTE or lemma in PARTICULAS_OUTE

PRONOMES_RELATIVOS = {
    "ὅς", "ἥ", "ὅ", "ὅσπερ", "ἥπερ", "ὅπερ", "ὅστις", "ἥτις", "ὅτι",
    "οὗ", "ἧς", "ᾧ", "ᾗ", "ὅν", "ἥν", "ὧν", "οἷς", "αἷς", "ούς", "ἅς", "ἅ"
}

def e_pronome_relativo(word):
    if not word:
        return False
    lemma = normalizar(word.get("lemma", ""))
    text = normalizar(word.get("text", ""))
    xpos = word.get("xpos", "")
    is_rel_xpos = len(xpos) > 1 and xpos[0] == "p" and xpos[1] == "r"
    return is_rel_xpos or lemma in PRONOMES_RELATIVOS or text in PRONOMES_RELATIVOS

CONJUNCOES_INTEGRANTES = {"ὅτι", "οτι"}

CONJUNCOES_SUBORDINATIVAS = {
    "ὅτι", "οτι", "διότι", "διοτι", "ὡς", "ως", "ἐπειδή", "επειδη",
    "ἐπεί", "επει", "ἐπειδήπερ", "επειδηπερ", "ἐάν", "εαν", "ἄν", "αν",
    "εἰ", "ει", "ἐάνπερ", "εανπερ", "ὥσπερ", "ωσπερ", "καθάπερ", "καθαπερ"
}

VERBOS_DICENDI = {
    "λέγω", "φημί", "εἶπον", "ἀγγέλλω", "φράζω", "δηλόω", "ἀποκρίνομαι", "βοάω", "κηρύσσω",
    "νομίζω", "ἡγέομαι", "οἴομαι", "δοκέω", "πιστεύω",
    "ὁράω", "ἀκούω", "γιγνώσκω", "οἶδα", "πυνθάνομαι", "αἰσθάνομαι", "μανθάνω"
}

NEGACOES = {"οὐ", "οὐκ", "οὐχ", "οὐχι", "μὴ"}

PREPOSICOES_GREGAS = {
    "ἀμφί", "ἀμφὶ", "ἀνά", "ἀνὰ", "ἀντί", "ἀντὶ", "ἀπό", "ἀπὸ",
    "διά", "διὰ", "εἰς", "ἐκ", "ἐξ", "ἐν", "ἐπί", "ἐπὶ", "κατά", "κατὰ",
    "μετά", "μετὰ", "παρά", "παρὰ", "περί", "περὶ", "πρό", "πρὸ",
    "πρός", "πρὸς", "σύν", "σὺν", "ὑπέρ", "ὑπὲρ", "ὑπό", "ὑπὸ"
}
PONTUACAO_AUXG = {"ʼ", "῾", "'", "’", "“", "”", "‘", "’", '"'}
PONTUACAO_AUXK = {".", "·", "·", ";", ":", ";"}
PONTUACAO_AUXX = {","}

# --- FUNÇÕES AUXILIARES DE INSPEÇÃO ---

def normalizar(s):
    return s.lower().strip() if s else ""

def get_word(words, word_id):
    for w in words:
        if w["id"] == int(word_id): return w
    return None

def e_verbo_dicendi(word):
    if not word:
        return False
    lemma = normalizar(word.get("lemma", ""))
    return lemma in VERBOS_DICENDI

def extrair_caso(word):
    """Extrai a letra referente ao caso gramatical da palavra (n, g, d, a, v)."""
    if not word:
        return None
    xpos = word.get("xpos") or ""
    feats = word.get("feats") or ""
    
    if len(xpos) > 4 and xpos[4] in {"n", "g", "d", "a", "v"}:
        return xpos[4]
        
    for feat in feats.split("|"):
        if feat.startswith("Case="):
            case_val = feat.split("=")[1].lower()
            return case_val[0] if case_val else None
            
    return None

# --- REGRAS DE TRANSFORMAÇÃO ---

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
    if text in {"γὰρ", "γαρ", "μὲν", "μέν", "μεν", "δέ", "δε"} and rel == "advmod": return "AuxY"
    if text == "καὶ" and rel == "advmod": return "AuxZ"

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
        if conj["new_rel"] != "AuxC": 
            continue
        if normalizar(conj["text"]) in NEGACOES and conj["deprel"] not in {"mark", "sconj"}:
            continue
            
        subordinate = get_word(words, conj["head"])
        if subordinate is None: 
            continue

        conj["new_head"] = subordinate["new_head"]
        subordinate["new_head"] = conj["id"]

        text_conj = normalizar(conj["text"])
        matrix_verb = get_word(words, conj["new_head"])

        is_integrante = text_conj in CONJUNCOES_INTEGRANTES or (
            text_conj in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
        )

        subordinate["new_rel"] = "OBJ" if is_integrante else "ADV"

def aplicar_oute_correlativo(words):
    ocorrencias = [w for w in words if e_particula_oute(w)]
    if len(ocorrencias) >= 2:
        coord_principal = ocorrencias[-1]
        coord_principal["new_rel"] = "COORD"

        for auxy in ocorrencias[:-1]:
            auxy["new_rel"] = "AuxY"
            auxy["new_head"] = coord_principal["id"]

def resolver_predicados_excedentes(words):
    for w in words:
        head_word = get_word(words, w["new_head"])
        if head_word and head_word.get("new_rel") == "AuxC":
            if w["new_rel"] in {"PRED", "PRED_CO"}:
                conj_text = normalizar(head_word["text"])
                matrix_verb = get_word(words, head_word["new_head"])
                
                is_integrante = conj_text in CONJUNCOES_INTEGRANTES or (
                    conj_text in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
                )
                w["new_rel"] = "OBJ" if is_integrante else "ADV"

    preds = [w for w in words if w["new_rel"] == "PRED"]
    if len(preds) <= 1:
        return

    main_pred = next((p for p in preds if p["new_head"] == 0), preds[0])

    for p in preds:
        if p["id"] == main_pred["id"]:
            continue
        deprel = p.get("deprel", "")
        if deprel.startswith("conj") or any(w["deprel"].startswith("cc") for w in words):
            p["new_rel"] = "PRED_CO"
            main_pred["new_rel"] = "PRED_CO"
        else:
            p["new_rel"] = "ADV"

def aplicar_copula(words):
    for cop in words:
        if cop["new_rel"] != "cop": 
            continue
            
        predicative = get_word(words, cop["head"])
        if predicative is None: 
            continue

        case_pred = extrair_caso(predicative)
        nom_candidate = None

        # --- 1. BUSCA POR PREDICATIVO NOMINATIVO EXPRESSO ---
        if case_pred == "g":
            for w in words:
                if w["id"] != predicative["id"] and extrair_caso(w) == "n":
                    is_sbj = w["new_rel"] == "SBJ" or w["deprel"] == "nsubj" or any(
                        art["new_rel"] == "AuxE" and art["head"] == w["id"] for art in words
                    )
                    if not is_sbj:
                        nom_candidate = w
                        break
            
            if nom_candidate:
                predicative["new_rel"] = "ATR"
                predicative["new_head"] = nom_candidate["id"]
                predicative = nom_candidate

        cop_old_head = predicative["new_head"]
        parent_word = get_word(words, cop_old_head)

        # --- 2. ORAÇÃO RELATIVA (ATR) ---
        has_relative = any(
            e_pronome_relativo(w) and (w["head"] in {cop["id"], predicative["id"]} or w.get("new_head") in {cop["id"], predicative["id"]})
            for w in words
        )

        if has_relative:
            cop_relation = "ATR"
            antecedente = None
            rel_word = next((w for w in words if e_pronome_relativo(w)), None)
            if rel_word:
                for w in sorted(words, key=lambda x: int(x["id"]), reverse=True):
                    if int(w["id"]) < int(rel_word["id"]) and w["upos"] in {"NOUN", "PROPN", "ADJ"}:
                        antecedente = w
                        break
            if antecedente:
                cop_old_head = antecedente["id"]

        # --- 3. ORAÇÃO SUBORDINADA (ADV / OBJ) ---
        elif parent_word and parent_word.get("new_rel") == "AuxC":
            conj_text = normalizar(parent_word["text"])
            matrix_verb = get_word(words, parent_word["new_head"])
            is_integrante = conj_text in CONJUNCOES_INTEGRANTES or (
                conj_text in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
            )
            cop_relation = "OBJ" if is_integrante else "ADV"

        # --- 4. ORAÇÃO PRINCIPAL (PRED / PRED_CO) ---
        else:
            if predicative.get("new_rel") == "PRED_CO" or cop.get("deprel", "").startswith("conj"):
                cop_relation = "PRED_CO"
            else:
                cop_relation = "PRED"

        # --- REESTRUTURAÇÃO DOS NÓS ---
        cop["new_rel"] = cop_relation
        cop["new_head"] = cop_old_head
        
        predicative["new_rel"] = "PNOM"
        predicative["new_head"] = cop["id"]

        for child in words:
            if child["id"] == predicative["id"]:
                continue
            
            if nom_candidate and child["new_head"] == cop["id"] and extrair_caso(child) == "g" and child["new_rel"] not in {"SBJ", "OBJ"}:
                child["new_rel"] = "ATR"
                child["new_head"] = predicative["id"]

            if child["new_head"] == predicative["id"] or child["head"] == predicative["id"]:
                if child["new_rel"] in {"SBJ", "nsubj"} or e_pronome_relativo(child):
                    child["new_head"] = cop["id"]
                    if child["new_rel"] == "nsubj":
                        child["new_rel"] = "SBJ"

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
        is_part = "VerbForm=Part" in feats or (len(w["xpos"]) > 2 and w["xpos"][2] == "p")
        if not is_part:
            continue
        
        artigos = [
            art for art in words 
            if art["upos"] == "DET" and art["head"] == w["id"] and normalizar(art["lemma"]) in {"ὁ", "ο"}
        ]
        
        if artigos:
            is_nom = "Case=Nom" in feats or (len(w["xpos"]) > 7 and w["xpos"][7] == "n")
            if is_nom or w["deprel"] in {"nsubj", "nsubj:pass", "csubj"}:
                w["new_rel"] = "SBJ"
            elif w["deprel"] in {"obj", "iobj", "obl:arg"}:
                w["new_rel"] = "OBJ"
                
            for artigo in artigos:
                artigo["new_rel"] = "ATR"
                artigo["new_head"] = w["id"]

def aplicar_regras_infinitivo(words):
    for infinitivo in list(words):
        feats = infinitivo.get("feats") or ""
        xpos = infinitivo.get("xpos") or ""
        
        # Correção: O modo no XPOS do AGDT/Perseus (ex: v--pne---) fica no índice 4
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 4 and xpos[4] == "n")
        if not is_inf:
            continue

        head_word = get_word(words, infinitivo["head"])
        deprel = infinitivo.get("deprel", "")

        # 1. ELIPSE VERBAL COM NÓ ARTIFICIAL [0]
        if (infinitivo["head"] == 0 or deprel == "root") and not any(
            w["upos"] == "VERB" and w["id"] != infinitivo["id"] for w in words
        ):
            pred = None
            for w in words:
                if w["id"] != infinitivo["id"] and w["head"] == infinitivo["id"]:
                    if w["upos"] in {"ADJ", "NOUN"} or (len(w["xpos"]) > 0 and w["xpos"][0] in {"a", "n"}):
                        pred = w
                        break
            
            if pred:
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
                pred["new_rel"] = "PNOM"
                pred["new_head"] = artificial_id
                words.append(artificial)

                # Ajusta os dependentes do predicativo (ex: dativo de referência "οὐδενὶ")
                for child in words:
                    if child["head"] == pred["id"] and extrair_caso(child) == "d":
                        child["new_rel"] = "OBJ"
                continue

        # 2. INFINITIVO SUJEITO DE VERBO IMPESSOAL
        if deprel in {"csubj", "csubj:pass"}:
            infinitivo["new_rel"] = "SBJ"
            continue

        # 3. INFINITIVO OBJETO / COMPLETIVO
        infinitivo["new_rel"] = "OBJ"

def converter_sentenca(sent):
    words = construir_words(sent)
    inicializar_agdt(words)
    aplicar_auxiliares_especiais(words)
    aplicar_regras_infinitivo(words)
    aplicar_participios_substantivados(words)
    
    aplicar_oute_correlativo(words)
    
    aplicar_auxv(words)
    aplicar_auxc(words)
    aplicar_copula(words)
    aplicar_coordenacao(words)
    aplicar_auxp(words)
    aplicar_artigos_repetidos(words)

    resolver_predicados_excedentes(words)

    for w in words:
        if w["new_head"] is None: 
            w["new_head"] = w["head"]
        if w["new_rel"] is None: 
            w["new_rel"] = w["deprel"]
        if w["new_rel"] == "AuxK": 
            w["new_head"] = 0

        xpos = w.get("xpos") or ""
        feats = w.get("feats") or ""
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 2 and xpos[2] == "n")
        
        if is_inf and w["new_rel"] in {"PRED", "PRED_CO"}:
            w["new_rel"] = "OBJ" if w["new_rel"] == "PRED" else "OBJ_CO"

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

def gerar_conllu(doc):
    lines = []
    for i, sent in enumerate(doc.sentences, start=1):
        lines.append(f"# sent_id = {i}")
        lines.append(f"# text = {sent.text}")
        for word in sent.words:
            fields = [
                str(word.id),
                word.text or "_",
                word.lemma or "_",
                word.upos or "_",
                word.xpos or "_",
                word.feats or "_",
                str(word.head),
                word.deprel or "_",
                "_",
                "_"
            ]
            lines.append("\t".join(fields))
        lines.append("")
    return "\n".join(lines)

# ============================================================
# 5. INTERFACE DO USUÁRIO (STREAMLIT)
# ============================================================

st.subheader("1. Entrada de Texto")
modo_entrada = st.radio(
    "Escolha o método de inserção do texto:",
    ("Digitar / Colar Texto", "Carregar arquivo TXT"),
    horizontal=True
)

entrada_texto = ""
texto_exemplo = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος.\nοὐλομένην, ἣ μυρί' Ἀχαιοῖς άλγε' ἔθηκε."

if modo_entrada == "Digitar / Colar Texto":
    entrada_texto = st.text_area(
        "Digite ou cole o texto em Grego Antigo:",
        value=texto_exemplo,
        height=140
    )
else:
    arquivo_carregado = st.file_uploader(
        "Selecione um arquivo .txt:",
        type=["txt"]
    )
    if arquivo_carregado is not None:
        entrada_texto = arquivo_carregado.read().decode("utf-8")
        st.text_area("Pré-visualização do arquivo carregado:", value=entrada_texto, height=140, disabled=True)

if st.button("Processar Texto", type="primary"):
    if not entrada_texto.strip():
        st.warning("Por favor, insira ou carregue um texto para conversão.")
    else:
        with st.spinner("Processando anotação sintática com Stanza..."):
            texto_formatado = pre_processar_sentencas(entrada_texto)
            
            doc = nlp(texto_formatado)
            sentences_convertidas = [converter_sentenca(sent) for sent in doc.sentences]
            
            xml_str = gerar_agdt_xml(sentences_convertidas)
            conllu_str = gerar_conllu(doc)

        st.success(f"Processamento concluído com sucesso! ({len(doc.sentences)} sentença(s) identificada(s))")
        st.subheader("2. Resultados e Exportação")

        tab_xml, tab_conllu = st.tabs(["📄 XML Arethusa (AGDT)", "📝 CoNLL-U (UD)"])

        with tab_xml:
            st.download_button(
                label="💾 Baixar XML Arethusa",
                data=xml_str,
                file_name="autotree_agdt.xml",
                mime="application/xml",
                key="btn_xml"
            )
            st.markdown("**Preview do XML (Arethusa):**")
            st.code(xml_str, language="xml")

        with tab_conllu:
            st.download_button(
                label="💾 Baixar CoNLL-U",
                data=conllu_str,
                file_name="autotree_ud.conllu",
                mime="text/plain",
                key="btn_conllu"
            )
            st.markdown("**Preview do CoNLL-U:**")
            st.code(conllu_str, language="plaintext")
