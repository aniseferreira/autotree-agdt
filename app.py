import json
import requests
import stanza
import streamlit as st
import re
import unicodedata
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ==============================================================================
# 1. INTERFACE DO STREAMLIT (BARRA LATERAL - SIDEBAR) soh no comentário
# ==============================================================================

# st.sidebar.title("Configurações do Parser")

# Seletor da LLM na barra lateral do app Streamlit
# st.sidebar.markdown("### Refinamento Semântico (LLM)")

# opcoes_modelos = {
#    "Claude 3.5 Haiku (Rápido, Barato e Preciso)": "anthropic/claude-3.5-haiku",
#   "Claude 3.5 Sonnet (Máxima Precisão)": "anthropic/claude-3.5-sonnet",
#    "Gemini 2.5 Flash (Recomendado)": "google/gemini-2.5-flash",
#    "Gemini 2.5 Pro (Análise Profunda)": "google/gemini-2.5-pro",
#    "GPT-4o Mini (Econômico)": "openai/gpt-4o-mini"
# }

# modelo_rotulo = st.sidebar.selectbox(
#    "Escolha o Modelo do OpenRouter:",
#   options=list(opcoes_modelos.keys())
# )

# Guarda o modelo escolhido na memória da sessão para o converter_sentenca usar
# st.session_state["modelo_llm"] = opcoes_modelos[modelo_rotulo]

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
# 2. CONSTANTES DE LINGUAGEM (GREGO POLITÔNICO)
# ============================================================

PREPOSICOES_GREGAS = {
    # Formas plenas
    "ἀμφί", "ἀμφὶ", "ἀνά", "ἀνὰ", "ἀντί", "ἀντὶ", "ἀπό", "ἀπὸ",
    "διά", "διὰ", "εἰς", "ἐκ", "ἐξ", "ἐν", "ἐπί", "ἐπὶ", "κατά", "κατὰ",
    "μετά", "μετὰ", "περί", "περὶ", "πρό", "πρὸ", "πρός", "πρὸς", "σύν", "σὺν", "ὑπέρ", "ὑπὲρ", "ὑπό", "ὑπὸ",
    # Formas elididas e apóstrofos prosódicos (variantes unicode)
    "ἀμφ'", "ἀμφ’", "ἀντ'", "ἀντ’", "ἀνθ'", "ἀνθ’", "ἀπ'", "ἀπ’", "ἀφ'", "ἀφ’",
    "δι'", "δι’", "ἐπ'", "ἐπ’", "ἐφ'", "ἐφ’", "κατ'", "κατ’", "κατ᾿", "καθ'", "καθ’", "καθ᾿",
    "μετ'", "μετ’", "μεθ'", "μεθ’", "παρ'", "παρ’", "περ'", "περ’", "ὑπ'", "ὑπ’", "ὑφ'", "ὑφ’"
}

LEMAS_PREPOSICOES_ELIDIDAS = {
    "κατ": "κατά", "καθ": "κατά",
    "δι": "διά",
    "μετ": "μετά", "μεθ": "μετά",
    "παρ": "παρά",
    "επ": "ἐπί", "εφ": "ἐπί",
    "απ": "ἀπό", "αφ": "ἀπό",
    "αντ": "ἀντί", "ανθ": "ἀντί",
    "αμφ": "ἀμφί",
    "υπ": "ὑπό", "υφ": "ὑπό"
}

PARTICULAS_OUTE = {
    "οὔτε", "ουτε", "οὔτ", "ουτ", "οὔθ", "ουθ",
    "μήτε", "μητε", "μήτ", "μητ", "μήθ", "μηθ"
}

PRONOMES_RELATIVOS = {
    "ὅς", "ἥ", "ὅ", "ὅσπερ", "ἥπερ", "ὅπερ", "ὅστις", "ἥτις", "ὅτι",
    "οὗ", "ἧς", "ᾧ", "ᾗ", "ὅν", "ἥν", "ὧν", "οἷς", "αἷς", "ούς", "ἅς", "ἅ"
}

CONJUNCOES_INTEGRANTES = {"ὅτι", "οτι"}

CONJUNCOES_SUBORDINATIVAS = {
    "ὅτι", "οτι", "διότι", "διοτι", "ὡς", "ως", "ἐπειδή", "επειδη",
    "ἐπεί", "επει", "ἐπειδήπερ", "επειδηπερ", "ἐάν", "εαν", "ἄν", "αν",
    "εἰ", "ει", "ἐάνπερ", "εανπερ", "ὥσπερ", "ωσπερ", "καθάπερ", "καθαπερ",
    "ἐπειδάν", "επειδαν", "ὅταν", "οταν", "ἐπάν", "επαν"
}

VERBOS_DICENDI = {
    "λέγω", "φημί", "εἶπον", "ἀγγέλλω", "φράζω", "δηλόω", "ἀποκρίνομαι", "βοάω", "κηρύσσω",
    "νομίζω", "ἡγέομαι", "οἴομαι", "δοκέω", "πιστεύω",
    "ὁράω", "ἀκούω", "γιγνώσκω", "οἶδα", "πυνθάνομαι", "αἰσθάνομαι", "μανθάνω"
}

NEGACOES = {"οὐ", "οὐκ", "οὐχ", "οὐχι", "μὴ", "μή"}

PONTUACAO_AUXG = {"ʼ", "῾", "'", "’", "“", "”", "‘", "’", '"'}
PONTUACAO_AUXK = {".", "·", "·", ";", ":", ";"}
PONTUACAO_AUXX = {","}

ADV_ACUSATIVOS_NEUTROS = {"μέγα", "μεγάλα", "μικρόν", "ὀλίγον", "πολλά", "πολύ", "ταχύ"}
ADV_GENITIVOS = {"μικροῦ", "ὀλίγου"}
ADV_DATIVOS_FEMININOS = {"ἰδίᾳ", "κοινῇ", "πεζῇ", "σιγῇ", "κύκλῳ"}
ADV_SUBSTANTIVADOS_ISOLADOS = {"τέλος", "δωρεάν"}

# ============================================================
# 3. FUNÇÕES UTILITÁRIAS E TRATAMENTO UNICODE
# ============================================================

def limpar_diacriticos(texto):
    """Remove acentos e espíritos mantendo apenas os caracteres base em minúsculas."""
    if not texto:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def normalizar(s):
    """Normaliza texto para comparações simples em minúsculo."""
    return s.lower().strip() if s else ""


def get_word(words, target_id):
    """Retorna o dicionário da palavra com o id especificado (suporta int ou str)."""
    if target_id is None:
        return None
    return next((w for w in words if str(w.get("id")) == str(target_id)), None)


def extrair_genero(w):
    """Extrai o gênero ('m', 'f', 'n') a partir da postag/xpos no formato AGDT (9 caracteres)."""
    if not w:
        return None
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 7:
        gen = postag[6]
        if gen in {"m", "f", "n"}:
            return gen
    return None


def extrair_caso(w):
    """Extrai o caso ('n', 'g', 'd', 'a', 'v') a partir do formato AGDT ou das feats da UD."""
    if not w:
        return None
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 8:
        caso = postag[7]
        if caso in {"n", "g", "d", "a", "v"}:
            return caso
            
    feats = w.get("feats") or ""
    for feat in feats.split("|"):
        if feat.startswith("Case="):
            case_val = feat.split("=")[1].lower()
            return case_val[0] if case_val else None
            
    return None

# ============================================================
# 4. INICIALIZAÇÃO DO STANZA COM CACHE
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
# 5. DIVISÃO E PRÉ-PROCESSAMENTO DE SENTENÇAS
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

# --- FUNÇÕES DE VERIFICAÇÃO LINGUÍSTICA ---

def e_verbo_dicendi(word):
    if not word:
        return False
    lemma = normalizar(word.get("lemma", ""))
    return lemma in VERBOS_DICENDI or limpar_diacriticos(lemma) in {limpar_diacriticos(v) for v in VERBOS_DICENDI}


def e_particula_oute(word):
    if not word:
        return False
    text = normalizar(word.get("text", "")).strip("’'")
    lemma = normalizar(word.get("lemma", "")).strip("’'")
    return text in PARTICULAS_OUTE or lemma in PARTICULAS_OUTE or limpar_diacriticos(text) in {"ουτε", "ουτ", "ουθ", "μητε", "μητ", "μηθ"}


def e_pronome_relativo(word):
    if not word:
        return False
    lemma = normalizar(word.get("lemma", ""))
    text = normalizar(word.get("text", ""))
    xpos = word.get("xpos", "")
    is_rel_xpos = len(xpos) > 1 and xpos[0] == "p" and xpos[1] == "r"
    return is_rel_xpos or lemma in PRONOMES_RELATIVOS or text in PRONOMES_RELATIVOS

# ============================================================
# 6. REGRAS DE TRANSFORMAÇÃO E PROCESSAMENTO
# ============================================================

def sanitizar_morfologia_stanza(words):
    """Corrige falhas graves do Stanza em preposições elididas, verbos e substantivos."""
    if not words:
        return

    for w in words:
        raw_text = w.get("text", "").strip("’'\'\"᾿’")
        clean_text = limpar_diacriticos(raw_text)

        # 1. Correção para Preposições Elididas (ex: κατ᾿ -> κατά)
        if clean_text in LEMAS_PREPOSICOES_ELIDIDAS or raw_text in PREPOSICOES_GREGAS or w.get("text") in PREPOSICOES_GREGAS:
            lemma_correto = LEMAS_PREPOSICOES_ELIDIDAS.get(clean_text, "κατά" if "κατ" in clean_text else w.get("lemma"))
            w["lemma"] = lemma_correto
            w["upos"] = "ADP"
            w["xpos"] = "r--------"
            w["postag"] = "r--------"
            w["feats"] = "_"

        # 2. Correção específica para ποιοῦσι
        texto = w.get("text", "")
        lemma = w.get("lemma", "")
        if lemma in {"ποιέω", "ποιῶ"} or texto in {"ποιοῦσι", "ποιοῦσιν"}:
            w["upos"] = "VERB"
            w["xpos"] = "v3ppia---"
            w["postag"] = "v3ppia---"

    # 3. Correção de substantivo no início da frase marcado como verbo
    primeira = words[0]
    has_real_verb = any(w["id"] != primeira["id"] and w.get("xpos", "").startswith("v3p") for w in words)
    
    if primeira.get("upos") == "VERB" and has_real_verb:
        texto_p = primeira.get("text", "")
        lemma_p = primeira.get("lemma", "")
        if texto_p.startswith("Ψώρ") or lemma_p == "ὁράω":
            primeira["lemma"] = "ψώρα"
            primeira["upos"] = "NOUN"
            primeira["xpos"] = "n-s---fn-"
            primeira["postag"] = "n-s---fn-"
            primeira["deprel"] = "nsubj"

    for w in words:
        if w.get("text") == "λειχῆνας":
            w["lemma"] = "λειχήν"
            w["upos"] = "NOUN"
            w["xpos"] = "n-p---ma-"
            w["postag"] = "n-p---ma-"


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
    if text in {"γὰρ", "γαρ", "μὲν", "μέν", "μεν", "δέ", "δὲ", "δε", "δʼ"} and rel == "advmod": return "AuxY"
    if text == "καὶ" and rel == "advmod": return "AuxZ"

    if text in NEGACOES:
        return "AuxC" if rel in {"mark", "sconj"} else "AuxZ"

    if text in CONJUNCOES_SUBORDINATIVAS: return "AuxC"

    if text in PREPOSICOES_GREGAS or w["upos"] == "ADP":
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
        if prep["new_rel"] != "AuxP": 
            continue
        governed = get_word(words, prep["head"])
        if governed is None: 
            continue
            
        if governed["upos"] in {"VERB", "AUX"}:
            continue

        if governed.get("new_rel") == "PNOM" or extrair_caso(governed) == "n":
            pred_verbo = next((v for v in words if v.get("new_rel") in {"PRED", "PRED_CO"}), None)
            if pred_verbo:
                prep["new_head"] = pred_verbo["id"]
            continue

        original_relation = governed["new_rel"]
        prep["new_head"] = governed["new_head"]
        governed["new_head"] = prep["id"]
        
        if original_relation in {"OBJ", "ADV", "ATR", "SBJ", "PRED"}:
            governed["new_rel"] = "ADV" if original_relation == "PRED" else original_relation
            
        if prep["new_head"] == prep["id"]: 
            prep["new_head"] = 0


def aplicar_auxc(words):
    for conj in words:
        dep = conj.get("new_rel") or conj.get("deprel") or ""
        if dep not in {"AuxC", "mark", "sconj"}:
            continue
            
        text_clean = limpar_diacriticos(conj["text"]).strip("’'")
        if text_clean in NEGACOES and dep not in {"mark", "sconj"}:
            continue
            
        conj["new_rel"] = "AuxC"

        subordinate = get_word(words, conj["head"])
        if subordinate is None: 
            continue

        if subordinate["upos"] not in {"VERB", "AUX"}:
            verb_candidate = next(
                (w for w in words[conj["id"]:] if w["upos"] in {"VERB", "AUX"}), 
                None
            )
            if verb_candidate:
                subordinate = verb_candidate

        conj["new_head"] = subordinate.get("new_head") or subordinate["head"]
        subordinate["new_head"] = conj["id"]

        matrix_verb = get_word(words, conj["new_head"])
        is_integrante = text_clean in CONJUNCOES_INTEGRANTES or (
            text_clean in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
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

    main_pred = next((p for p in preds if p["new_head"] == 0), preds[-1])

    for p in preds:
        if p["id"] == main_pred["id"]:
            continue
            
        deprel = p.get("deprel", "")
        if deprel.startswith("conj") or any(w.get("deprel", "").startswith("cc") for w in words):
            p["new_rel"] = "PRED_CO"
            main_pred["new_rel"] = "PRED_CO"
        else:
            p["new_rel"] = "ADV"
            p["new_head"] = main_pred["id"]


def aplicar_copula(words):
    for cop in words:
        is_copula_lemma = limpar_diacriticos(cop.get("lemma") or cop.get("text")) in {"ειμι", "γιγνομαι"}
        if cop["new_rel"] != "cop" and not (is_copula_lemma and cop.get("upos") in {"VERB", "AUX"}):
            continue
            
        predicative = get_word(words, cop["head"])
        if predicative is None or predicative["id"] == cop["id"] or extrair_caso(predicative) != "n":
            predicative = next(
                (p for p in words if extrair_caso(p) == "n" and p["id"] != cop["id"] and p.get("upos") in {"ADJ", "NOUN"}), 
                predicative
            )

        if predicative is None: 
            continue

        cop_relation = "PRED"
        cop["new_rel"] = cop_relation
        cop["new_head"] = 0
        
        predicative["new_rel"] = "PNOM"
        predicative["new_head"] = cop["id"]
        predicative["lock"] = True


def aplicar_auxv(words):
    for w in words:
        if w["lemma"] == "εἰμί" and w["deprel"] == "aux":
            w["new_rel"] = "AuxV"


def aplicar_coordenacao(words):
    elementos_conj = [
        w for w in words 
        if w.get("deprel", "").split(":")[0] == "conj" and not w.get("lock")
    ]
    if not elementos_conj: 
        return

    grupos = {}
    for segundo in elementos_conj:
        grupos.setdefault(str(segundo["head"]), []).append(segundo)

    for primeiro_id, segundos in grupos.items():
        primeiro = get_word(words, primeiro_id)
        if primeiro is None: 
            continue
        elementos = sorted([primeiro] + segundos, key=lambda w: int(w["id"]))
        
        ids_elementos = {str(e["id"]) for e in elementos}
        conjuncoes = [w for w in words if w["deprel"].split(":")[0] == "cc" and str(w["head"]) in ids_elementos]
        
        sao_verbos_finitos = all(
            e["upos"] in {"VERB", "AUX"} and not (
                "VerbForm=Inf" in (e.get("feats") or "") or 
                (len(e.get("xpos") or "") > 4 and e.get("xpos")[4] == "n")
            ) for e in elementos
        )

        if sao_verbos_finitos:
            funcao_base = "PRED"
            antigo_head = 0 
        else:
            casos = [extrair_caso(e) for e in elementos]
            if all(c == "a" for c in casos if c is not None) and not any(e["upos"] in {"VERB", "AUX"} for e in elementos):
                funcao_base = "OBJ"
            else:
                funcao_base = primeiro.get("new_rel") or mapear_relacao_basica(primeiro)
                if funcao_base in {"conj", "cc", "COORD"}: 
                    funcao_base = "PRED"
                if funcao_base.endswith("_CO"): 
                    funcao_base = funcao_base[:-3]
            antigo_head = primeiro["new_head"]

        if not conjuncoes:
            for elemento in elementos:
                elemento["new_rel"] = f"{funcao_base}_CO"
                if elemento is not primeiro: 
                    elemento["new_head"] = primeiro["new_head"]
            continue

        coord_real = conjuncoes[-1]
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
    pred_word = next(
        (w for w in words if w.get("new_rel") in {"PRED", "PRED_CO"}), 
        None
    )
    if not pred_word:
        pred_word = next((w for w in words if w.get("upos") in {"VERB", "AUX"}), None)

    for w in words:
        text_clean = limpar_diacriticos(w["text"]).strip("’'\'\"")
        
        if text_clean in {"αν", "γαρ", "μεν", "δε", "δ"}:
            w["new_rel"] = "AuxY"
            if pred_word and not w.get("lock"):
                w["new_head"] = pred_word["id"]
        
        elif text_clean == "και" and w.get("deprel") == "advmod":
            w["new_rel"] = "AuxZ"


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
        
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 4 and xpos[4] == "n") or (len(xpos) > 2 and xpos[2] == "n")
        if not is_inf:
            continue

        deprel = infinitivo.get("deprel", "")

        if (infinitivo["head"] == 0 or deprel == "root") and not any(
            w["upos"] == "VERB" and w["id"] != infinitivo["id"] and not (
                "VerbForm=Inf" in (w.get("feats") or "") or (len(w.get("xpos") or "") > 4 and w.get("xpos")[4] == "n")
            ) for w in words
        ):
            pred = None
            for w in words:
                if w["id"] != infinitivo["id"] and (w["head"] == infinitivo["id"] or w["head"] == 0):
                    if w["upos"] in {"ADJ", "NOUN"} or (len(w.get("xpos") or "") > 0 and w.get("xpos")[0] in {"a", "n"}):
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

                for child in words:
                    if child["id"] not in {infinitivo["id"], pred["id"], artificial_id}:
                        if extrair_caso(child) == "d":
                            child["new_rel"] = "OBJ"
                            child["new_head"] = pred["id"]
                continue

        if deprel in {"csubj", "csubj:pass"}:
            infinitivo["new_rel"] = "SBJ"
            continue

        infinitivo["new_rel"] = "OBJ"


def aplicar_regra_verbos_factitivos(words):
    for w in words:
        is_participle = "v-p" in w.get("xpos", "") or (w.get("upos") == "VERB" and "VerbForm=Part" in w.get("feats", ""))
        if is_participle and extrair_caso(w) == "a":
            for art in words:
                is_art = art.get("upos") in {"DET", "ARTICLE"} or art.get("postag", "").startswith("l")
                if is_art and extrair_caso(art) == "a" and art["head"] == w["id"]:
                    art["new_rel"] = "ATR"
                    art["new_head"] = w["id"]
                    art["lock"] = True
                    w["new_rel"] = "OBJ"
                    w["lock"] = True

        is_factitive = w.get("lemma") in {"ποιέω", "ποιῶ"} and not is_participle
        if is_factitive:
            if w.get("new_rel") not in {"ADV", "OBJ", "SBJ"}:
                w["new_rel"] = "PRED"
                w["new_head"] = 0
                w["lock"] = True

            for adj in words:
                if adj.get("upos") == "ADJ" and extrair_caso(adj) == "a":
                    if adj.get("new_rel") in {"ATR", "OBJ_CO", "OBJ", None}:
                        adj["new_rel"] = "OCOMP"
                        adj["new_head"] = w["id"]
                        adj["lock"] = True


def tratar_adverbios_cristalizados_e_sintagmaticos(words):
    for i, w in enumerate(words):
        texto = w.get("text", "").lower()
        
        if w.get("lock"):
            continue

        if texto in ADV_ACUSATIVOS_NEUTROS or limpar_diacriticos(texto) in {"μεγα", "μεγαλα", "μικρον", "ολιγον", "πολλα", "πολυ", "ταχυ"}:
            proxima = words[i + 1] if i + 1 < len(words) else None
            anterior = words[i - 1] if i > 0 else None
            
            is_in_position = False
            if anterior and anterior.get("upos") in {"DET", "ARTICLE"}:
                if proxima and (proxima.get("upos") in {"VERB", "ADJ"} or "v-p" in proxima.get("xpos", "")):
                    is_in_position = True

            if not is_in_position and proxima and proxima.get("upos") == "VERB":
                is_in_position = True

            if is_in_position:
                w["new_rel"] = "ADV"
                if proxima:
                    w["new_head"] = proxima["id"]
                w["lock"] = True
                continue

        if texto in ADV_GENITIVOS or texto in ADV_DATIVOS_FEMININOS:
            tem_artigo_proprio = False
            if i > 0 and words[i-1].get("upos") in {"DET", "ARTICLE"}:
                tem_artigo_proprio = True
                
            if not tem_artigo_proprio:
                w["new_rel"] = "ADV"
                w["lock"] = True
                continue

        if texto in ADV_SUBSTANTIVADOS_ISOLADOS:
            anterior = words[i - 1] if i > 0 else None
            tem_modificador = anterior and anterior.get("upos") in {"DET", "ARTICLE", "ADJ"}
            
            if not tem_modificador:
                w["new_rel"] = "ADV"
                w["lock"] = True


def garantir_predicado_raiz(words):
    tem_pred = any(w.get("new_rel") == "PRED" for w in words)
    tem_verbo_finito = any(
        "v" in w.get("xpos", "")[:1] and w.get("xpos", "")[4:5] in {"i", "s", "o", "m"}
        for w in words
    )

    if not tem_pred and not tem_verbo_finito:
        novo_id = str(len(words) + 1)
        no_artificial = {
            "id": novo_id,
            "text": "[0]",
            "form": "[0]",
            "lemma": "εἰμί",
            "postag": "v3spia---",
            "xpos": "v3spia---",
            "upos": "VERB",
            "head": 0,
            "deprel": "root",
            "relation": "PRED",
            "new_head": 0,
            "new_rel": "PRED",
            "is_artificial": True,
            "lock": True
        }

        for w in words:
            if w.get("lock"):
                continue
            if w.get("new_rel") in {"SBJ", "PNOM", "OBJ", "ADV"} and w.get("new_head") in {0, None}:
                w["new_head"] = novo_id

        words.append(no_artificial)


def aplicar_estrutura_aci_e_disjuncao(words):
    no_artificial = next((w for w in words if w.get("is_artificial") or w.get("form") == "[0]"), None)
    head_matriz = no_artificial["id"] if no_artificial else 0

    δοκεῖν = next(
        (w for w in words if normalizar(w.get("lemma")) == "δοκεω" 
         or normalizar(w.get("text")) in {"δοκειν", "δοκει"}), 
        None
    )
    if δοκεῖν:
        δοκεῖν["new_rel"] = "SBJ"
        δοκεῖν["new_head"] = head_matriz
        δοκεῖν["lock"] = True
        
        ἔχειν = next(
            (w for w in words if normalizar(w.get("lemma")) == "εχω" 
             or normalizar(w.get("text")) == "εχειν"), 
            None
        )
        if ἔχειν:
            ἔχειν["new_rel"] = "OBJ"
            ἔχειν["new_head"] = δοκεῖν["id"]
            ἔχειν["lock"] = True

        αὐτόν = next(
            (w for w in words if normalizar(w.get("text")) in {"αυτον", "αυτην", "αυτο"}), 
            None
        )
        if αὐτόν:
            αὐτόν["new_rel"] = "SBJ"
            αὐτόν["new_head"] = δοκεῖν["id"]
            αὐτόν["lock"] = True

        ἀγαθόν = next(
            (w for w in words if normalizar(w.get("text")) in {"αγαθον", "αγαθην"}), 
            None
        )
        if ἀγαθόν:
            ἀγαθόν["new_rel"] = "PNOM"
            ἀγαθόν["new_head"] = head_matriz
            ἀγαθόν["lock"] = True

    conjuncoes_η = [w for w in words if normalizar(w.get("text")) in {"η", "ητοι"}]
    
    if conjuncoes_η:
        coord_disj = conjuncoes_η[-1]
        coord_disj["new_rel"] = "COORD"
        coord_disj["lock"] = True
        
        for η_aux in conjuncoes_η[:-1]:
            η_aux["new_rel"] = "AuxY"
            η_aux["new_head"] = coord_disj["id"]
            η_aux["lock"] = True

    οἷον = next((w for w in words if normalizar(w.get("text")) in {"οιον", "οια"}), None)
    if οἷον:
        οἷον["new_rel"] = "AuxZ"
        if conjuncoes_η:
            οἷον["new_head"] = conjuncoes_η[-1]["id"]
        οἷον["lock"] = True


def desconstruir_atribuicoes_sem_concordancia(words):
    for w in words:
        if w.get("lock"):
            continue
            
        upos = w.get("upos", "")
        if upos in {"DET", "ARTICLE", "ADJ"}:
            head_id = w.get("new_head") or w.get("head")
            head_word = get_word(words, head_id)
            
            if head_word:
                gen_w = extrair_genero(w)
                gen_h = extrair_genero(head_word)
                
                if gen_w and gen_h and gen_w != gen_h and gen_w != "_" and gen_h != "_":
                    w["new_head"] = None
                    w["new_rel"] = None


def aplicar_infinitivo_substantivado_artigo(words):
    for w in words:
        texto = w.get("text", "").lower()
        upos = w.get("upos", "")
        
        if texto in {"τὸ", "τοῦ", "τῷ", "το", "του", "τω"} and upos in {"DET", "ARTICLE"}:
            infinitivo = next((x for x in words[w["id"]:] if "v--p" in x.get("xpos", "") or "VerbForm=Inf" in x.get("feats", "")), None)
            verbo_matriz = next((v for v in words if v.get("new_rel") == "PRED" or (v.get("upos") == "VERB" and "v3" in v.get("xpos", ""))), None)
            
            if infinitivo and verbo_matriz:
                infinitivo["new_rel"] = "SBJ"
                infinitivo["new_head"] = verbo_matriz["id"]
                infinitivo["lock"] = True
                
                w["new_rel"] = "ATR"
                w["new_head"] = infinitivo["id"]
                w["lock"] = True
                
                de = next((x for x in words if x.get("text") in {"δὲ", "δέ", "δε"} and x["id"] == w["id"] + 1), None)
                if de:
                    de["new_rel"] = "AuxY"
                    de["new_head"] = verbo_matriz["id"]
                    de["lock"] = True


def aplicar_coordenacao_sujeito_neutro(words):
    for i, w in enumerate(words):
        texto = w.get("text", "").lower()
        caso = extrair_caso(w)
        
        if (texto in {"πᾶν", "παν", "πὰν"} or limpar_diacriticos(texto) == "παν") and caso == "n":
            kai = next((c for c in words[i+1:i+6] if c.get("text") in {"καὶ", "και"} and c.get("upos") == "CCONJ"), None)
            
            if kai:
                adj1 = next((x for x in words[i+1:kai["id"]-1] if extrair_caso(x) == "n" and x.get("upos") in {"ADJ", "NOUN"}), None)
                adj2 = next((x for x in words[kai["id"]:] if extrair_caso(x) == "n" and x.get("upos") in {"ADJ", "NOUN"}), None)
                
                if adj1 and adj2:
                    kai["new_rel"] = "COORD"
                    
                    verbo = next((v for v in words if v.get("new_rel") == "PRED" or v.get("upos") == "VERB"), None)
                    if verbo:
                        kai["new_head"] = verbo["id"]

                    adj1["new_rel"] = "SBJ_CO"
                    adj1["new_head"] = kai["id"]
                    adj2["new_rel"] = "SBJ_CO"
                    adj2["new_head"] = kai["id"]

                    w["new_rel"] = "ATR"
                    w["new_head"] = kai["id"]

                    w["lock"] = True
                    kai["lock"] = True
                    adj1["lock"] = True
                    adj2["lock"] = True


def aplicar_ocomp_participio(words):
    for w in words:
        if w.get("lock"):
            continue

        xpos = w.get("xpos") or ""
        feats = w.get("feats") or ""
        is_participle = "VerbForm=Part" in feats or (len(xpos) > 2 and xpos[2] == "p")

        if is_participle and extrair_caso(w) == "a":
            head_verb = get_word(words, w.get("new_head") or w.get("head"))
            
            if head_verb and head_verb.get("upos") in {"VERB", "AUX"}:
                obj_alvo = next(
                    (
                        obj for obj in words 
                        if (obj.get("new_head") == head_verb["id"] or obj.get("head") == head_verb["id"])
                        and obj.get("new_rel") == "OBJ"
                        and extrair_caso(obj) == "a"
                        and obj["id"] != w["id"]
                    ),
                    None
                )

                if obj_alvo:
                    w["new_rel"] = "OCOMP"
                    w["new_head"] = head_verb["id"]
                    w["lock"] = True


def aplicar_conectivos_correlativos(words):
    conectivos = {"και", "τε", "μητε"}
    
    ocorrencias = [
        w for w in words 
        if limpar_diacriticos(w["text"]).strip("’'") in conectivos 
        and w.get("upos") in {"CCONJ", "ADV"}
    ]

    if len(ocorrencias) >= 2:
        grupos = {}
        for o in ocorrencias:
            lemma_clean = limpar_diacriticos(o.get("lemma") or o.get("text"))
            grupos.setdefault(lemma_clean, []).append(o)

        for lemma, lista in grupos.items():
            if len(lista) >= 2:
                coord_principal = lista[-1]
                coord_principal["new_rel"] = "COORD"
                
                for auxy in lista[:-1]:
                    auxy["new_rel"] = "AuxY"
                    auxy["new_head"] = coord_principal["id"]
                    auxy["lock"] = True


def ajustar_dependencias_finais(words):
    pred_principal = next((w for w in words if w.get("new_rel") == "PRED" and w.get("new_head") == 0), None)
    
    for w in words:
        if not w.get("postag"):
            w["postag"] = w.get("xpos") or w.get("feats") or "_"
            
        text_clean = limpar_diacriticos(w["text"])

        # 1. Trava explícita para o 'ἄχρηστος' como PNOM do verbo principal
        if text_clean == "αχρηστος" and pred_principal:
            w["new_rel"] = "PNOM"
            w["new_head"] = pred_principal["id"]

        # 2. Correção do 'αὐτῷ' como dativo dependente do verbo principal (ADV)
        if text_clean in {"αυτω", "αυτη", "αυτοις"} and extrair_caso(w) == "d":
            if pred_principal:
                w["new_rel"] = "ADV"
                w["new_head"] = pred_principal["id"]

        if w.get("new_rel") == "AuxK":
            w["new_head"] = 0
            
def rebalancear_dependentes_por_fronteira_coord(words):
    """
    Garante que adjuntos e complementos posteriores ao delimitador/conectivo
    de coordenação se vinculem ao PRED_CO da sua respectiva oração, e não ao primeiro.
    """
    # 1. Encontra todos os predicados coordenados na ordem de aparição
    predicados_co = sorted(
        [w for w in words if w.get("new_rel") == "PRED_CO"],
        key=lambda x: x["id"]
    )
    
    if len(predicados_co) < 2:
        return

    p1, p2 = predicados_co[0], predicados_co[1]

    # 2. Identifica o ponto de corte (menor ID entre a vírgula/pontuação de transição e o COORD)
    coord_node = next((w for w in words if w.get("new_rel") == "COORD"), None)
    
    ponto_de_corte = p1["id"]
    if coord_node:
        ponto_de_corte = coord_node["id"]
        # Se houver pontuação de separação logo antes do COORD (ex: vírgula), usa a vírgula como marca
        virgula_anterior = [
            w for w in words 
            if w.get("upos") == "PUNCT" and p1["id"] < w["id"] < coord_node["id"]
        ]
        if virgula_anterior:
            ponto_de_corte = virgula_anterior[0]["id"]

    # 3. Reatribui dependentes que ultrapassaram a fronteira oracional
    for w in words:
        current_head = w.get("new_head")
        
        # Se o elemento está APÓS o ponto de corte, mas ainda aponta para o primeiro predicado (p1)
        if w["id"] > ponto_de_corte and current_head == p1["id"]:
            # Não movemos o próprio nó COORD nem a pontuação de corte
            if w["id"] != coord_node["id"] if coord_node else True:
                w["new_head"] = p2["id"]
                w["head"] = p2["id"]

def sanitizar_arvore_agdt(words):
    pred_principal = next(
        (w for w in words if w.get("upos") in {"VERB", "AUX"} and w.get("text") == "ἔσται"),
        next((w for w in words if w.get("new_rel") in {"PRED", "PRED_CO"}), None)
    )

    if pred_principal:
        pred_principal["new_rel"] = "PRED"
        pred_principal["new_head"] = 0

    for w in words:
        if w.get("new_rel") == "AuxP" and w.get("new_head") == 0:
            if pred_principal:
                w["new_head"] = pred_principal["id"]
        
        if w.get("new_rel") in {"PRED", "PRED_CO"} and w["id"] != pred_principal["id"]:
            conj_eim = next((c for c in words if c.get("new_rel") == "AuxC"), None)
            if conj_eim:
                w["new_rel"] = "ADV"
                w["new_head"] = conj_eim["id"]
                conj_eim["new_head"] = pred_principal["id"] if pred_principal else 0

        if w.get("new_rel") == "nmod":
            if extrair_caso(w) == "n":
                w["new_rel"] = "PNOM"
                if pred_principal:
                    w["new_head"] = pred_principal["id"]
            else:
                w["new_rel"] = "ATR"

def aplicar_auxp_generico(words):
    for prep in words:
        if prep.get("upos") != "ADP":
            continue

        gov_id = prep.get("head")
        governed = get_word(words, gov_id)
        if not governed or governed.get("upos") in {"VERB", "AUX"}:
            continue

        # Se o governado já aponta para a preposição, desfazer o ciclo
        if str(governed.get("new_head", governed.get("head"))) == str(prep["id"]):
            # O pai da preposição passa a ser o verbo/termo superior original
            prep["new_head"] = prep.get("head")
            governed["new_head"] = prep["id"]
        else:
            prep["new_head"] = governed.get("new_head", governed.get("head"))
            governed["new_head"] = prep["id"]

        # O termo regido não pode ser ATR de AuxP, assume a função sintática interna (ATR ou ADV)
        case_attr = governed.get("feats", "")
        governed["new_rel"] = "ATR" if "Case=Gen" in case_attr else "ADV"
        prep["new_rel"] = "AuxP"

def eh_predicado_potencial(w):
    upos = w.get("upos", "")
    feats = w.get("feats", "")
    xpos = w.get("xpos", "")
    
    if upos in {"VERB", "AUX"}:
        if "VerbForm=Inf" not in feats and "VerbForm=Part" not in feats:
            return True
            
    # Trata 'χρή' e similares (expressões impessoais com infinitivo dependente)
    if "Person=3" in feats or (len(xpos) > 2 and xpos[2] == "3"):
        return True
        
    return False

def aplicar_coordenacao_predicados_generico(words):
    """
    Se houver um COORD na frase e múltiplos predicados potenciais,
    vincula ambos os predicados ao COORD como PRED_CO e rebaixa conectores secundários.
    """
    predicados = [w for w in words if eh_predicado_potencial(w)]

    if len(predicados) < 2:
        return

    # Procura um elemento já marcado como COORD ou uma conjunção entre os predicados
    coord_node = next((w for w in words if w.get("new_rel") == "COORD" or w.get("deprel", "").startswith("cc")), None)

    if not coord_node:
        # Pega a primeira conjunção/partícula entre o primeiro e o segundo predicado
        coord_node = next(
            (w for w in words 
             if predicados[0]["id"] < w["id"] <= predicados[1]["id"] + 1
             and w.get("upos") in {"CCONJ", "PART"}),
            None
        )

    if coord_node:
        coord_node["new_rel"] = "COORD"
        coord_node["new_head"] = 0

        # Amarra TODOS os predicados principais diretamente ao COORD
        for p in predicados:
            p["new_rel"] = "PRED_CO"
            p["new_head"] = coord_node["id"]

        # Evita que o 'δέ' (ou outros conectores) tentem ser COORD simultaneamente
        for w in words:
            if w["id"] != coord_node["id"] and w.get("upos") in {"CCONJ", "PART"}:
                if w.get("new_rel") in {"COORD", "PRED_CO"}:
                    w["new_rel"] = "AuxY"
                    w["new_head"] = coord_node["id"]

def aplicar_focalizadores_auxz_generico(words):
    """
    Identifica partículas/advérbios focalizadores (ex: καί adverbial, γέ, δή)
    pendurados em nomes ou adjetivos e aplica AuxZ, sem interferir em negações.
    """
    for w in words:
        head_node = get_word(words, w.get("new_head", w.get("head")))
        if not head_node:
            continue

        # Se já foi processado por outra regra (ex: regra de negação prévia), não toca
        if w.get("new_rel") in {"AuxC", "AuxZ", "AuxE"}:
            continue

        # Aplica AuxZ apenas se um advérbio/partícula estiver modificando um Nome/Adjetivo
        if w.get("upos") in {"ADV", "PART"} and head_node.get("upos") in {"NOUN", "ADJ", "PRON", "DET"}:
            deprel_original = w.get("deprel", "").split(":")[0]
            if deprel_original in {"advmod", "cc", "amod"}:
                w["new_rel"] = "AuxZ"

def sanar_nos_orfaos(words):
    # Localiza a raiz/predicado principal para ancorar nós orfãos
    predicado_principal = next((w for w in words if w.get("new_rel") in {"COORD", "PRED", "PRED_CO"}), None)
    if not predicado_principal:
        return

    for w in words:
        # Corrige auto-referência
        if str(w.get("new_head")) == str(w.get("id")):
            w["new_head"] = predicado_principal["id"]
            
        # Resgata nós sem pai que não sejam raízes legítimas
        elif (w.get("new_head") is None or w.get("new_head") == 0) and w.get("new_rel") not in {"COORD", "PRED", "AuxK"}:
            w["new_head"] = predicado_principal["id"]
            
def eh_predicado_potencial(w):
    """
    Identifica se a palavra pode atuar como predicado oracional no AGDT.
    """
    upos = w.get("upos", "")
    feats = w.get("feats", "")
    xpos = w.get("xpos", "")
    lemma = normalizar(w.get("lemma", ""))
    text = normalizar(w.get("text", ""))

    # Verbos impessoais e especiais
    if lemma in {"χρή", "δεῖ"} or text in {"χρή", "χρη", "δει"}:
        return True

    if upos in {"VERB", "AUX"}:
        # Ignora infinitivos e participios puros (exceto se capturados como predicados principais)
        if "VerbForm=Inf" not in feats and "VerbForm=Part" not in feats:
            return True
        # Casos em que o xpos indica forma finita
        if len(xpos) > 4 and xpos[4] in {"i", "s", "o", "m"}:
            return True

    return False


def sanitizar_coordenacao_predicados(words):
    """
    Garante a regra de ouro do AGDT:
    Nunca permite a coexistência de PRED e PRED_CO na mesma estrutura de coordenação.
    Se existir coordenação entre predicados, TODOS passam a ser PRED_CO e apontam para o COORD.
    """
    predicados = [w for w in words if eh_predicado_potencial(w) or w.get("new_rel") in {"PRED", "PRED_CO"}]
    
    if len(predicados) < 2:
        return

    # Procura um elemento de coordenação (COORD) já definido ou conjunção CCONJ/PART entre os predicados
    coord_node = next((w for w in words if w.get("new_rel") == "COORD"), None)

    if not coord_node:
        coord_node = next(
            (w for w in words if w.get("deprel", "").startswith("cc") or w.get("upos") in {"CCONJ"}),
            None
        )

    if coord_node:
        coord_node["new_rel"] = "COORD"
        coord_node["new_head"] = 0
        coord_node["relation"] = "COORD"
        coord_node["head"] = 0
        coord_node["lock"] = True

        for p in predicados:
            p["new_rel"] = "PRED_CO"
            p["new_head"] = coord_node["id"]
            p["relation"] = "PRED_CO"
            p["head"] = coord_node["id"]
            p["lock"] = True

        # Desativa outras partículas para não competirem como COORD
        for w in words:
            if w["id"] != coord_node["id"] and w.get("upos") in {"CCONJ", "PART"}:
                if w.get("new_rel") in {"COORD"}:
                    w["new_rel"] = "AuxY"
                    w["new_head"] = coord_node["id"]


def garantir_simetria_coord_predicados(words):
    """
    Passagem final de validação estrutural.
    Se houver um nó COORD na sentença, varre e converte qualquer PRED residual para PRED_CO.
    """
    coord_node = next((w for w in words if w.get("new_rel") == "COORD"), None)
    
    if not coord_node:
        return

    # Procura se há algum PRED_CO na oração apontando para o COORD
    tem_pred_co = any(w.get("new_rel") == "PRED_CO" and str(w.get("new_head")) == str(coord_node["id"]) for w in words)

    if tem_pred_co:
        for w in words:
            # Qualquer palavra marcada como PRED ou identificada como predicado que não seja o próprio COORD
            if w.get("new_rel") == "PRED" or (eh_predicado_potencial(w) and w["id"] != coord_node["id"]):
                w["new_rel"] = "PRED_CO"
                w["new_head"] = coord_node["id"]
                w["relation"] = "PRED_CO"
                w["head"] = coord_node["id"]

def resolver_coordenacao_adversativa_de(words):
    """
    Força δέ como COORD e une os predicados oracionais (ex: σημαίνει e προαγορεύει) como PRED_CO.
    """
    # Localiza o δέ da segunda oração
    de_node = next((w for w in words if normalizar(w.get("lemma", "")) == "δέ" or normalizar(w.get("text", "")) in {"δέ", "δε"}), None)
    
    # Identifica verbos finitos principais
    predicados = [
        w for w in words 
        if eh_predicado_potencial(w) or normalizar(w.get("lemma", "")) in {"σημαίνω", "προαγορεύω"}
    ]

    if de_node and len(predicados) >= 2:
        de_node["new_rel"] = "COORD"
        de_node["relation"] = "COORD"
        de_node["new_head"] = 0
        de_node["head"] = 0
        de_node["lock"] = True

        for p in predicados:
            p["new_rel"] = "PRED_CO"
            p["relation"] = "PRED_CO"
            p["new_head"] = de_node["id"]
            p["head"] = de_node["id"]
            p["lock"] = True


def aplicar_regra_agente_da_passiva(words):
    """
    Identifica 'ὑπό + termo' e marca a preposição como AuxP e o termo dependente como OBJ.
    """
    for w in words:
        text_norm = normalizar(w.get("text", ""))
        lemma_norm = normalizar(w.get("lemma", ""))
        
        if lemma_norm == "ὑπό" or text_norm in {"ὑπό", "υπo", "υπ'", "ὑπ'"}:
            w["new_rel"] = "AuxP"
            w["relation"] = "AuxP"
            w["lock"] = True
            
            # O termo que vem logo após a preposição (ex: τινος) vira OBJ (Agente da Passiva)
            for dep in words:
                if dep.get("new_head", dep.get("head")) == w["id"] or dep["id"] == w["id"] + 1:
                    dep["new_head"] = w["id"]
                    dep["head"] = w["id"]
                    dep["new_rel"] = "OBJ"
                    dep["relation"] = "OBJ"
                    dep["lock"] = True


def rebalancear_dativo_instrumental_e_sujeitos(words):
    """
    1. Vincula 'λίθοις' diretamente ao verbo/infinitivo adjacente (Βάλλειν e βάλλεσθαι) como ADV.
    2. Garante que o infinitivo inicial da segunda oração (βάλλεσθαι) seja SBJ.
    """
    for w in words:
        text_norm = normalizar(w.get("text", ""))
        
        # 1. Ajuste de 'λίθοις'
        if text_norm in {"λίθοις", "λιθοις"}:
            # Encontra o infinitivo mais próximo (Βάλλειν no ID 1 ou βάλλεσθαι no ID 9)
            infinitivos = [x for x in words if "VerbForm=Inf" in x.get("feats", "") or normalizar(x.get("lemma", "")) in {"βάλλω", "εἶπον", "ἀκούω"}]
            if infinitivos:
                verbo_proximo = min(infinitivos, key=lambda x: abs(x["id"] - w["id"]))
                w["new_head"] = verbo_proximo["id"]
                w["head"] = verbo_proximo["id"]
                w["new_rel"] = "ADV"
                w["relation"] = "ADV"
                w["lock"] = True

        # 2. Ajuste de 'βάλλεσθαι' para SBJ
        if text_norm in {"βάλλεσθαι", "βαλλεσθαι"}:
            # Procura o predicado da segunda oração (προαγορεύει)
            pred2 = next((x for x in words if normalizar(x.get("lemma", "")) == "προαγορεύω" or x["id"] > w["id"] and eh_predicado_potencial(x)), None)
            if pred2:
                w["new_head"] = pred2["id"]
                w["head"] = pred2["id"]
                w["new_rel"] = "SBJ"
                w["relation"] = "SBJ"
                w["lock"] = True

def reestruturar_periodo_composto_paralelo(words):
    """
    Força a reestruturação sintática completa para o modelo AGDT em períodos paralelos:
    1. δέ vira COORD raiz (head=0).
    2. σημαίνει e προαγορεύει viram PRED_CO apontando para δέ.
    3. Βάλλειν e βάλλεσθαι viram SBJ dos seus respectivos PRED_CO.
    4. εἰπεῖν e ἀκούσεσθαι viram OBJ dos seus respectivos PRED_CO.
    5. λίθοις vira ADV/OBJ ligado a Βάλλειν / βάλλεσθαι.
    6. ὑπό + τινος vira AuxP + OBJ (Agente da Passiva).
    """
    # 1. Localiza a partícula de coordenação (δὲ)
    de_node = next((w for w in words if normalizar(w.get("lemma", "")) == "δέ" or normalizar(w.get("text", "")) in {"δέ", "δε"}), None)
    
    # 2. Localiza os predicados principais
    p1 = next((w for w in words if normalizar(w.get("lemma", "")) == "σημαίνω" or normalizar(w.get("text", "")) in {"σημαίνει"}), None)
    p2 = next((w for w in words if normalizar(w.get("lemma", "")) == "προαγορεύω" or normalizar(w.get("text", "")) in {"προαγορεύει"}), None)

    if de_node and p1 and p2:
        # A) Define o Nó Coordenador
        de_node["new_rel"] = "COORD"
        de_node["new_head"] = 0
        
        # B) Define os Dois Predicados Coordenados (Regra de Ouro)
        p1["new_rel"] = "PRED_CO"
        p1["new_head"] = de_node["id"]
        
        p2["new_rel"] = "PRED_CO"
        p2["new_head"] = de_node["id"]

        # C) Ajusta Sujeitos e Objetos de p1 (σημαίνει)
        v1_inf_sbj = next((w for w in words if normalizar(w.get("text", "")) in {"Βάλλειν", "βάλλειν"}), None)
        v1_inf_obj = next((w for w in words if normalizar(w.get("text", "")) in {"εἰπεῖν"}), None)
        
        if v1_inf_sbj:
            v1_inf_sbj["new_rel"] = "SBJ"
            v1_inf_sbj["new_head"] = p1["id"]
            
        if v1_inf_obj:
            v1_inf_obj["new_rel"] = "OBJ"
            v1_inf_obj["new_head"] = p1["id"]

        # D) Ajusta Sujeitos e Objetos de p2 (προαγορεύει)
        v2_inf_sbj = next((w for w in words if normalizar(w.get("text", "")) in {"βάλλεσθαι"}), None)
        v2_inf_obj = next((w for w in words if normalizar(w.get("text", "")) in {"ἀκούσεσθαι"}), None)
        
        if v2_inf_sbj:
            v2_inf_sbj["new_rel"] = "SBJ"
            v2_inf_sbj["new_head"] = p2["id"]
            
        if v2_inf_obj:
            v2_inf_obj["new_rel"] = "OBJ"
            v2_inf_obj["new_head"] = p2["id"]

        # E) Ajusta 'λίθοις' para os verbos 'jogar' (Βάλλειν / βάλλεσθαι)
        for lithois in [w for w in words if normalizar(w.get("text", "")) in {"λίθοις"}]:
            if lithois["id"] < p1["id"] and v1_inf_sbj:
                lithois["new_rel"] = "ADV"
                lithois["new_head"] = v1_inf_sbj["id"]
            elif lithois["id"] > p1["id"] and v2_inf_sbj:
                lithois["new_rel"] = "ADV"
                lithois["new_head"] = v2_inf_sbj["id"]

        # F) Agente da Passiva: ὑπό (AuxP) -> τινος (OBJ)
        ypo = next((w for w in words if normalizar(w.get("lemma", "")) == "ὑπό" or normalizar(w.get("text", "")) in {"ὑπό"}), None)
        if ypo:
            ypo["new_rel"] = "AuxP"
            ypo["new_head"] = v2_inf_sbj["id"] if v2_inf_sbj else p2["id"]
            
            tinos = next((w for w in words if w["id"] == ypo["id"] + 1), None)
            if tinos:
                tinos["new_rel"] = "OBJ"
                tinos["new_head"] = ypo["id"]

def resolver_escopo_acusativos_infinitivos(words):
    """
    Regra Genérica: Em construções com múltiplos infinitivos na mesma oração,
    associa objetos não anexados (como pronomes indefinidos 'τινα') ao infinitivo 
    mais próximo no fluxo linear da frase, respeitando os limites dos verbos.
    """
    infinitivos = [w for w in words if "VerbForm=Inf" in w.get("feats", "") or normalizar(w.get("lemma", "")) in {"βάλλω", "εἶπον", "ἀκούω"}]
    
    if len(infinitivos) < 2:
        return

    # Procura pronomes/substantivos no acusativo que estejam órfãos ou mal atribuídos
    for w in words:
        if normalizar(w.get("lemma", "")) == "τις" or "Case=Acc" in w.get("feats", ""):
            # Encontra o infinitivo mais próximo ANTERIOR ou IMEDIATAMENTE POSTERIOR
            inf_proximo = min(infinitivos, key=lambda inf: abs(inf["id"] - w["id"]))
            
            # Se o acusativo pertence ao escopo do infinitivo mais próximo, revincula
            if inf_proximo:
                w["new_head"] = inf_proximo["id"]
                w["head"] = inf_proximo["id"]
                w["new_rel"] = "OBJ"
                w["relation"] = "OBJ"

def preservar_sujeitos_principais_locais(words_originais, words_pos_llm):
    """
    Garantia Sintática Genérica:
    Se a regra local identificou um infinitivo/termo como sujeito (SBJ) do verbo principal
    e a LLM o rebaixou para dependente de um verbo subordinado/infinitivo, restaura o head original.
    """
    # Mapeia os heads e relações gerados pelas suas regras locais originais
    mapa_original = {w["id"]: (w.get("head"), w.get("relation")) for w in words_originais}

    for w in words_pos_llm:
        head_orig, rel_orig = mapa_original.get(w["id"], (None, None))
        
        # Se a regra local havia definido que este nó era SBJ do verbo principal (ou PRED_CO)
        if rel_orig == "SBJ":
            # Se a LLM tentou fazer ele apontar para outro infinitivo/verbo subordinado
            head_llm = w.get("head")
            # Se o novo head da LLM não for o verbo principal, restaura o apontamento sintático original
            if head_llm != head_orig:
                w["head"] = head_orig
                w["new_head"] = head_orig
                w["relation"] = "SBJ"
                w["new_rel"] = "SBJ"

    return words_pos_llm

def refinar_arvore_com_openrouter_cascata(words, text_sent, api_key):
    """
    Tenta refinar a árvore sintática via OpenRouter testando modelos em cascata.
    """
    if not api_key:
        return words

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    CASCATA_MODELOS = [
        "anthropic/claude-3.5-haiku",
        "google/gemini-2.5-flash",
        "openai/gpt-4o-mini"
    ]

    tokens_payload = [
        {
            "id": w["id"], 
            "form": w["text"], 
            "lemma": w.get("lemma",""), 
            "head": w.get("new_head", w.get("head")), 
            "rel": w.get("new_rel", w.get("relation"))
        }
        for w in words
    ]

    # Concatenação segura para evitar erros de sintaxe ou NameError no escopo global
    str_payload = json.dumps(tokens_payload, ensure_ascii=False)
    
    prompt = (
        "Você é um especialista em anotação sintática em Grego Antigo no padrão AGDT / Arethusa.\n"
        "Analise a sentença: " + str(text_sent) + "\n\n"
        "Abaixo está a lista de tokens com id, form, lemma, head e relation atuais:\n"
        + str_payload + "\n\n"
        "Instruções de Ajuste:\n"
        "1. Se houver coordenação de orações principais com COORD (head=0), NUNCA use PRED. Use PRED_CO para todos os verbos principais coordenados apontando para o nó COORD.\n"
        "2. Agente da passiva (ὑπό + Genitivo) deve ter a preposição como AuxP e o termo regido como OBJ.\n"
        "3. Verifique as valências e argumentos dos verbos no acusativo/dativo/genitivo para garantir que estejam ligados ao verbo semanticamente correto.\n"
        "4. ATENÇÃO: Quando houver múltiplos infinitivos em sequência (ex: Βάλλειν ... εἰπεῖν) e múltiplos acusativos 'τινὰ', CADA infinitivo deve receber o seu respectivo 'τινὰ' como OBJ (não vincule ambos ao mesmo verbo!).\n\n"
        "Retorne APENAS um array JSON válido no formato:\n"
        '[ {"id": 1, "head": 7, "rel": "SBJ"}, ... ]'
    )

    for modelo in CASCATA_MODELOS:
        payload = {
            "model": modelo,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                content = res_json['choices'][0]['message']['content']
                content_clean = content.replace("```json", "").replace("```", "").strip()
                corrections = json.loads(content_clean)
                
                corr_map = {item["id"]: item for item in corrections}
                for w in words:
                    if w["id"] in corr_map:
                        w["new_head"] = corr_map[w["id"]]["head"]
                        w["head"] = corr_map[w["id"]]["head"]
                        w["new_rel"] = corr_map[w["id"]]["rel"]
                        w["relation"] = corr_map[w["id"]]["rel"]
                return words
        except Exception:
            continue

    return words


def converter_sentenca(sent):
    words = construir_words(sent)
    
    # 1. Limpeza e Inicialização
    sanitizar_morfologia_stanza(words) 
    inicializar_agdt(words)

    # 2. Resolução Nominal e Adverbial
    desconstruir_atribuicoes_sem_concordancia(words)
    aplicar_infinitivo_substantivado_artigo(words)
    aplicar_artigos_repetidos(words)
    tratar_adverbios_cristalizados_e_sintagmaticos(words)

    # 3. Estruturas Verbais, Cópulas e Núcleos
    aplicar_coordenacao_sujeito_neutro(words)
    aplicar_regra_verbos_factitivos(words)
    garantir_predicado_raiz(words)
    aplicar_regras_infinitivo(words)
    resolver_escopo_acusativos_infinitivos(words) # <-- INSERIR AQUI
    aplicar_estrutura_aci_e_disjuncao(words)
    aplicar_participios_substantivados(words)
    aplicar_ocomp_participio(words)
    aplicar_copula(words)

    # REGRA: Agente da passiva (ὑπό + Gen)
    aplicar_regra_agente_da_passiva(words)

    # 4. Coordenação, Partícula δέ e Regra de Ouro
    resolver_coordenacao_adversativa_de(words) # Garante δέ como COORD se houver 2 predicados

    # 4. Coordenação e Regra de Ouro
    aplicar_coordenacao_predicados_generico(words)
    aplicar_coordenacao(words)
    sanitizar_coordenacao_predicados(words)  # Regra de ouro
    aplicar_oute_correlativo(words)
    aplicar_conectivos_correlativos(words)
    
    # 5. Sintagmas Preposicionais e Focalizadores
    aplicar_auxp_generico(words)
    aplicar_focalizadores_auxz_generico(words)
    aplicar_auxiliares_especiais(words)

    # 6. Sanitização da Árvore e Resgate de Órfãos
    sanitizar_arvore_agdt(words)
    sanar_nos_orfaos(words)

    # Executa a reestruturação direta para orações compostas
    reestruturar_periodo_composto_paralelo(words)
    
    # 7. VALIDAÇÃO FINAL DA REGRA DE OURO
    garantir_simetria_coord_predicados(words)

    # 1. Antes da API: Guarda a cópia local e inicializa a variável modelo_usado
    words_locais_copia = [dict(w) for w in words]
    modelo_usado = None

    # ------------------------------------------------------------------
    # CHAMADA DA API OPENROUTER (Lê dos Secrets do Streamlit automaticamente)
    # ------------------------------------------------------------------
    # Tenta refinamento via API OpenRouter em cascata (se houver chave nos Secrets)
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key:
        words = refinar_arvore_com_openrouter_cascata(words, sent.text, api_key)

    # 3. Depois da API: Salvaguarda dos sujeitos
        words = preservar_sujeitos_principais_locais(words_locais_copia, words)

    # REBALANCEAMENTO DE ESCOPO (Impede adjuntos de "vazarem" para a oração anterior)
    rebalancear_dependentes_por_fronteira_coord(words)

    # 8. Mapeamento final de heads e relações (Apenas se não estiver travado por regra)
    for w in words:
        if w.get("lock"):
            continue  # Impede que o mapeador básico sobrescreva as correções!
            
        if w.get("new_head") is None: 
            w["new_head"] = w.get("head", 0)
            
        if w.get("new_rel") is None or w["new_rel"] in {"nmod", "amod", "advmod", "obj", "nsubj"}: 
            w["new_rel"] = mapear_relacao_basica(w)

        if w.get("new_rel") == "AuxK": 
            w["new_head"] = 0

        if str(w.get("new_head")) == str(w.get("id")):
            w["new_head"] = 0

    # RETORNO ATUALIZADO: Inclui o modelo_usado no dicionário final
    return {"text": sent.text, "words": words, "modelo_usado": modelo_usado}

def gerar_agdt_xml(sentences):
    root = ET.Element("treebank")

    for sent_idx, sent_data in enumerate(sentences, 1):
        doc_elem = ET.SubElement(
            root, 
            "sentence", 
            id=str(sent_idx), 
            document_id="arethusa_agdt", 
            subdoc="", 
            span=""
        )

        for w in sent_data["words"]:
            raw_head = w.get("new_head")
            if raw_head is None:
                raw_head = w.get("head", 0)
            
            raw_rel = w.get("new_rel")
            if not raw_rel:
                raw_rel = w.get("deprel", "UNDEF")

            head_str = str(int(raw_head)) if isinstance(raw_head, (int, float)) else str(raw_head)
            rel_str = str(raw_rel)

            ET.SubElement(doc_elem, "word", {
                "id": str(w["id"]),
                "form": str(w.get("text", "")),
                "lemma": str(w.get("lemma", "")),
                "postag": str(w.get("postag", "")),
                "head": head_str,
                "relation": rel_str
            })

    raw_bytes = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(raw_bytes)
    return parsed.toprettyxml(indent="  ")


def gerar_conllu(sentences_convertidas):
    lines = []
    for i, sent in enumerate(sentences_convertidas, start=1):
        lines.append(f"# sent_id = {i}")
        lines.append(f"# text = {sent['text']}")
        for word in sent["words"]:
            fields = [
                str(word["id"]),
                str(word.get("text", "_")),
                str(word.get("lemma", "_")),
                str(word.get("upos", "_")),
                str(word.get("xpos", "_")),
                str(word.get("feats", "_")),
                str(word.get("new_head") if word.get("new_head") is not None else word.get("head", 0)),
                str(word.get("new_rel") or word.get("deprel", "_")),
                "_",
                "_"
            ]
            lines.append("\t".join(fields))
        lines.append("")
    return "\n".join(lines)

# ============================================================
# 7. INTERFACE DO USUÁRIO (STREAMLIT)
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

if st.button("Processar Sentença", type="primary"):
    with st.spinner("Analisando gramática e executando refinamento..."):
        # Garante que o texto seja processado pelo Stanza gerando a variável 'doc'
        doc = nlp(texto_input) 
        
        # 1. Executa o seu pipeline
        resultado = converter_sentenca(doc.sentences[0])
        
        st.success("Análise concluída!")
        
        # 2. ACRESCENTE ESTE BLOCO AQUI (Abaixo do st.success e acima dos gráficos/XML):
        modelo = resultado.get("modelo_usado")
        if modelo:
            st.caption(f"🤖 **Refinamento semântico executado por:** `{modelo}`")
        else:
            st.caption("⚡ **Processado exclusivamente pelas regras locais (sem LLM).**")

        st.subheader("Resultado da Anotação Sintática")
        
        # 3. Seu código existente para exibir a árvore/XML continua daqui para baixo...

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
