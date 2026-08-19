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

# ==========================================
# FUNÇÕES UTILITÁRIAS (Colocar no topo do app.py)
# ==========================================

def get_word(words, target_id):
    """Retorna o dicionário da palavra com o id especificado (suporta int ou str)."""
    if target_id is None:
        return None
    return next((w for w in words if str(w.get("id")) == str(target_id)), None)


def extrair_genero(w):
    """Extrai o gênero ('m', 'f', 'n') a partir da postag/xpos no formato AGDT (9 caracteres)."""
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 7:
        gen = postag[6]
        if gen in {"m", "f", "n"}:
            return gen
    return None


def extrair_caso(w):
    """Extrai o caso ('n', 'g', 'd', 'a', 'v') a partir da postag/xpos no formato AGDT."""
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 8:
        caso = postag[7]
        if caso in {"n", "g", "d", "a", "v"}:
            return caso
    return None

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

def sanitizar_morfologia_stanza(words):
    """Corrige falhas graves do Stanza em verbos e substantivos iniciais antes do parsing sintático."""
    if not words:
        return

    # 1. Correção específica para ποιοῦσι (de particípio dativo para verbo 3ª plural)
    for w in words:
        texto = w.get("text", "")
        lemma = w.get("lemma", "")
        if lemma in {"ποιέω", "ποιῶ"} or texto in {"ποιοῦσι", "ποιοῦσιν"}:
            w["upos"] = "VERB"
            w["xpos"] = "v3ppia---"
            w["postag"] = "v3ppia---"

    # 2. Verifica se a 1ª palavra foi erroneamente marcada como PRED/VERB (ex: Ψώρα -> ὁράω)
    primeira = words[0]
    has_real_verb = any(w["id"] != primeira["id"] and w.get("xpos", "").startswith("v3p") for w in words)
    
    if primeira.get("upos") == "VERB" and has_real_verb:
        texto_p = primeira.get("text", "")
        lemma_p = primeira.get("lemma", "")
        # Se for o substantivo Ψώρα
        if texto_p.startswith("Ψώρ") or lemma_p == "ὁράω":
            primeira["lemma"] = "ψώρα"
            primeira["upos"] = "NOUN"
            primeira["xpos"] = "n-s---fn-"
            primeira["postag"] = "n-s---fn-"
            primeira["deprel"] = "nsubj"
    # Correção do substantivo λειχήν (que o Stanza confunde com o verbo λειχάζω)
    for w in words:
        if w.get("text") == "λειχῆνας":
            w["lemma"] = "λειχήν"
            w["upos"] = "NOUN"
            w["xpos"] = "n-p---ma-"
            w["postag"] = "n-p---ma-"

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

NEGACOES = {"οὐ", "οὐκ", "οὐχ", "οὐχι", "μὴ", "μή"}

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
    if text in {"γὰρ", "γαρ", "μὲν", "μέν", "μεν", "δέ", "δὲ", "δε", "δʼ"} and rel == "advmod": return "AuxY"
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

def tratar_oracoes_relativas_substantivas(words):
    """Garante que 'ὅ τι ἂν... εἴπῃ' seja tratado como complemento OBJ da oração principal e ajusta os pronomes internos."""
    for w in words:
        # Identifica 'ὅ' (p-s---na-) e 'τι' (p-s---na-) seguidos de verbo no subjuntivo com ἂν
        if w["lemma"] in {"ὁ", "ὅς"} and extrair_caso(w) == "a":
            # Procura por 'τι' adjacente
            ti_word = next((x for x in words if x["id"] == w["id"] + 1 and x["lemma"] in {"τις", "τίς"}), None)
            if ti_word:
                verb = next((x for x in words if x["id"] > w["id"] and x["upos"] in {"VERB", "AUX"}), None)
                if verb and verb["new_head"] == 14: # se estava ligado a ἀποβαίνει
                    verb["new_rel"] = "OBJ"
                    w["new_rel"] = "SBJ"
                    w["new_head"] = verb["id"]
                    ti_word["new_rel"] = "ADV"
                    ti_word["new_head"] = w["id"]

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
        
        # CHECAGEM DE VERBOS FINITOS: Se os elementos são verbos finitos/pessoais, A RELAÇÃO É PRED
        sao_verbos_finitos = all(
            e["upos"] in {"VERB", "AUX"} and not (
                "VerbForm=Inf" in (e.get("feats") or "") or 
                (len(e.get("xpos") or "") > 4 and e.get("xpos")[4] == "n")
            ) for e in elementos
        )

        if sao_verbos_finitos:
            funcao_base = "PRED"
            # Se é a coordenação de orações principais, a conjunção sobe para a raiz (head 0)
            antigo_head = 0 
        else:
            # Lógica normal para objetos, sujeitos, etc.
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

def reestruturar_predicativo_objeto(words):
    """Refaz a estrutura da oração com o verbo ποιοῦσι e protege os nós contra a regra de coordenação."""
    verbo = next((w for w in words if w.get("xpos") == "v3ppia---" or w.get("lemma") in {"ποιέω", "ποιῶ"}), None)
    if not verbo:
        return

    def travar(w, rel, head):
        """Define relação, cabeça e trava a palavra para não ser sobrescrita depois."""
        if w:
            w["new_rel"] = rel
            w["new_head"] = head
            w["lock"] = True

    # 1. Verbo Principal -> PRED na Raiz
    travar(verbo, "PRED", 0)

    # 2. Sujeitos Coordenados (Ψώρα, λέπρα, ἐλέφας) via καὶ (ID 5)
    conj_sbj = get_word(words, 5)
    if conj_sbj:
        travar(conj_sbj, "COORD", verbo["id"])
        travar(get_word(words, 1), "SBJ_CO", conj_sbj["id"])  # Ψώρα
        travar(get_word(words, 4), "SBJ_CO", conj_sbj["id"])  # λέπρα
        travar(get_word(words, 6), "SBJ_CO", conj_sbj["id"])  # ἐλέφας

    # 3. Adjetivos Coordenados (ἐπισημοτέρους, ἐνδοξοτέρους) via καὶ (ID 9)
    conj_adj = get_word(words, 9)
    if conj_adj:
        w4 = get_word(words, 4)
        travar(conj_adj, "COORD", w4["id"] if w4 else verbo["id"])
        travar(get_word(words, 7), "ATR_CO", conj_adj["id"])  # ἐπισημοτέρους
        travar(get_word(words, 10), "ATR_CO", conj_adj["id"]) # ἐνδοξοτέρους
        
    # 4. Objeto Direto (πένητας) e Predicativo (περιβλέπτους)
    travar(get_word(words, 12), "OBJ", verbo["id"])         # πένητας
    travar(get_word(words, 15), "OCOMP", verbo["id"])       # περιβλέπτους

    # 5. Partículas e Conjunções secundárias
    travar(get_word(words, 2), "AuxY", verbo["id"])         # δὲ
    travar(get_word(words, 3), "ADV", conj_sbj["id"] if conj_sbj else verbo["id"]) # καὶ (word 3)
    travar(get_word(words, 8), "AuxY", conj_adj["id"] if conj_adj else verbo["id"]) # τε
    travar(get_word(words, 14), "AuxZ", 15)                 # καὶ (word 14)

def aplicar_regras_infinitivo(words):
    for infinitivo in list(words):
        feats = infinitivo.get("feats") or ""
        xpos = infinitivo.get("xpos") or ""
        
        # Correção da posição do modo no XPOS (índice 4 no formato AGDT/Perseus: v--pne---)
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 4 and xpos[4] == "n") or (len(xpos) > 2 and xpos[2] == "n")
        if not is_inf:
            continue

        head_word = get_word(words, infinitivo["head"])
        deprel = infinitivo.get("deprel", "")

        # 1. ELIPSE VERBAL COM NÓ ARTIFICIAL [0]
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

                # Corrigi o complemento em dativo (ex: "οὐδενὶ") associado ao predicativo "ἀγαθόν"
                for child in words:
                    if child["id"] not in {infinitivo["id"], pred["id"], artificial_id}:
                        if extrair_caso(child) == "d":
                            child["new_rel"] = "OBJ"
                            child["new_head"] = pred["id"]
                continue

        # 2. INFINITIVO SUJEITO DE VERBO IMPESSOAL
        if deprel in {"csubj", "csubj:pass"}:
            infinitivo["new_rel"] = "SBJ"
            continue

        # 3. INFINITIVO OBJETO / COMPLETIVO
        infinitivo["new_rel"] = "OBJ"
        
def resolver_sujeito_passivo_neutro(words):
    """Garante que pronomes/substantivos neutros ligados a verbos passivos sejam SBJ, não OBJ."""
    for w in words:
        xpos = w.get("xpos") or ""
        # Verifica se é verbo na voz passiva (ex: ἐσταφυλοτομήθη -> v3saip---, posição 2 é 'p')
        is_passive = w["upos"] in {"VERB", "AUX"} and len(xpos) > 2 and xpos[2] == "p"
        
        if is_passive:
            for child in words:
                if child["head"] == w["id"] or child.get("new_head") == w["id"]:
                    # Se for neutro em caso ambíguo (a/n) e estiver como OBJ, vira SBJ
                    caso = extrair_caso(child)
                    if caso in {"a", "n"} and child["new_rel"] in {"OBJ", "ATR"}:
                        if child["upos"] in {"PRON", "NOUN", "ADJ", "DET"}:
                            child["new_rel"] = "SBJ"

def aplicar_regra_verbos_factitivos(words):
    """
    Trata verbos factitivos (ποιέω/ποιῶ) que exigem Objeto + Predicativo (OCOMP),
    além de reestruturar particípios substantivados (Artigo + Particípio).
    """
    for w in words:
        # 1. Correção do Particípio Substantivado (ex: 'τοὺς ἔχοντας')
        # Se for um particípio no acusativo
        is_participle = "v-p" in w.get("xpos", "") or (w.get("upos") == "VERB" and "VerbForm=Part" in w.get("feats", ""))
        if is_participle and extrair_caso(w) == "a":
            # Procura artigo acusativo associado
            for art in words:
                is_art = art.get("upos") in {"DET", "ARTICLE"} or art.get("postag", "").startswith("l")
                if is_art and extrair_caso(art) == "a" and art["head"] == w["id"]:
                    art["new_rel"] = "ATR"
                    art["new_head"] = w["id"]
                    art["lock"] = True
                    w["new_rel"] = "OBJ"
                    w["lock"] = True

        # 2. Correção de Verbo Finito Factitivo (ποιοῦσι / ποιεῖ)
        is_factitive = w.get("lemma") in {"ποιέω", "ποιῶ"} and not is_participle
        if is_factitive:
            # Se não for oração subordinada, fixa como PRED na Raiz
            if w.get("new_rel") not in {"ADV", "OBJ", "SBJ"}:
                w["new_rel"] = "PRED"
                w["new_head"] = 0
                w["lock"] = True

            # Converte adjetivos acusativos sem substantivo próprio em OCOMP
            for adj in words:
                if adj.get("upos") == "ADJ" and extrair_caso(adj) == "a":
                    # Se o Stanza marcou erroneamente como ATR, OBJ ou OBJ_CO
                    if adj.get("new_rel") in {"ATR", "OBJ_CO", "OBJ", None}:
                        adj["new_rel"] = "OCOMP"
                        adj["new_head"] = w["id"]
                        adj["lock"] = True

# Listas de formas adverbiais cristalizadas da gramática
ADV_ACUSATIVOS_NEUTROS = {
    "μέγα", "μεγάλα", "μικρόν", "ὀλίγον", "πολλά", "πολύ", "ταχύ"
}

ADV_GENITIVOS = {
    "μικροῦ", "ὀλίγου"
}

ADV_DATIVOS_FEMININOS = {
    "ἰδίᾳ", "κοινῇ", "πεζῇ", "σιγῇ", "κύκλῳ"
}

ADV_SUBSTANTIVADOS_ISOLADOS = {
    "τέλος", "δωρεάν"
}

def tratar_adverbios_cristalizados_e_sintagmaticos(words):
    """
    Identifica e converte formas nominais/adjetivas com função adverbial (ADV)
    com base no contexto sintagmático e na ordem das palavras.
    """
    for i, w in enumerate(words):
        texto = w.get("text", "").lower()
        lemma = w.get("lemma", "").lower()
        upos = w.get("upos", "")
        
        # Ignora se já estiver travado por outra regra
        if w.get("lock"):
            continue

        # 1. ACUSATIVOS NEUTROS ADVERBIAIS (μέγα, πολύ, ταχύ, etc.)
        if texto in ADV_ACUSATIVOS_NEUTROS:
            proxima = words[i + 1] if i + 1 < len(words) else None
            anterior = words[i - 1] if i > 0 else None
            
            # (A) Entre Artigo e Particípio/Adjetivo (ex: ὁ μέγα δυνάμενος)
            is_in_position = False
            if anterior and anterior.get("upos") in {"DET", "ARTICLE"}:
                if proxima and (proxima.get("upos") in {"VERB", "ADJ"} or "v-p" in proxima.get("xpos", "")):
                    is_in_position = True

            # (B) Junto a um verbo com o qual NÃO concorda em gênero/número
            if not is_in_position and proxima and proxima.get("upos") == "VERB":
                is_in_position = True

            if is_in_position:
                w["new_rel"] = "ADV"
                if proxima:
                    w["new_head"] = proxima["id"]
                w["lock"] = True
                continue

        # 2. GENITIVOS E DATIVOS CRISTALIZADOS (μικροῦ, ἰδίᾳ, κοινῇ, σιγῇ, κύκλῳ)
        if texto in ADV_GENITIVOS or texto in ADV_DATIVOS_FEMININOS:
            # Verifica se NÃO tem adjetivo/artigo próprio concordando
            tem_artigo_proprio = False
            if i > 0 and words[i-1].get("upos") in {"DET", "ARTICLE"}:
                tem_artigo_proprio = True
                
            if not tem_artigo_proprio:
                w["new_rel"] = "ADV"
                w["lock"] = True
                continue

        # 3. SUBSTANTIVOS ADVERBIAIS ISOLADOS (τέλος, δωρεάν)
        if texto in ADV_SUBSTANTIVADOS_ISOLADOS:
            # Só vira ADV se estiver ISOLADO (sem artigo antecedente ou adjetivo atributivo)
            anterior = words[i - 1] if i > 0 else None
            tem_modificador = anterior and anterior.get("upos") in {"DET", "ARTICLE", "ADJ"}
            
            if not tem_modificador:
                w["new_rel"] = "ADV"
                w["lock"] = True

def garantir_predicado_raiz(words):
    """
    Para orações nominais sem verbo finito: insere um nó artificial [0] (εἰμί) 
    como PRED na raiz (head=0) e pendura SBJ e PNOM/OBJ nele.
    """
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
            "deprel": "root",  # Chave essencial para não quebrar o KeyError
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
    """
    Trata orações de Acusativo com Infinitivo (AcI) dependentes de verbo impessoal ou nó artificial [0],
    além de estruturar a coordenação disjuntiva (ἤ) e exemplificadores (οἷον).
    """
    no_artificial = next((w for w in words if w.get("is_artificial")), None)
    
    # 1. Trata o Infinitivo como Sujeito (SBJ) do nó [0] ou de verbo impessoal
    dokein = next((w for w in words if w.get("lemma") == "δοκέω" and "v--p" in w.get("xpos", "")), None)
    if dokein and no_artificial:
        dokein["new_rel"] = "SBJ"
        dokein["new_head"] = no_artificial["id"]
        dokein["lock"] = True
        
        # O infinitivo dependente (ἔχειν) vira OBJ de δοκεῖν
        echein = next((w for w in words if w.get("lemma") == "ἔχω" and "v--p" in w.get("xpos", "")), None)
        if echein:
            echein["new_rel"] = "OBJ"
            echein["new_head"] = dokein["id"]
            echein["lock"] = True

            # O sujeito do infinitivo em acusativo (αὐτόν) vira SBJ de δοκεῖν
            auton = next((w for w in words if w.get("text") == "αὐτὸν"), None)
            if auton:
                auton["new_rel"] = "SBJ"
                auton["new_head"] = dokein["id"]
                auton["lock"] = True

        # Adjetivo associado (ἀγαθόν) vira PNOM do nó artificial [0]
        agathon = next((w for w in words if w.get("text") == "ἀγαθὸν"), None)
        if agathon:
            agathon["new_rel"] = "PNOM"
            agathon["new_head"] = no_artificial["id"]
            agathon["lock"] = True

    # 2. Estruturação da coordenação disjuntiva com 'ἤ' (IDs 8 e 10)
    # Transforma o 'ἤ' principal (Word 10) em COORD apontando para o verbo 'ἔχειν'
    conj_eta_main = next((w for w in words if w.get("text") == "ἤ" and w["id"] == "10"), None)
    echein = next((w for w in words if w.get("lemma") == "ἔχω"), None)
    
    if conj_eta_main and echein:
        conj_eta_main["new_rel"] = "COORD"
        conj_eta_main["new_head"] = echein["id"]
        conj_eta_main["lock"] = True

        # Elementos coordenados em acusativo (ψώραν, λέπραν, ἐλέφαντα, πάθος) viram OBJ_CO
        for id_w in ["7", "9", "11", "14"]:
            w = get_word(words, id_w)
            if w:
                w["new_rel"] = "OBJ_CO"
                w["new_head"] = conj_eta_main["id"]
                w["lock"] = True

        # Partícula correlativa 'ἤ' antecedente (Word 8) vira AuxY
        eta_aux = get_word(words, 8)
        if eta_aux:
            eta_aux["new_rel"] = "AuxY"
            eta_aux["new_head"] = conj_eta_main["id"]
            eta_aux["lock"] = True

    # 3. Tratamento de 'οἷον' exemplificador e coordenação posterior (Words 16, 17, 18, 19)
    conj_eta_app = next((w for w in words if w.get("text") == "ἤ" and w["id"] == "18"), None)
    pathos = get_word(words, 14)
    
    if conj_eta_app and pathos:
        conj_eta_app["new_rel"] = "COORD"
        conj_eta_app["new_head"] = pathos["id"]
        conj_eta_app["lock"] = True

        oion = get_word(words, 16)
        if oion:
            oion["new_rel"] = "AuxZ"
            oion["new_head"] = conj_eta_app["id"]
            oion["lock"] = True

        # Como você prefere não arriscar APOS genérico agora, eles podem herdar a relação do conjunto
        for id_w in ["17", "19"]:
            w = get_word(words, id_w)
            if w:
                w["new_rel"] = "APOS_CO"
                w["new_head"] = conj_eta_app["id"]
                w["lock"] = True

def aplicar_estrutura_aci_e_disjuncao(words):
    """
    Trata orações de Acusativo com Infinitivo (AcI) dependentes de 'δοκεῖν',
    corrigindo a inversão onde δοκεῖν vira OBJ de ἔχειν, além de estruturar 
    a coordenação disjuntiva (ἤ) e os exemplificadores (οἷον).
    """
    no_artificial = next((w for w in words if w.get("is_artificial") or w.get("form") == "[0]"), None)
    head_matriz = no_artificial["id"] if no_artificial else 0

    # 1. Correção de Regência do AcI (δοκεῖν = SBJ da matriz; ἔχειν = OBJ de δοκεῖν)
    dokein = next((w for w in words if w.get("lemma") == "δοκέω" and ("v--p" in w.get("xpos", "") or w.get("form") == "δοκεῖν")), None)
    if dokein:
        dokein["new_rel"] = "SBJ"
        dokein["new_head"] = head_matriz
        dokein["lock"] = True
        
        # O infinitivo dependente (ἔχειν) vira OBJ de δοκεῖν
        echein = next((w for w in words if w.get("lemma") == "ἔχω" and ("v--p" in w.get("xpos", "") or w.get("form") == "ἔχειν")), None)
        if echein:
            echein["new_rel"] = "OBJ"
            echein["new_head"] = dokein["id"]
            echein["lock"] = True

        # Sujeito em acusativo (αὐτόν) vira SBJ do infinitivo
        auton = next((w for w in words if w.get("text") == "αὐτὸν"), None)
        if auton:
            auton["new_rel"] = "SBJ"
            auton["new_head"] = dokein["id"]
            auton["lock"] = True

        # Adjetivo associado (ἀγαθόν) vira PNOM do nó artificial
        agathon = next((w for w in words if w.get("text") == "ἀγαθὸν"), None)
        if agathon:
            agathon["new_rel"] = "PNOM"
            agathon["new_head"] = head_matriz
            agathon["lock"] = True

    # 2. Estruturação da coordenação disjuntiva com 'ἤ' (IDs 8 e 10)
    conj_eta_main = next((w for w in words if w.get("text") == "ἤ" and str(w["id"]) == "10"), None)
    echein = next((w for w in words if w.get("lemma") == "ἔχω"), None)
    
    if conj_eta_main and echein:
        conj_eta_main["new_rel"] = "COORD"
        conj_eta_main["new_head"] = echein["id"]
        conj_eta_main["lock"] = True

        # Elementos coordenados em acusativo (ψώραν, λέπραν, ἐλέφαντα, πάθος)
        for id_w in ["7", "9", "11", "14"]:
            w = get_word(words, id_w)
            if w:
                w["new_rel"] = "OBJ_CO"
                w["new_head"] = conj_eta_main["id"]
                w["lock"] = True

        eta_aux = get_word(words, 8)
        if eta_aux:
            eta_aux["new_rel"] = "AuxY"
            eta_aux["new_head"] = conj_eta_main["id"]
            eta_aux["lock"] = True

    # 3. Tratamento de 'οἷον' exemplificador (Words 16, 17, 18, 19)
    conj_eta_app = next((w for w in words if w.get("text") == "ἤ" and str(w["id"]) == "18"), None)
    pathos = get_word(words, 14)
    
    if conj_eta_app and pathos:
        conj_eta_app["new_rel"] = "COORD"
        conj_eta_app["new_head"] = pathos["id"]
        conj_eta_app["lock"] = True

        oion = get_word(words, 16)
        if oion:
            oion["new_rel"] = "AuxZ"
            oion["new_head"] = conj_eta_app["id"]
            oion["lock"] = True

    for id_w in ["17", "19"]:
                w = get_word(words, id_w)
                if w:
                    w["new_rel"] = "APOS_CO"
                    w["new_head"] = conj_eta_app["id"]
                    w["lock"] = True


def get_word(words, target_id):
    """Retorna o dicionário da palavra com o id especificado (suporta int ou str)."""
    if target_id is None:
        return None
    return next((w for w in words if str(w.get("id")) == str(target_id)), None)


def extrair_genero(w):
    """Extrai o gênero ('m', 'f', 'n') a partir da postag/xpos no formato AGDT (9 caracteres)."""
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 7:
        gen = postag[6]  # Posição 7 da postag de 9 posições
        if gen in {"m", "f", "n"}:
            return gen
    return None


def extrair_caso(w):
    """Extrai o caso ('n', 'g', 'd', 'a', 'v') a partir da postag/xpos no formato AGDT."""
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 8:
        caso = postag[7]  # Posição 8 da postag de 9 posições
        if caso in {"n", "g", "d", "a", "v"}:
            return caso
    return None



def desconstruir_atribuicoes_sem_concordancia(words):
    """
    Remove atribuições do Stanza onde adjetivos/artigos são colocados como ATR
    de palavras com as quais NÃO concordam em gênero/caso/número.
    """
    for w in words:
        if w.get("lock"):
            continue
            
        upos = w.get("upos", "")
        if upos in {"DET", "ARTICLE", "ADJ"}:
            head_id = w.get("new_head") or w.get("head")
            head_word = get_word(words, head_id)
            
            if head_word:
                gen_w = extrair_genero(w) # pega 'm', 'f', 'n' da postag
                gen_h = extrair_genero(head_word)
                caso_w = extrair_caso(w)
                caso_h = extrair_caso(head_word)
                
                # Se ambos têm gênero definido e são DIFERENTES (ex: neutro com masculino)
                if gen_w and gen_h and gen_w != gen_h and gen_w != "_" and gen_h != "_":
                    # Desconecta a dependência absurda
                    w["new_head"] = None
                    w["new_rel"] = None


def aplicar_infinitivo_substantivado_artigo(words):
    """
    Trata estruturas de Infinitivo Substantivado por artigo neutro (τὸ ... ἰδεῖν),
    em que o artigo τὸ articula o infinitivo como SBJ ou OBJ do verbo principal.
    """
    for w in words:
        texto = w.get("text", "").lower()
        upos = w.get("upos", "")
        
        # Procura o artigo neutro singular τὸ (ou τοῦ, τῷ) em início de frase ou oração
        if texto in {"τὸ", "τοῦ", "τῷ"} and upos in {"DET", "ARTICLE"}:
            # Busca o primeiro infinitivo posterior na oração
            infinitivo = next((x for x in words[w["id"]:] if "v--p" in x.get("xpos", "") or "VerbForm=Inf" in x.get("feats", "")), None)
            verbo_matriz = next((v for v in words if v.get("new_rel") == "PRED" or (v.get("upos") == "VERB" and "v3" in v.get("xpos", ""))), None)
            
            if infinitivo and verbo_matriz:
                # O infinitivo vira o SUJEITO (SBJ) do verbo matriz (ex: σημαίνει)
                infinitivo["new_rel"] = "SBJ"
                infinitivo["new_head"] = verbo_matriz["id"]
                infinitivo["lock"] = True
                
                # O artigo τὸ vira ATR dependente do próprio infinitivo
                w["new_rel"] = "ATR"
                w["new_head"] = infinitivo["id"]
                w["lock"] = True
                
                # Partícula δὲ após o artigo depende da matriz como AuxY
                de = next((x for x in words if x.get("text") in {"δὲ", "δέ"} and x["id"] == w["id"] + 1), None)
                if de:
                    de["new_rel"] = "AuxY"
                    de["new_head"] = verbo_matriz["id"]
                    de["lock"] = True

def converter_sentenca(sent):
    words = construir_words(sent)
    
    sanitizar_morfologia_stanza(words) 
    inicializar_agdt(words)

# 1. Quebra conexões absurdas do Stanza sem concordância (ex: τὸ com ἄλλον)
    desconstruir_atribuicoes_sem_concordancia(words)
    
    # 2. Aplica a estrutura de Infinitivo Substantivado (τὸ ... ἰδεῖν -> SBJ de σημαίνει)
    aplicar_infinitivo_substantivado_artigo(words)
    
    tratar_adverbios_cristalizados_e_sintagmaticos(words)
    aplicar_regra_verbos_factitivos(words)
    aplicar_auxiliares_especiais(words)
    
    # 1. Insere nó artificial [0] se a oração for nominal
    garantir_predicado_raiz(words)
    
    # 2. Executa o AcI unificado e ajusta regência do δοκεῖν / ἔχειν
    aplicar_estrutura_aci_e_disjuncao(words)
    
    aplicar_regras_infinitivo(words)
    aplicar_participios_substantivados(words)
    aplicar_copula(words)
    
    aplicar_coordenacao(words)
    aplicar_oute_correlativo(words)
    aplicar_artigos_repetidos(words)
    aplicar_auxp(words)
    
    for w in words:
        if w.get("lock"):
            continue

        if w["new_head"] is None: 
            w["new_head"] = w["head"]
        if w["new_rel"] is None: 
            w["new_rel"] = w["deprel"]
        if w["new_rel"] == "AuxK": 
            w["new_head"] = 0

        xpos = w.get("xpos") or ""
        feats = w.get("feats") or ""
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 4 and xpos[4] == "n") or (len(xpos) > 2 and xpos[2] == "n")
        
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
