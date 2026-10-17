"""
===============================================================================
 SCIC - Sistema de Comunicação Interplanetária da Colônia  |  Aurora Siger
===============================================================================
Arquivo PRINCIPAL do projeto (Fase 6).

O que o sistema faz:
  1. Carrega (ou cadastra) dados simulados dos módulos da colônia (CSV + Pandas)
  2. Consulta registros operacionais e de comunicação
  3. Calcula indicadores de comunicação (latência, jitter, perda, disponibilidade,
     atraso de luz Terra-Marte)
  4. Calcula erro absoluto e erro relativo (previsto x observado) e discute
     precisão numérica / ponto flutuante
  5. Simula a recuperação da latência após a tempestade com método de Euler e
     usa o método de Newton para estimar quando o enlace volta ao normal
  6. Treina modelos simples de regressão (scikit-learn), avalia com MAE, MSE,
     RMSE, R², compara com AIC/BIC e usa Grid Search
  7. Prioriza alertas com um HEAP MÁXIMO implementado do zero
  8. Faz busca por prefixo/autocomplete com uma TRIE implementada do zero
  9. Relaciona o sistema com COA: dispositivos de E/S, bases numéricas,
     IEEE 754, tensão, corrente, potência e Lei de Ohm
 10. Painel de gerenciamento inteligente (tendência, redundância, manutenção
     preditiva)
 11. Fluxo "Analisar alertas de comunicação" (exemplo do enunciado)
 12. Análise final + geração dos gráficos

Como executar:
  python codigo_fonte.py          -> menu interativo no terminal
  python codigo_fonte.py --demo   -> executa todas as funcionalidades em sequência

Bibliotecas: numpy, pandas, matplotlib, scikit-learn (todas vistas na fase).
===============================================================================
"""

import math
import os
import struct
import sys
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # gera imagens sem precisar de janela gráfica
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# -----------------------------------------------------------------------------
# Configurações gerais
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQ_DADOS = os.path.join(BASE_DIR, "dados_aurora_siger.csv")
DIR_GRAFICOS = os.path.join(BASE_DIR, "graficos_ou_imagens")

VEL_LUZ_KM_S = 299_792.458          # velocidade da luz no vácuo (km/s)
LIMITE_ERRO_ACEITAVEL = 0.10        # erro relativo até 10%  -> aceitável
LIMITE_ERRO_PREOCUPANTE = 0.25      # erro relativo acima de 25% -> crítico
LIMITE_PERDA_PACOTES = 4.0          # % de perda que dispara alerta

COLUNAS_OBRIGATORIAS = [
    "id_registro", "ciclo", "data_registro", "modulo_id", "modulo", "tipo",
    "essencial", "codigo_sensor", "carga_rede_pct", "intensidade_sinal_dbm",
    "tempestade_poeira", "latencia_prevista_ms", "latencia_observada_ms",
    "perda_pacotes_pct", "tensao_v", "corrente_a", "potencia_w",
    "distancia_terra_mkm", "status", "prioridade", "mensagem_alerta",
]

# Comandos cadastrados na colônia (usados na busca por prefixo da trie)
COMANDOS = [
    "comando_reiniciar_enlace", "comando_trocar_rota", "comunicar_terra",
    "controle_termico", "controle_de_acesso", "consultar_status",
    "calibrar_antena", "diagnosticar_sensor", "desligar_carga_nao_essencial",
    "ativar_enlace_redundante", "agendar_manutencao",
]

# Estado compartilhado entre as opções do menu
ESTADO = {"df": None, "modelo": None, "trie": None, "resultados": {}}


# -----------------------------------------------------------------------------
# Utilitários de interface
# -----------------------------------------------------------------------------
def titulo(texto):
    """Imprime um cabeçalho de seção."""
    print("\n" + "=" * 78)
    print(f" {texto}")
    print("=" * 78)


def subtitulo(texto):
    print(f"\n--- {texto} ---")


def ler(mensagem, padrao=""):
    """input() seguro: se não houver terminal (EOF), devolve o valor padrão."""
    try:
        valor = input(mensagem).strip()
        return valor if valor else padrao
    except EOFError:
        return padrao


def garantir_dados():
    """Carrega a base automaticamente se o usuário ainda não carregou."""
    if ESTADO["df"] is None:
        carregar_dados(verbose=False)
    return ESTADO["df"]


def normalizar(texto):
    """Minúsculas e sem acentos: 'Comunicação' -> 'comunicacao'."""
    sem_acento = unicodedata.normalize("NFKD", str(texto))
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower().strip()


# =============================================================================
# 1. ORGANIZAÇÃO DOS DADOS (CSV + Pandas)
# =============================================================================
def carregar_dados(caminho=ARQ_DADOS, verbose=True):
    """Lê o CSV, valida colunas, ajusta tipos e cria colunas derivadas."""
    if not os.path.exists(caminho):
        print(f"[ERRO] Arquivo não encontrado: {caminho}")
        print("       Execute 'python gerar_dados.py' para recriar a base.")
        sys.exit(1)

    df = pd.read_csv(caminho, encoding="utf-8")

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")

    # Limpeza básica: remove duplicados e linhas sem latência
    antes = len(df)
    df = df.drop_duplicates(subset="id_registro")
    df = df.dropna(subset=["latencia_observada_ms", "latencia_prevista_ms"])
    df["data_registro"] = pd.to_datetime(df["data_registro"])
    df["essencial"] = df["essencial"].astype(int)
    df["tempestade_poeira"] = df["tempestade_poeira"].astype(int)

    # Colunas derivadas usadas em várias análises
    df["erro_abs_ms"] = (df["latencia_observada_ms"] - df["latencia_prevista_ms"]).abs()
    df["erro_rel"] = df["erro_abs_ms"] / df["latencia_observada_ms"].abs()

    ESTADO["df"] = df
    ESTADO["trie"] = None  # força reconstrução da trie com os dados novos

    if verbose:
        titulo("1. DADOS CARREGADOS")
        print(f"Arquivo: {os.path.basename(caminho)}")
        print(f"Registros lidos: {antes} | após limpeza: {len(df)}")
        print(f"Módulos: {df['modulo_id'].nunique()} | Ciclos: "
              f"{df['ciclo'].min()} a {df['ciclo'].max()} "
              f"({df['data_registro'].min():%d/%m/%Y} a {df['data_registro'].max():%d/%m/%Y})")
        print("\nStatus dos registros:")
        print(df["status"].value_counts().to_string())
        print("\nAmostra (5 primeiros registros):")
        print(df[["id_registro", "ciclo", "modulo", "codigo_sensor",
                  "latencia_prevista_ms", "latencia_observada_ms",
                  "status"]].head().to_string(index=False))

        # Também mostramos a organização em estruturas Python puras
        modulos = (df[["modulo_id", "modulo", "tipo", "essencial", "codigo_sensor"]]
                   .drop_duplicates("modulo_id"))
        catalogo = {r.modulo_id: {"nome": r.modulo, "tipo": r.tipo,
                                  "essencial": bool(r.essencial),
                                  "sensor": r.codigo_sensor}
                    for r in modulos.itertuples()}
        print(f"\nCatálogo de módulos (dicionário Python) - {len(catalogo)} itens, ex.:")
        print(f"  'COM-01' -> {catalogo.get('COM-01')}")
    return df


def cadastrar_registro():
    """Cadastro manual de um novo registro (salvo no CSV)."""
    df = garantir_dados()
    titulo("CADASTRAR NOVO REGISTRO")
    print("Módulos existentes:", ", ".join(sorted(df["modulo_id"].unique())))
    mid = ler("ID do módulo (ex.: COM-01): ", "COM-01").upper()
    base = df[df["modulo_id"] == mid]
    if base.empty:
        print("[ERRO] Módulo inexistente. Cadastro cancelado.")
        return
    ref = base.iloc[-1]
    try:
        lat_obs = float(ler("Latência observada (ms): ", str(ref["latencia_observada_ms"])))
        carga = float(ler("Carga da rede (%): ", str(ref["carga_rede_pct"])))
        sinal = float(ler("Intensidade do sinal (dBm): ", str(ref["intensidade_sinal_dbm"])))
        tensao = float(ler("Tensão (V): ", str(ref["tensao_v"])))
        corrente = float(ler("Corrente (A): ", str(ref["corrente_a"])))
    except ValueError:
        print("[ERRO] Valor numérico inválido. Cadastro cancelado.")
        return

    lat_prev = float(ref["latencia_prevista_ms"])
    status = "alerta" if lat_obs > 1.25 * lat_prev else "ativo"
    novo = {
        "id_registro": f"R{len(df) + 1:04d}",
        "ciclo": int(df["ciclo"].max()) + 1,
        "data_registro": (df["data_registro"].max() + pd.Timedelta(days=1)).date().isoformat(),
        "modulo_id": mid, "modulo": ref["modulo"], "tipo": ref["tipo"],
        "essencial": int(ref["essencial"]), "codigo_sensor": ref["codigo_sensor"],
        "carga_rede_pct": carga, "intensidade_sinal_dbm": sinal, "tempestade_poeira": 0,
        "latencia_prevista_ms": lat_prev, "latencia_observada_ms": lat_obs,
        "perda_pacotes_pct": 0.0, "tensao_v": tensao, "corrente_a": corrente,
        "potencia_w": round(tensao * corrente, 2),
        "distancia_terra_mkm": float(df["distancia_terra_mkm"].iloc[-1]),
        "status": status, "prioridade": 4 if status == "alerta" else 1,
        "mensagem_alerta": "Latência acima do previsto" if status == "alerta" else "Operação normal",
    }
    bruto = pd.read_csv(ARQ_DADOS, encoding="utf-8")
    bruto = pd.concat([bruto, pd.DataFrame([novo])], ignore_index=True)
    bruto.to_csv(ARQ_DADOS, index=False, encoding="utf-8")
    print(f"Registro {novo['id_registro']} salvo com status '{status}'. "
          f"Potência calculada: {novo['potencia_w']} W")
    carregar_dados(verbose=False)


# =============================================================================
# 2. CONSULTA DE REGISTROS
# =============================================================================
def consultar_registros(modulo_id=None, status=None, ciclo=None):
    df = garantir_dados()
    titulo("2. CONSULTA DE REGISTROS")
    if modulo_id is None and status is None and ciclo is None:
        modulo_id = ler("Filtrar por módulo (ex.: COM-02, Enter = todos): ").upper() or None
        status = ler("Filtrar por status (ativo/alerta/manutencao, Enter = todos): ").lower() or None
        c = ler("Filtrar por ciclo (1-15, Enter = todos): ")
        ciclo = int(c) if c.isdigit() else None

    filtro = df
    if modulo_id:
        filtro = filtro[filtro["modulo_id"] == modulo_id]
    if status:
        filtro = filtro[filtro["status"] == status]
    if ciclo:
        filtro = filtro[filtro["ciclo"] == ciclo]

    print(f"Filtros -> módulo: {modulo_id or 'todos'} | status: {status or 'todos'} | "
          f"ciclo: {ciclo or 'todos'}  =>  {len(filtro)} registro(s)")
    if not filtro.empty:
        print(filtro[["id_registro", "ciclo", "modulo_id", "latencia_prevista_ms",
                      "latencia_observada_ms", "perda_pacotes_pct", "status",
                      "mensagem_alerta"]].head(20).to_string(index=False))
        if len(filtro) > 20:
            print(f"... ({len(filtro) - 20} registros omitidos)")
    return filtro


# =============================================================================
# 3. INDICADORES DE COMUNICAÇÃO
# =============================================================================
def calcular_indicadores(verbose=True):
    df = garantir_dados()
    lat = df["latencia_observada_ms"].to_numpy()

    ind = {
        "latencia_media_ms": float(np.mean(lat)),
        "latencia_mediana_ms": float(np.median(lat)),
        "latencia_p95_ms": float(np.percentile(lat, 95)),
        "jitter_ms": float(np.std(lat, ddof=1)),
        "perda_media_pct": float(df["perda_pacotes_pct"].mean()),
        "disponibilidade_pct": float((df["status"] == "ativo").mean() * 100),
        "taxa_alerta_pct": float((df["status"] == "alerta").mean() * 100),
    }

    # Atraso da luz Terra-Marte (só ida) = distância / c
    dist = df.groupby("ciclo")["distancia_terra_mkm"].first()
    atraso_min = dist * 1e6 / VEL_LUZ_KM_S / 60
    ind["atraso_luz_min_inicio"] = float(atraso_min.iloc[0])
    ind["atraso_luz_min_fim"] = float(atraso_min.iloc[-1])

    # Comparação tempestade x tempo normal
    por_evento = df.groupby("tempestade_poeira").agg(
        latencia_media_ms=("latencia_observada_ms", "mean"),
        perda_media_pct=("perda_pacotes_pct", "mean"),
        sinal_medio_dbm=("intensidade_sinal_dbm", "mean"))
    por_evento.index = ["sem tempestade", "com tempestade"]

    por_modulo = df.groupby("modulo_id").agg(
        latencia_media=("latencia_observada_ms", "mean"),
        jitter=("latencia_observada_ms", "std"),
        perda_media=("perda_pacotes_pct", "mean"),
        disponibilidade=("status", lambda s: (s == "ativo").mean() * 100),
    ).round(2).sort_values("latencia_media", ascending=False)

    ESTADO["resultados"]["indicadores"] = ind
    ESTADO["resultados"]["indicadores_evento"] = por_evento
    if verbose:
        titulo("3. INDICADORES DE COMUNICAÇÃO")
        print(f"Latência média da rede interna ..: {ind['latencia_media_ms']:.1f} ms")
        print(f"Latência mediana .................: {ind['latencia_mediana_ms']:.1f} ms")
        print(f"Latência P95 (pior 5%) ...........: {ind['latencia_p95_ms']:.1f} ms")
        print(f"Jitter (desvio-padrão) ...........: {ind['jitter_ms']:.1f} ms")
        print(f"Perda média de pacotes ...........: {ind['perda_media_pct']:.2f} %")
        print(f"Disponibilidade (status ativo) ...: {ind['disponibilidade_pct']:.1f} %")
        print(f"Registros em alerta ..............: {ind['taxa_alerta_pct']:.1f} %")
        print(f"\nEnlace Terra-Marte (atraso da luz, só ida):")
        print(f"  ciclo 1 : {dist.iloc[0]:.2f} milhões de km -> {ind['atraso_luz_min_inicio']:.2f} min")
        print(f"  ciclo {dist.index[-1]}: {dist.iloc[-1]:.2f} milhões de km -> {ind['atraso_luz_min_fim']:.2f} min")
        print(f"  Ida e volta de um comando: ~{2 * ind['atraso_luz_min_fim']:.1f} min "
              "-> por isso a colônia precisa decidir localmente.")
        subtitulo("Impacto da tempestade de poeira")
        print(por_evento.round(2).to_string())
        subtitulo("Indicadores por módulo")
        print(por_modulo.to_string())
    return ind


# =============================================================================
# 4. ANÁLISE NUMÉRICA E AVALIAÇÃO DE ERROS
# =============================================================================
def classificar_erro(erro_rel):
    if erro_rel <= LIMITE_ERRO_ACEITAVEL:
        return "aceitável"
    if erro_rel <= LIMITE_ERRO_PREOCUPANTE:
        return "atenção"
    return "crítico"


def analisar_erros(verbose=True):
    df = garantir_dados()
    df["classe_erro"] = df["erro_rel"].apply(classificar_erro)

    por_modulo = df.groupby("modulo_id").agg(
        lat_media=("latencia_observada_ms", "mean"),
        erro_abs_medio=("erro_abs_ms", "mean"),
        erro_rel_medio=("erro_rel", "mean"),
        erro_rel_max=("erro_rel", "max"),
    )
    por_modulo["classe"] = por_modulo["erro_rel_medio"].apply(classificar_erro)
    por_modulo = por_modulo.sort_values("erro_rel_medio", ascending=False)
    ESTADO["resultados"]["erros_por_modulo"] = por_modulo
    ESTADO["resultados"]["classes_erro"] = df["classe_erro"].value_counts()

    if verbose:
        titulo("4. ANÁLISE NUMÉRICA E AVALIAÇÃO DE ERROS")
        print("Erro absoluto  = |latência observada - latência prevista|  (em ms)")
        print("Erro relativo  = erro absoluto / |latência observada|       (sem unidade)")
        print(f"Critério da colônia: <= {LIMITE_ERRO_ACEITAVEL:.0%} aceitável | "
              f"<= {LIMITE_ERRO_PREOCUPANTE:.0%} atenção | acima disso crítico\n")

        ex = df.sort_values("erro_abs_ms", ascending=False).iloc[0]
        print("Exemplo do maior erro absoluto:")
        print(f"  {ex['id_registro']} {ex['modulo']} (ciclo {ex['ciclo']}): previsto "
              f"{ex['latencia_prevista_ms']} ms, observado {ex['latencia_observada_ms']} ms")
        print(f"  erro absoluto = |{ex['latencia_observada_ms']} - {ex['latencia_prevista_ms']}| "
              f"= {ex['erro_abs_ms']:.1f} ms")
        print(f"  erro relativo = {ex['erro_abs_ms']:.1f} / {ex['latencia_observada_ms']} "
              f"= {ex['erro_rel']:.4f} ({ex['erro_rel']:.1%}) -> {classificar_erro(ex['erro_rel'])}")

        subtitulo("Por que o erro RELATIVO é necessário")
        a = df[df["modulo_id"] == "COM-01"]["erro_abs_ms"].mean()
        b = df[df["modulo_id"] == "COM-02"]["erro_abs_ms"].mean()
        ra = df[df["modulo_id"] == "COM-01"]["erro_rel"].mean()
        rb = df[df["modulo_id"] == "COM-02"]["erro_rel"].mean()
        print(f"  COM-01 (latência ~75 ms):  erro abs médio {a:.1f} ms -> relativo {ra:.1%}")
        print(f"  COM-02 (latência ~215 ms): erro abs médio {b:.1f} ms -> relativo {rb:.1%}")
        pior_abs = "COM-02" if b > a else "COM-01"
        pior_rel = "COM-02" if rb > ra else "COM-01"
        print(f"  Pelo erro absoluto o pior é o {pior_abs}; pelo relativo, o {pior_rel}.")
        print("  O relé orbital tem latência ~3x maior, então errar ~20 ms nele pesa menos")
        print("  do que errar ~11 ms na Comunicação Central. O erro relativo coloca os")
        print("  módulos na mesma escala e é ele que usamos para classificar o risco.")

        subtitulo("Erros por módulo")
        tab = por_modulo.copy()
        tab["erro_rel_medio"] = (tab["erro_rel_medio"] * 100).round(1).astype(str) + " %"
        tab["erro_rel_max"] = (tab["erro_rel_max"] * 100).round(1).astype(str) + " %"
        print(tab.round(2).to_string())
        print("\nClassificação dos 150 registros:")
        print(df["classe_erro"].value_counts().to_string())

        demonstrar_ponto_flutuante()
    return por_modulo


def demonstrar_ponto_flutuante():
    """Mostra efeitos de ponto flutuante e arredondamento nos dados da colônia."""
    df = garantir_dados()
    subtitulo("Precisão numérica e ponto flutuante")

    # (a) 0.1 + 0.2: números decimais não têm representação binária exata
    soma = 0.1 + 0.2
    print(f"(a) 0.1 + 0.2 = {soma!r}  | soma == 0.3? {soma == 0.3} | "
          f"math.isclose? {math.isclose(soma, 0.3)}")
    print("    Por isso o SCIC compara latências com tolerância, nunca com '=='.")

    # (b) Acumulação em float32 x float64 (como num sensor embarcado simples)
    lat = df["latencia_observada_ms"].to_numpy()
    serie = np.tile(lat, 2000)  # 300 mil leituras simuladas
    soma32 = np.float32(0.0)
    for v in serie.astype(np.float32)[:300_000]:
        soma32 += v  # soma ingênua, valor a valor, em precisão simples
    soma64 = math.fsum(serie)  # soma com compensação de erro (referência)
    err_abs = abs(float(soma32) - soma64)
    print(f"(b) Soma de {len(serie):,} leituras de latência:")
    print(f"    float32 ingênuo = {float(soma32):,.2f} | referência (fsum) = {soma64:,.2f}")
    print(f"    erro absoluto = {err_abs:,.2f} ms | erro relativo = {err_abs / soma64:.2e}")
    print("    Somas longas em precisão simples acumulam erro de arredondamento.")

    # (c) Arredondamento do CSV (1 casa decimal)
    print("(c) As latências são gravadas com 1 casa decimal: erro máximo de")
    print(f"    arredondamento = 0,05 ms -> no menor valor ({lat.min():.1f} ms) isso é "
          f"{0.05 / lat.min():.3%} de erro relativo, desprezível frente aos erros de previsão.")

    ESTADO["resultados"]["float32_erro_rel"] = err_abs / soma64


# =============================================================================
# 5. MÉTODOS NUMÉRICOS: EULER E NEWTON
# =============================================================================
def simular_recuperacao(verbose=True, gerar_grafico=True):
    """
    Após a tempestade, a latência do módulo COM-01 decai até o valor normal.
    Modelo (lei de relaxação):  dL/dt = -k (L - L_eq)
    Solução exata:              L(t) = L_eq + (L0 - L_eq) e^(-k t)
    Comparamos Euler (vários passos h) com a solução exata e usamos Newton
    para achar o instante em que L(t) atinge o limite operacional.
    """
    df = garantir_dados()
    com = df[df["modulo_id"] == "COM-01"]
    L0 = float(com[com["tempestade_poeira"] == 1]["latencia_observada_ms"].mean())
    L_eq = float(com[com["tempestade_poeira"] == 0]["latencia_observada_ms"].mean())
    k = 0.35          # taxa de recuperação (1/h), parâmetro de engenharia
    T = 12.0          # horas simuladas
    limite = L_eq * 1.10  # "normal" = até 10% acima do equilíbrio

    def exata(t):
        return L_eq + (L0 - L_eq) * np.exp(-k * t)

    def euler(h):
        n = int(round(T / h))
        t = np.linspace(0, T, n + 1)
        L = np.empty(n + 1)
        L[0] = L0
        for i in range(n):
            L[i + 1] = L[i] + h * (-k * (L[i] - L_eq))  # passo de Euler
        return t, L

    resultados = []
    curvas = {}
    for h in (2.0, 1.0, 0.5, 0.1):
        t, L = euler(h)
        erro_max = float(np.max(np.abs(L - exata(t))))
        resultados.append((h, erro_max))
        curvas[h] = (t, L)

    # Newton-Raphson: f(t) = L(t) - limite = 0
    f = lambda t: L_eq + (L0 - L_eq) * math.exp(-k * t) - limite
    df_dt = lambda t: -k * (L0 - L_eq) * math.exp(-k * t)
    t_n, iteracoes = 1.0, []
    for it in range(1, 30):
        t_novo = t_n - f(t_n) / df_dt(t_n)
        iteracoes.append((it, t_novo, abs(t_novo - t_n)))
        if abs(t_novo - t_n) < 1e-10:
            t_n = t_novo
            break
        t_n = t_novo
    t_analitico = math.log((L0 - L_eq) / (limite - L_eq)) / k

    ESTADO["resultados"]["euler"] = {"L0": L0, "L_eq": L_eq, "k": k,
                                     "erros": resultados, "t_newton": t_n,
                                     "t_analitico": t_analitico,
                                     "iteracoes_newton": len(iteracoes),
                                     "limite": limite}
    if verbose:
        titulo("5. SIMULAÇÃO NUMÉRICA: EULER E NEWTON")
        print("Cenário: recuperação da latência do módulo COM-01 depois da tempestade.")
        print(f"  L0 (média na tempestade) = {L0:.2f} ms | L_eq (normal) = {L_eq:.2f} ms | "
              f"k = {k} /h")
        print("  EDO: dL/dt = -k (L - L_eq)    Euler: L[i+1] = L[i] + h * f(L[i])")
        subtitulo("Erro de discretização do método de Euler")
        print("   passo h (h) | erro máximo vs solução exata (ms)")
        for h, e in resultados:
            print(f"   {h:>11} | {e:.4f}")
        print("  Ao dividir o passo por 2, o erro cai ~pela metade: Euler é de 1ª ordem.")
        subtitulo("Método de Newton: quando o enlace volta ao normal?")
        print(f"  Limite operacional: {limite:.2f} ms (L_eq + 10%)")
        for it, tv, d in iteracoes:
            print(f"   iteração {it}: t = {tv:.10f} h  (variação {d:.2e})")
        print(f"  Newton: t = {t_n:.6f} h | analítico: t = {t_analitico:.6f} h | "
              f"diferença = {abs(t_n - t_analitico):.2e} h")
        print(f"  -> O COM-01 leva ~{t_n:.1f} h para voltar ao normal; até lá o SCIC")
        print("     deve manter o enlace redundante ativo.")

    if gerar_grafico:
        os.makedirs(DIR_GRAFICOS, exist_ok=True)
        tt = np.linspace(0, T, 400)
        plt.figure(figsize=(8, 4.5))
        plt.plot(tt, exata(tt), "k-", lw=2, label="Solução exata")
        for h, est in (("2.0", "o--"), ("0.5", "s--")):
            t, L = curvas[float(h)]
            plt.plot(t, L, est, ms=4, label=f"Euler h = {h} h")
        plt.axhline(limite, color="tab:red", ls=":", label=f"Limite {limite:.1f} ms")
        plt.axvline(t_n, color="tab:green", ls=":", label=f"Newton t = {t_n:.2f} h")
        plt.xlabel("Tempo após a tempestade (h)")
        plt.ylabel("Latência COM-01 (ms)")
        plt.title("Recuperação da latência: Euler x solução exata")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_GRAFICOS, "05_euler_newton.png"), dpi=120)
        plt.close()


# =============================================================================
# 6. MODELO SIMPLES DE PREVISÃO E AVALIAÇÃO DE PERFORMANCE
# =============================================================================
def aic_bic(y_true, y_pred, k):
    """AIC e BIC para regressão linear com erro gaussiano (k = nº de parâmetros)."""
    n = len(y_true)
    rss = float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return aic, bic


def metricas(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": math.sqrt(mse), "R2": r2_score(y_true, y_pred)}


def preparar_features(df, colunas_modulo=None):
    """Variáveis explicativas: carga, sinal, tempestade + one-hot do módulo."""
    X = df[["carga_rede_pct", "intensidade_sinal_dbm", "tempestade_poeira"]].copy()
    dummies = pd.get_dummies(df["modulo_id"], prefix="mod", drop_first=True, dtype=float)
    if colunas_modulo is not None:
        dummies = dummies.reindex(columns=colunas_modulo, fill_value=0.0)
    return pd.concat([X, dummies], axis=1)


def treinar_modelo(verbose=True, gerar_grafico=True):
    df = garantir_dados()
    y = df["latencia_observada_ms"]
    X = preparar_features(df)
    colunas_modulo = [c for c in X.columns if c.startswith("mod_")]

    # Divisão treino/teste (80/20), estratificada pela tempestade para que
    # o teste também contenha registros de tempestade
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42,
        stratify=df["tempestade_poeira"])

    # Baseline: previsão do sistema legado que já está no CSV
    base_te = df.loc[idx_te, "latencia_prevista_ms"]
    m_base = metricas(y_te, base_te)

    # Modelo A: regressão linear simples (só carga da rede)
    mA = LinearRegression().fit(X_tr[["carga_rede_pct"]], y_tr)
    m_A = metricas(y_te, mA.predict(X_te[["carga_rede_pct"]]))
    aicA, bicA = aic_bic(y_tr, mA.predict(X_tr[["carga_rede_pct"]]), k=2)

    # Modelo B: regressão linear múltipla (carga + sinal + tempestade + módulo)
    mB = LinearRegression().fit(X_tr, y_tr)
    pred_B = mB.predict(X_te)
    m_B = metricas(y_te, pred_B)
    aicB, bicB = aic_bic(y_tr, mB.predict(X_tr), k=X_tr.shape[1] + 1)

    # Modelo C: Ridge com Grid Search do parâmetro de regularização (validação cruzada)
    grid = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]},
                        scoring="neg_mean_absolute_error", cv=5)
    grid.fit(X_tr, y_tr)
    m_C = metricas(y_te, grid.predict(X_te))

    tabela = pd.DataFrame({
        "Baseline (sistema legado)": m_base,
        "A: Linear simples (carga)": m_A,
        "B: Linear múltipla": m_B,
        f"C: Ridge (alpha={grid.best_params_['alpha']})": m_C,
    }).T
    criterios = pd.DataFrame({"parâmetros": [2, X_tr.shape[1] + 1],
                              "AIC": [aicA, aicB], "BIC": [bicA, bicB]},
                             index=["A: Linear simples", "B: Linear múltipla"])

    coefs = pd.Series(mB.coef_, index=X_tr.columns)
    ESTADO["modelo"] = {"modelo": mB, "colunas_modulo": colunas_modulo}
    ESTADO["resultados"]["metricas"] = tabela
    ESTADO["resultados"]["criterios"] = criterios
    ESTADO["resultados"]["coeficientes"] = coefs
    ESTADO["resultados"]["n_treino_teste"] = (len(X_tr), len(X_te))
    ESTADO["resultados"]["alpha"] = grid.best_params_["alpha"]

    if verbose:
        titulo("6. MODELO SIMPLES DE PREVISÃO DE LATÊNCIA")
        print(f"Alvo: latencia_observada_ms | Treino: {len(X_tr)} registros | "
              f"Teste: {len(X_te)} registros (80/20, random_state=42)")
        print("Variáveis: carga_rede_pct, intensidade_sinal_dbm, tempestade_poeira, "
              "módulo (one-hot)")
        subtitulo("Métricas no conjunto de TESTE")
        print(tabela.round(3).to_string())
        subtitulo("Comparação de modelos no treino (menor é melhor)")
        print(criterios.round(1).to_string())
        subtitulo("Coeficientes do modelo B (interpretação física)")
        print(f"  carga_rede_pct ........: {coefs['carga_rede_pct']:+.3f} ms por ponto de carga")
        print(f"  intensidade_sinal_dbm .: {coefs['intensidade_sinal_dbm']:+.3f} ms por dBm "
              "(sinal mais forte -> menos latência)")
        print(f"  tempestade_poeira .....: {coefs['tempestade_poeira']:+.2f} ms durante tempestade")
        subtitulo("Interpretação")
        b, base = m_B, m_base
        print(f"- O modelo B reduz o MAE de {base['MAE']:.2f} ms (legado) para {b['MAE']:.2f} ms.")
        print(f"- R² = {b['R2']:.3f}: explica {b['R2']:.1%} da variação da latência no teste;")
        print("  isso NÃO quer dizer que ele acerta todos os registros.")
        razao = b["RMSE"] / b["MAE"]
        razao_base = base["RMSE"] / base["MAE"]
        print(f"- RMSE/MAE: modelo B = {razao:.2f} | legado = {razao_base:.2f}. Para erros")
        print("  'bem comportados' essa razão fica perto de 1,25; quanto maior, mais o erro")
        print("  se concentra em poucos registros. No legado a razão é alta porque ele erra")
        print("  MUITO nos ciclos de tempestade (não conhece essa variável). No modelo B a")
        print("  razão é baixa: os erros estão espalhados e pequenos.")
        print("- Limite: picos isolados (reinício de roteador, cabo danificado) não estão nas")
        print("  variáveis de entrada; o modelo não os prevê e eles viram alertas pelo heap.")
        print("- AIC e BIC são menores no modelo B: as variáveis extras compensam a")
        print("  complexidade adicional. O modelo A (só carga) quase não explica a latência.")
        print(f"- Grid Search escolheu alpha={grid.best_params_['alpha']}; o Ridge fica muito")
        print("  próximo do modelo B, sinal de que não há sobreajuste relevante.")

    if gerar_grafico:
        os.makedirs(DIR_GRAFICOS, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        lim = [y_te.min() - 10, y_te.max() + 10]
        ax[0].scatter(y_te, base_te, alpha=0.6, label="Baseline legado", marker="x")
        ax[0].scatter(y_te, pred_B, alpha=0.8, label="Modelo B (linear múltipla)")
        ax[0].plot(lim, lim, "k--", lw=1, label="Previsão perfeita")
        ax[0].set_xlabel("Latência observada (ms)")
        ax[0].set_ylabel("Latência prevista (ms)")
        ax[0].set_title("Observado x previsto (teste)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        residuos = y_te - pred_B
        ax[1].hist(residuos, bins=12, color="tab:orange", edgecolor="k")
        ax[1].axvline(0, color="k", ls="--")
        ax[1].set_xlabel("Resíduo (observado - previsto) [ms]")
        ax[1].set_ylabel("Frequência")
        ax[1].set_title("Distribuição dos resíduos do modelo B")
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_GRAFICOS, "03_modelo_previsao.png"), dpi=120)
        plt.close()
    return tabela


def prever_latencia(df):
    """Usa o modelo B treinado para prever a latência de registros."""
    if ESTADO["modelo"] is None:
        treinar_modelo(verbose=False, gerar_grafico=False)
    info = ESTADO["modelo"]
    X = preparar_features(df, info["colunas_modulo"])
    return info["modelo"].predict(X)


# =============================================================================
# 7. PRIORIZAÇÃO DE ALERTAS COM HEAP MÁXIMO (implementado do zero)
# =============================================================================
@dataclass
class Alerta:
    score: float
    ciclo: int
    id_registro: str
    modulo_id: str
    modulo: str
    mensagem: str
    detalhes: dict = field(default_factory=dict)

    def chave(self):
        # Maior score primeiro; empate -> alerta mais recente (ciclo maior)
        return (self.score, self.ciclo)


class HeapMaxAlertas:
    """
    Heap máximo em vetor (árvore binária quase completa):
      pai(i) = (i-1)//2 | filho_esq(i) = 2i+1 | filho_dir(i) = 2i+2
    O alerta mais urgente fica sempre na raiz (posição 0).
    """

    def __init__(self):
        self.dados = []
        self.comparacoes = 0  # para comparar o custo com uma lista simples

    def __len__(self):
        return len(self.dados)

    def _maior(self, i, j):
        self.comparacoes += 1
        return self.dados[i].chave() > self.dados[j].chave()

    def _trocar(self, i, j):
        self.dados[i], self.dados[j] = self.dados[j], self.dados[i]

    def heapify_up(self, i):
        """Sobe o elemento enquanto ele for maior que o pai.  O(log n)"""
        while i > 0:
            pai = (i - 1) // 2
            if self._maior(i, pai):
                self._trocar(i, pai)
                i = pai
            else:
                break

    def heapify_down(self, i):
        """Desce o elemento trocando com o maior filho.  O(log n)"""
        n = len(self.dados)
        while True:
            maior = i
            esq, dir_ = 2 * i + 1, 2 * i + 2
            if esq < n and self._maior(esq, maior):
                maior = esq
            if dir_ < n and self._maior(dir_, maior):
                maior = dir_
            if maior == i:
                break
            self._trocar(i, maior)
            i = maior

    def inserir(self, alerta):
        self.dados.append(alerta)
        self.heapify_up(len(self.dados) - 1)

    def topo(self):
        """Consulta o mais urgente sem remover.  O(1)"""
        return self.dados[0] if self.dados else None

    def remover_max(self):
        """Remove e devolve o alerta mais urgente.  O(log n)"""
        if not self.dados:
            return None
        self._trocar(0, len(self.dados) - 1)
        maximo = self.dados.pop()
        if self.dados:
            self.heapify_down(0)
        return maximo


def heap_sort(valores):
    """Heap Sort in-place (ordem crescente) usando heapify-down.  O(n log n)"""
    a = list(valores)
    n = len(a)

    def descer(i, tamanho):
        while True:
            maior, e, d = i, 2 * i + 1, 2 * i + 2
            if e < tamanho and a[e] > a[maior]:
                maior = e
            if d < tamanho and a[d] > a[maior]:
                maior = d
            if maior == i:
                return
            a[i], a[maior] = a[maior], a[i]
            i = maior

    for i in range(n // 2 - 1, -1, -1):  # constrói o heap
        descer(i, n)
    for fim in range(n - 1, 0, -1):      # extrai o máximo repetidamente
        a[0], a[fim] = a[fim], a[0]
        descer(0, fim)
    return a


def calcular_score(reg, lat_modelo=None):
    """
    Critério de prioridade (pontuação de 0 a ~120):
      prioridade operacional (1-5) x 10
    + 15 se o módulo é essencial (suporte de vida, comunicação, controle)
    + excesso de latência sobre o previsto (até 30 pontos)
    + perda de pacotes x 2 (até 20 pontos)
    + 5 se o próprio modelo do SCIC também não explica a latência (anomalia)
    - 10 se é manutenção programada (evento esperado)
    """
    score = reg["prioridade"] * 10
    if reg["essencial"]:
        score += 15
    excesso = (reg["latencia_observada_ms"] - reg["latencia_prevista_ms"]) / reg["latencia_prevista_ms"]
    score += min(max(excesso, 0) * 100 * 0.5, 30)
    score += min(reg["perda_pacotes_pct"] * 2, 20)
    if lat_modelo is not None and reg["latencia_observada_ms"] > 1.2 * lat_modelo:
        score += 5
    if reg["status"] == "manutencao":
        score -= 10
    return round(float(score), 2)


def montar_heap_alertas(df_alertas):
    heap = HeapMaxAlertas()
    pred = prever_latencia(df_alertas) if len(df_alertas) else []
    for (_, r), lm in zip(df_alertas.iterrows(), pred):
        heap.inserir(Alerta(
            score=calcular_score(r, lm), ciclo=int(r["ciclo"]),
            id_registro=r["id_registro"], modulo_id=r["modulo_id"],
            modulo=r["modulo"], mensagem=r["mensagem_alerta"],
            detalhes={"obs": r["latencia_observada_ms"], "prev": r["latencia_prevista_ms"],
                      "modelo": round(float(lm), 1), "perda": r["perda_pacotes_pct"],
                      "prioridade": int(r["prioridade"])}))
    return heap


def priorizar_alertas(top=8, verbose=True):
    df = garantir_dados()
    alertas = df[df["status"].isin(["alerta", "manutencao"])]
    heap = montar_heap_alertas(alertas)
    comp_construcao = heap.comparacoes

    if verbose:
        titulo("7. PRIORIZAÇÃO DE ALERTAS COM HEAP MÁXIMO")
        print(f"{len(heap)} alertas inseridos no heap ({comp_construcao} comparações "
              "na construção via heapify-up).")
        r = heap.topo()
        print(f"Raiz do heap (consulta O(1)): [{r.score}] {r.modulo} - {r.mensagem}")
        print("\nVetor interno (primeiros 7 nós = 3 níveis da árvore):")
        for i, a in enumerate(heap.dados[:7]):
            pai = (i - 1) // 2 if i else "-"
            print(f"  posição {i} (pai {pai}): score {a.score:6.2f}  {a.modulo_id} ciclo {a.ciclo}")

    heap.comparacoes = 0
    ordem = []
    for _ in range(min(top, len(heap))):
        ordem.append(heap.remover_max())
    comp_heap = heap.comparacoes

    # Comparação com lista simples: achar o máximo exige percorrer tudo a cada vez
    lista = alertas.apply(lambda r: calcular_score(r), axis=1).tolist()
    comp_lista, restante = 0, list(lista)
    for _ in range(min(top, len(restante))):
        idx_max = 0
        for j in range(1, len(restante)):
            comp_lista += 1
            if restante[j] > restante[idx_max]:
                idx_max = j
        restante.pop(idx_max)

    ESTADO["resultados"]["heap"] = {"n": len(alertas), "top": ordem,
                                    "comp_heap": comp_heap, "comp_lista": comp_lista}
    if verbose:
        subtitulo(f"Top {top} alertas mais urgentes (remover_max + heapify-down)")
        for i, a in enumerate(ordem, 1):
            d = a.detalhes
            print(f"{i:>2}. [{a.score:6.2f}] ciclo {a.ciclo:>2} {a.modulo_id} {a.modulo:<27} "
                  f"obs {d['obs']:.1f} ms | prev {d['prev']:.1f} | perda {d['perda']:.2f}%")
            print(f"     -> {a.mensagem}")
        subtitulo("Heap x lista simples")
        print(f"Extrair os {top} mais urgentes: heap = {comp_heap} comparações | "
              f"lista simples = {comp_lista} comparações")
        print("Heap: inserir/remover O(log n), consultar topo O(1).")
        print("Lista: achar o máximo O(n) a cada consulta; ordenar tudo O(n log n) a cada")
        print("novo alerta. Com alertas chegando o tempo todo, o heap escala melhor.")
        scores = [a.score for a in ordem[:5]]
        print(f"\nHeap Sort (crescente) dos 5 maiores scores: {heap_sort(scores)}")
    return ordem


# =============================================================================
# 8. BUSCA POR PREFIXO COM TRIE (implementada do zero)
# =============================================================================
class NoTrie:
    __slots__ = ("filhos", "fim", "termos", "referencias")

    def __init__(self):
        self.filhos = {}          # caractere -> NoTrie
        self.fim = False          # marca o fim de uma chave cadastrada
        self.termos = set()       # texto original (com acento) da chave
        self.referencias = set()  # ids de módulos/registros/comandos ligados à chave


class Trie:
    def __init__(self):
        self.raiz = NoTrie()
        self.total_chaves = 0
        self.total_nos = 1

    def inserir(self, termo, referencia=None):
        """Insere caractere a caractere.  O(m), m = tamanho da chave."""
        no = self.raiz
        for ch in normalizar(termo):
            if ch not in no.filhos:
                no.filhos[ch] = NoTrie()
                self.total_nos += 1
            no = no.filhos[ch]
        if not no.fim:
            self.total_chaves += 1
        no.fim = True
        no.termos.add(termo)
        if referencia:
            no.referencias.add(referencia)

    def _no_do_prefixo(self, prefixo):
        no = self.raiz
        for ch in normalizar(prefixo):
            if ch not in no.filhos:
                return None
            no = no.filhos[ch]
        return no

    def buscar(self, termo):
        """Busca exata.  O(m)"""
        no = self._no_do_prefixo(termo)
        return bool(no and no.fim)

    def autocompletar(self, prefixo, limite=25):
        """Todas as chaves que começam com o prefixo (busca em profundidade)."""
        no = self._no_do_prefixo(prefixo)
        if no is None:
            return []
        saida, pilha = [], [no]
        while pilha and len(saida) < limite:
            atual = pilha.pop()
            if atual.fim:
                for t in sorted(atual.termos):
                    saida.append((t, sorted(atual.referencias)))
            for ch in sorted(atual.filhos, reverse=True):
                pilha.append(atual.filhos[ch])
        return saida


def construir_trie():
    df = garantir_dados()
    trie = Trie()
    modulos = df.drop_duplicates("modulo_id")
    for _, r in modulos.iterrows():
        trie.inserir(r["modulo"], r["modulo_id"])          # nome completo
        trie.inserir(r["modulo_id"], r["modulo_id"])       # ID (COM-01)
        trie.inserir(r["codigo_sensor"], r["modulo_id"])   # código hex (0x0301)
        for palavra in r["modulo"].split():                # cada palavra do nome
            if len(palavra) >= 4:
                trie.inserir(palavra, r["modulo_id"])
    for _, r in df[df["status"] != "ativo"].iterrows():   # mensagens de alerta
        trie.inserir(r["mensagem_alerta"], r["id_registro"])
    for cmd in COMANDOS:
        trie.inserir(cmd, "comando")
    ESTADO["trie"] = trie
    return trie


def buscar_prefixo(prefixos=None, verbose=True):
    trie = ESTADO["trie"] or construir_trie()
    interativo = prefixos is None
    if verbose:
        titulo("8. BUSCA POR PREFIXO COM TRIE")
        print(f"Trie com {trie.total_chaves} chaves e {trie.total_nos} nós "
              "(nomes de módulos, IDs, códigos de sensores, alertas e comandos).")
        print("Busca sem diferenciar maiúsculas/acentos: 'comunic' encontra 'Comunicação'.")

    if interativo:
        prefixos = []
        while True:
            p = ler("\nDigite um prefixo (Enter para sair): ")
            if not p:
                break
            prefixos.append(p)
            mostrar_resultado_trie(trie, p)
        return
    for p in prefixos:
        mostrar_resultado_trie(trie, p)


def mostrar_resultado_trie(trie, prefixo):
    res = trie.autocompletar(prefixo)
    print(f"\n> prefixo '{prefixo}': {len(res)} resultado(s)")
    for termo, refs in res:
        refs_txt = ", ".join(refs[:6]) + (" ..." if len(refs) > 6 else "")
        print(f"   - {termo:<48} [{refs_txt}]")
    if not res:
        print("   (nenhuma chave começa com esse prefixo)")


# =============================================================================
# 9. COA: DISPOSITIVOS, BASES NUMÉRICAS E ELETRICIDADE BÁSICA
# =============================================================================
DIGITOS = "0123456789ABCDEF"


def decimal_para_base(n, base):
    """Conversão manual por divisões sucessivas."""
    if n == 0:
        return "0"
    restos = []
    while n > 0:
        restos.append(DIGITOS[n % base])
        n //= base
    return "".join(reversed(restos))


def base_para_decimal(texto, base):
    """Conversão manual pela soma de dígito x base^posição."""
    total = 0
    for ch in texto.upper():
        total = total * base + DIGITOS.index(ch)
    return total


def ieee754_bits(valor):
    """Bits de um float32: sinal | expoente (8) | mantissa (23)."""
    bits = struct.unpack(">I", struct.pack(">f", valor))[0]
    b = f"{bits:032b}"
    return b[0], b[1:9], b[9:]


def coa_eletricidade(verbose=True):
    df = garantir_dados()
    if verbose:
        titulo("9. COA: DISPOSITIVOS, BASES NUMÉRICAS E ELETRICIDADE")
        subtitulo("Dispositivos de entrada e saída do SCIC (conceitual)")
        print("ENTRADA : sensores de latência (ping/eco), medidores de intensidade de sinal")
        print("          (dBm), sensores de tensão/corrente dos transmissores, teclado do")
        print("          operador (menu) e o arquivo CSV (simula o barramento de telemetria).")
        print("SAÍDA   : terminal (menu e relatórios), gráficos PNG, arquivo CSV atualizado;")
        print("          numa versão real: painel na sala de controle e alarme sonoro.")
        print("INTERFACES (conceitual): rede cabeada/fibra entre módulos, Wi-Fi nas")
        print("          habitações, Bluetooth para sensores portáteis, USB para manutenção,")
        print("          rádio UHF até o relé em Phobos e banda X/Ka até a Terra.")

        subtitulo("Códigos de sensores: hexadecimal <-> decimal <-> binário")
        print("Formato: 0xTTNN -> byte alto TT = tipo do módulo, byte baixo NN = número")
    codigos = df.drop_duplicates("modulo_id")[["modulo_id", "codigo_sensor"]]
    tabela = []
    for _, r in codigos.iterrows():
        hexa = r["codigo_sensor"][2:]
        dec = base_para_decimal(hexa, 16)
        binario = decimal_para_base(dec, 2).zfill(16)
        # confere a conversão manual com as funções nativas do Python
        assert dec == int(hexa, 16) and binario == format(dec, "016b")
        tipo, numero = dec >> 8, dec & 0xFF   # deslocamento e máscara de bits
        tabela.append((r["modulo_id"], r["codigo_sensor"], dec,
                       f"{binario[:8]} {binario[8:]}", tipo, numero))
    if verbose:
        print(f"{'módulo':<8}{'hex':<9}{'decimal':>8}  {'binário (16 bits)':<19}tipo nº")
        for m, h, d, b, t, n in tabela:
            print(f"{m:<8}{h:<9}{d:>8}  {b:<19}{t:>4} {n:>2}")
        ex = tabela[4]
        print(f"\nPasso a passo ({ex[1]}): {ex[1][2:]} (base 16) = "
              f"{' + '.join(f'{DIGITOS.index(c)}x16^{3 - i}' for i, c in enumerate(ex[1][2:]))}"
              f" = {ex[2]}")
        print(f"  {ex[2]} em binário (divisões sucessivas por 2) = {decimal_para_base(ex[2], 2)}")

        subtitulo("Representação em ponto flutuante (IEEE 754, 32 bits)")
        valor = float(df["latencia_observada_ms"].iloc[3])
        s, e, m = ieee754_bits(valor)
        guardado = struct.unpack(">f", struct.pack(">f", valor))[0]
        print(f"Latência {valor} ms -> sinal {s} | expoente {e} ({int(e, 2) - 127:+d}) | "
              f"mantissa {m}")
        print(f"Valor efetivamente armazenado em float32: {guardado!r} "
              f"(erro {abs(guardado - valor):.2e} ms)")

    # ------------------------- Eletricidade básica -------------------------
    eletr = df.groupby("modulo_id").agg(
        tensao_v=("tensao_v", "mean"), corrente_a=("corrente_a", "mean"),
        potencia_w=("potencia_w", "mean")).round(2)
    eletr["resistencia_ohm"] = (eletr["tensao_v"] / eletr["corrente_a"]).round(2)  # Ohm
    eletr["energia_sol_wh"] = (eletr["potencia_w"] * 24.62).round(1)  # 1 sol ~ 24,62 h
    com = df[df["modulo_id"] == "COM-02"]
    p_normal = com[com["tempestade_poeira"] == 0]["potencia_w"].mean()
    p_temp = com[com["tempestade_poeira"] == 1]["potencia_w"].mean()
    ESTADO["resultados"]["eletrica"] = eletr
    ESTADO["resultados"]["potencia_com02"] = (p_normal, p_temp)

    if verbose:
        subtitulo("Eletricidade básica aplicada aos transmissores")
        r = df[df["modulo_id"] == "COM-01"].iloc[0]
        V, I = r["tensao_v"], r["corrente_a"]
        P = V * I
        R = V / I
        print(f"Transmissor COM-01 (ciclo 1): V = {V} V, I = {I} A")
        print(f"  Potência  P = V x I      = {V} x {I} = {P:.2f} W")
        print(f"  Lei de Ohm R = V / I     = {V} / {I} = {R:.2f} ohm")
        print(f"  Conferência P = I² x R   = {I}² x {R:.2f} = {I ** 2 * R:.2f} W")
        print(f"  Em dBm: 10 x log10(P em mW) = {10 * math.log10(P * 1000):.1f} dBm")
        print(f"  Energia por sol (24,62 h) = {P * 24.62:.0f} Wh = {P * 24.62 / 1000:.2f} kWh")
        print("\nMédias por módulo:")
        print(eletr.to_string())
        print(f"\nRelé orbital COM-02: {p_normal:.1f} W em tempo normal x {p_temp:.1f} W na "
              f"tempestade (+{(p_temp / p_normal - 1):.0%}).")
        print("Aumentar a potência para vencer a poeira custa energia da microrrede solar,")
        print("que também perde geração com a poeira -> decisão que o SCIC deve priorizar.")
    return eletr


# =============================================================================
# 10. GERENCIAMENTO INTELIGENTE DA COMUNICAÇÃO
# =============================================================================
def painel_gerenciamento(verbose=True):
    df = garantir_dados()
    linhas = []
    for mid, g in df.groupby("modulo_id"):
        g = g.sort_values("ciclo")
        normal = g[g["tempestade_poeira"] == 0]
        # Tendência de latência fora da tempestade (ms por ciclo) -> manutenção preditiva
        inclinacao = float(np.polyfit(normal["ciclo"], normal["latencia_observada_ms"], 1)[0])
        perda_temp = g[g["tempestade_poeira"] == 1]["perda_pacotes_pct"].mean()
        p95 = float(np.percentile(g["latencia_observada_ms"], 95))
        alertas = int((g["status"] == "alerta").sum())
        recs = []
        if perda_temp > LIMITE_PERDA_PACOTES:
            recs.append("enlace redundante na tempestade")
        if inclinacao > 0.8:
            recs.append("manutenção preditiva (latência subindo)")
        # Erro da previsão legada fora da tempestade: se já é alto em dia normal,
        # a estimativa do módulo está mal calibrada (não é só efeito da poeira)
        if normal["erro_rel"].mean() > LIMITE_ERRO_ACEITAVEL and g["essencial"].iloc[0]:
            recs.append("recalibrar previsão (erro alto mesmo sem tempestade)")
        if not recs:
            recs.append("monitoramento normal")
        linhas.append({"modulo_id": mid, "essencial": "sim" if g["essencial"].iloc[0] else "não",
                       "lat_p95_ms": round(p95, 1), "tendencia_ms_ciclo": round(inclinacao, 2),
                       "perda_tempestade_pct": round(perda_temp, 2),
                       "erro_rel_normal_pct": round(normal["erro_rel"].mean() * 100, 1),
                       "alertas": alertas,
                       "recomendacao": "; ".join(recs)})
    painel = pd.DataFrame(linhas).sort_values("alertas", ascending=False)
    ESTADO["resultados"]["painel"] = painel

    if verbose:
        titulo("10. PAINEL DE GERENCIAMENTO INTELIGENTE DA COMUNICAÇÃO")
        print(painel.to_string(index=False))
        print("\nLeitura do painel:")
        print("- Monitoramento contínuo: P95 e alertas por módulo mostram quem degrada primeiro.")
        print("- Redundância: módulos com perda > 4% na tempestade devem migrar para o")
        print("  enlace secundário (cabo/fibra ou rota via COM-01) automaticamente.")
        print("- Manutenção preditiva: tendência positiva fora da tempestade indica desgaste")
        print("  (antena, conector, roteador) antes que vire falha.")
        print("- Automação com supervisão humana: o SCIC sugere a ação; o operador confirma.")
    return painel


# =============================================================================
# 11. FLUXO "ANALISAR ALERTAS DE COMUNICAÇÃO" (exemplo do enunciado)
# =============================================================================
def analisar_alertas_comunicacao(prefixo="com"):
    df = garantir_dados()
    titulo("11. ANALISAR ALERTAS DE COMUNICAÇÃO (fluxo integrado)")
    print("Passo 1 - base carregada:", len(df), "registros")

    acima = df[df["latencia_observada_ms"] > df["latencia_prevista_ms"] * 1.10].copy()
    print(f"Passo 2 - {len(acima)} registros com latência > 10% acima do previsto")
    resumo = (acima.groupby("modulo_id")
              .agg(ocorrencias=("id_registro", "count"),
                   erro_abs_medio_ms=("erro_abs_ms", "mean"),
                   erro_rel_medio=("erro_rel", "mean"))
              .sort_values("ocorrencias", ascending=False))
    resumo["erro_rel_medio"] = (resumo["erro_rel_medio"] * 100).round(1).astype(str) + " %"
    print("Passo 3 - erro entre latência estimada e real nesses registros:")
    print(resumo.round(2).to_string())

    heap = montar_heap_alertas(acima)
    print(f"Passo 4 - heap com {len(heap)} alertas; os 5 mais críticos:")
    for i in range(min(5, len(heap))):
        a = heap.remover_max()
        print(f"   {i + 1}. [{a.score:6.2f}] ciclo {a.ciclo:>2} {a.modulo:<27} {a.mensagem}")

    print(f"Passo 5 - busca por prefixo '{prefixo}' na trie:")
    trie = ESTADO["trie"] or construir_trie()
    mostrar_resultado_trie(trie, prefixo)


# =============================================================================
# 12. ANÁLISE FINAL E GRÁFICOS
# =============================================================================
def gerar_graficos():
    df = garantir_dados()
    os.makedirs(DIR_GRAFICOS, exist_ok=True)

    # Gráfico 1: latência média por ciclo (prevista x observada)
    por_ciclo = df.groupby("ciclo")[["latencia_prevista_ms", "latencia_observada_ms"]].mean()
    plt.figure(figsize=(9, 4.5))
    plt.axvspan(8.5, 11.5, color="tab:orange", alpha=0.15, label="Tempestade de poeira")
    plt.plot(por_ciclo.index, por_ciclo["latencia_prevista_ms"], "o--", label="Prevista (legado)")
    plt.plot(por_ciclo.index, por_ciclo["latencia_observada_ms"], "o-", label="Observada")
    plt.xlabel("Ciclo (sol)")
    plt.ylabel("Latência média da rede (ms)")
    plt.title("Latência média por ciclo - colônia Aurora Siger")
    plt.xticks(por_ciclo.index)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "01_latencia_por_ciclo.png"), dpi=120)
    plt.close()

    # Gráfico 2: erro relativo médio por módulo com faixas de aceitação
    erros = df.groupby("modulo_id")["erro_rel"].mean().sort_values() * 100
    cores = ["tab:green" if v <= 10 else "tab:orange" if v <= 25 else "tab:red" for v in erros]
    plt.figure(figsize=(9, 4.5))
    plt.barh(erros.index, erros.values, color=cores)
    plt.axvline(10, color="k", ls="--", lw=1, label="10% (limite aceitável)")
    plt.xlabel("Erro relativo médio (%)")
    plt.title("Erro relativo da previsão legada por módulo")
    plt.legend()
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "02_erro_relativo_modulos.png"), dpi=120)
    plt.close()

    # Gráfico 4: potência média por módulo
    pot = df.groupby("modulo_id")["potencia_w"].mean().sort_values()
    plt.figure(figsize=(9, 4.5))
    plt.barh(pot.index, pot.values, color="tab:blue")
    plt.xlabel("Potência média (W) = V x I")
    plt.title("Potência elétrica média por módulo")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "04_potencia_modulos.png"), dpi=120)
    plt.close()

    # Gráficos 3 e 5 são gerados pelo modelo e pela simulação
    treinar_modelo(verbose=False, gerar_grafico=True)
    simular_recuperacao(verbose=False, gerar_grafico=True)
    return sorted(os.listdir(DIR_GRAFICOS))


def analise_final():
    df = garantir_dados()
    ind = calcular_indicadores(verbose=False)
    erros = analisar_erros(verbose=False)
    tabela = ESTADO["resultados"].get("metricas")
    if tabela is None:
        tabela = treinar_modelo(verbose=False, gerar_grafico=False)
    if "heap" not in ESTADO["resultados"]:
        priorizar_alertas(verbose=False)
    if "euler" not in ESTADO["resultados"]:
        simular_recuperacao(verbose=False, gerar_grafico=False)
    painel = painel_gerenciamento(verbose=False)
    arquivos = gerar_graficos()

    titulo("12. ANÁLISE FINAL DOS RESULTADOS")
    b = tabela.loc["B: Linear múltipla"]
    base = tabela.loc["Baseline (sistema legado)"]
    top = ESTADO["resultados"]["heap"]["top"][0]
    eu = ESTADO["resultados"]["euler"]
    criticos = erros[erros["classe"] != "aceitável"].index.tolist()
    print(f"1. Rede: latência média {ind['latencia_media_ms']:.1f} ms, P95 "
          f"{ind['latencia_p95_ms']:.1f} ms, disponibilidade {ind['disponibilidade_pct']:.1f}%.")
    ev = ESTADO["resultados"]["indicadores_evento"]
    print(f"2. A tempestade elevou a latência média de {ev.iloc[0, 0]:.1f} para "
          f"{ev.iloc[1, 0]:.1f} ms e a perda de {ev.iloc[0, 1]:.2f}% para {ev.iloc[1, 1]:.2f}%.")
    print(f"3. A previsão legada tem erro relativo médio fora do aceitável em: "
          f"{', '.join(criticos) if criticos else 'nenhum módulo'}.")
    print(f"4. O modelo do SCIC reduziu o MAE de {base['MAE']:.2f} para {b['MAE']:.2f} ms "
          f"(R² {base['R2']:.2f} -> {b['R2']:.2f}).")
    print(f"5. Alerta mais urgente: {top.modulo} no ciclo {top.ciclo} (score {top.score}).")
    print(f"6. Após a tempestade, o COM-01 leva ~{eu['t_newton']:.1f} h para voltar ao normal.")
    print(f"7. Enlace Terra-Marte com ~{ind['atraso_luz_min_fim']:.1f} min de atraso (só ida):")
    print("   decisões críticas precisam ser tomadas localmente, com validação humana.")
    print(f"8. Recomendações: {(painel['recomendacao'] != 'monitoramento normal').sum()} "
          "módulos com ação sugerida no painel de gerenciamento.")
    print(f"\nGráficos salvos em '{os.path.basename(DIR_GRAFICOS)}/': {', '.join(arquivos)}")


# =============================================================================
# MENU PRINCIPAL
# =============================================================================
MENU = [
    ("1", "Carregar dados da colônia (CSV)", lambda: carregar_dados()),
    ("2", "Cadastrar novo registro", cadastrar_registro),
    ("3", "Consultar registros", lambda: consultar_registros()),
    ("4", "Indicadores de comunicação", lambda: calcular_indicadores()),
    ("5", "Erros numéricos (absoluto, relativo, ponto flutuante)", lambda: analisar_erros()),
    ("6", "Simulação numérica (Euler e Newton)", lambda: simular_recuperacao()),
    ("7", "Modelo de previsão e métricas (MAE, MSE, RMSE, R², AIC, BIC)", lambda: treinar_modelo()),
    ("8", "Priorizar alertas (heap)", lambda: priorizar_alertas()),
    ("9", "Busca por prefixo (trie)", lambda: buscar_prefixo()),
    ("10", "COA: dispositivos, bases numéricas e eletricidade", lambda: coa_eletricidade()),
    ("11", "Painel de gerenciamento inteligente", lambda: painel_gerenciamento()),
    ("12", "Analisar alertas de comunicação (fluxo completo)", lambda: analisar_alertas_comunicacao()),
    ("13", "Análise final + gerar gráficos", analise_final),
]


def menu():
    print("\n" + "#" * 78)
    print("#  SCIC - Sistema de Comunicação Interplanetária da Colônia | Aurora Siger  #")
    print("#" * 78)
    carregar_dados(verbose=False)
    print(f"Base carregada automaticamente: {len(ESTADO['df'])} registros.")
    acoes = {k: f for k, _, f in MENU}
    while True:
        print("\nMENU PRINCIPAL")
        for k, texto, _ in MENU:
            print(f" {k:>2}. {texto}")
        print("  0. Sair")
        op = ler("Escolha uma opção: ", "0")
        if op == "0":
            print("Encerrando o SCIC. Boa transmissão, Aurora Siger!")
            break
        if op in acoes:
            acoes[op]()
            ler("\n[Enter para voltar ao menu]")
        else:
            print("Opção inválida. Digite um número do menu.")


def demo():
    """Executa todas as funcionalidades em sequência (útil para o vídeo e testes)."""
    carregar_dados()
    consultar_registros(modulo_id="COM-02", status="alerta")
    calcular_indicadores()
    analisar_erros()
    simular_recuperacao()
    treinar_modelo()
    priorizar_alertas()
    buscar_prefixo(["co", "comunic", "0x03", "temp", "hab"])
    coa_eletricidade()
    painel_gerenciamento()
    analisar_alertas_comunicacao()
    analise_final()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        menu()
