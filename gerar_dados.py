"""
gerar_dados.py - Gerador da base simulada da Aurora Siger (SCIC)

Gera o arquivo dados_aurora_siger.csv de forma REPRODUTÍVEL (semente fixa = 42).
A base já acompanha a entrega; este script só é necessário se você quiser
recriá-la ou alterar os parâmetros da simulação.

Cenário simulado:
- 10 módulos da colônia x 15 ciclos (sóis) = 150 registros.
- Ciclos 9, 10 e 11: tempestade de poeira que degrada o sinal e aumenta a latência.
- Alguns picos isolados de latência (reinício de roteador, falha de cabo) para
  criar registros com erro grande e mostrar a diferença entre MAE e RMSE.
- "latencia_prevista_ms" é a estimativa do sistema LEGADO da colônia: usa a carga
  média do módulo e NÃO considera tempestades. Ela serve de baseline.

Uso: python gerar_dados.py
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEMENTE = 42
N_CICLOS = 15
CICLOS_TEMPESTADE = {9, 10, 11}
DATA_INICIAL = date(2026, 9, 1)

# (id, nome, tipo, essencial, latência base [ms], tensão nominal [V],
#  corrente nominal [A], carga média de rede [%], sinal médio [dBm], código do tipo)
MODULOS = [
    ("HAB-01", "Habitação Alfa",             "habitacao",      1,  55, 24, 3.0, 55, -62, 0x01),
    ("HAB-02", "Habitação Beta",             "habitacao",      1,  60, 24, 2.8, 50, -64, 0x01),
    ("AGR-01", "Estufa Agrícola",            "agricultura",    0,  95, 24, 4.5, 40, -72, 0x02),
    ("COM-01", "Comunicação Central",        "comunicacao",    1,  22, 48, 2.5, 75, -55, 0x03),
    ("COM-02", "Comunicação Orbital Phobos", "comunicacao",    1, 140, 48, 3.5, 65, -78, 0x03),
    ("CTL-01", "Controle Ambiental",         "controle",       1,  40, 24, 2.0, 45, -60, 0x04),
    ("CMD-01", "Comando de Missão",          "controle",       1,  30, 12, 1.8, 60, -57, 0x04),
    ("LAB-01", "Laboratório Geológico",      "laboratorio",    0,  85, 12, 2.2, 35, -74, 0x05),
    ("MED-01", "Suporte Médico",             "suporte_medico", 1,  35, 12, 1.5, 30, -59, 0x06),
    ("DAT-01", "Armazenamento de Dados",     "armazenamento",  0,  28, 12, 2.6, 70, -58, 0x07),
]

# Eventos pontuais (módulo, ciclo) -> (ms extras, mensagem)
PICOS = {
    ("COM-02", 13): (58.0, "Pico de latência: reinício do roteador do relé orbital"),
    ("LAB-01", 5):  (42.0, "Pico de latência: cabo de fibra danificado"),
    ("HAB-02", 7):  (35.0, "Pico de latência: congestionamento no switch local"),
}
# Manutenções programadas (módulo, ciclo)
MANUTENCOES = {("AGR-01", 6), ("LAB-01", 12), ("DAT-01", 14)}


def gerar():
    rng = np.random.default_rng(SEMENTE)
    registros = []
    contador = 1

    for ciclo in range(1, N_CICLOS + 1):
        tempestade = 1 if ciclo in CICLOS_TEMPESTADE else 0
        # Distância Terra-Marte simulada (milhões de km), diminuindo lentamente
        distancia = round(205.4 - 1.05 * (ciclo - 1) + rng.normal(0, 0.05), 2)

        for idx, (mid, nome, tipo, essencial, lat_base, v_nom, i_nom,
                  carga_media, sinal_medio, cod_tipo) in enumerate(MODULOS):
            carga = float(np.clip(rng.normal(carga_media, 12), 5, 98))
            sinal = sinal_medio + rng.normal(0, 3) - (12 if tempestade else 0)

            # Latência "real" observada (o que o sensor mediu)
            lat_obs = (lat_base
                       + 0.6 * carga
                       + 0.9 * (-sinal - 55)
                       + tempestade * (25 + 0.15 * lat_base)
                       + rng.normal(0, 4))
            mensagem_pico = None
            if (mid, ciclo) in PICOS:
                extra, mensagem_pico = PICOS[(mid, ciclo)]
                lat_obs += extra

            # Latência prevista pelo sistema legado (baseline): carga média e
            # sinal médio, sem saber da tempestade
            lat_prev = lat_base + 0.6 * carga_media + 0.9 * (-sinal_medio - 55)

            perda = max(0.0, 0.3 + 0.08 * (-sinal - 60) + tempestade * 3
                        + rng.normal(0, 0.4))

            tensao = v_nom + rng.normal(0, 0.02 * v_nom)
            corrente = i_nom * (0.8 + 0.4 * carga / 100) + rng.normal(0, 0.05)
            if tipo == "comunicacao" and tempestade:
                corrente *= 1.15  # transmissor aumenta potência na tempestade
            potencia = tensao * corrente

            # Status, prioridade e mensagem
            if (mid, ciclo) in MANUTENCOES:
                status, prioridade = "manutencao", 2
                mensagem = "Manutenção programada do módulo"
            elif lat_obs > 1.25 * lat_prev or perda > 4:
                status = "alerta"
                prioridade = 3 + essencial
                if lat_obs > 1.5 * lat_prev:
                    prioridade = min(5, prioridade + 1)
                if mensagem_pico:
                    mensagem = mensagem_pico
                elif tempestade and perda > 4:
                    mensagem = "Tempestade de poeira: perda de pacotes elevada"
                elif tempestade:
                    mensagem = "Tempestade de poeira: latência acima do previsto"
                elif perda > 4:
                    mensagem = "Perda de pacotes elevada no enlace"
                else:
                    mensagem = "Latência acima do previsto"
            else:
                status, prioridade = "ativo", 1
                mensagem = "Operação normal"

            # Código do sensor em hexadecimal: byte alto = tipo do módulo,
            # byte baixo = ordem do módulo dentro do tipo (ex.: 0x0302 = COM-02, 0x0402 = CMD-01)
            numero_no_tipo = sum(1 for m in MODULOS[:idx + 1] if m[9] == cod_tipo)
            codigo_sensor = f"0x{cod_tipo:02X}{numero_no_tipo:02X}"

            registros.append({
                "id_registro": f"R{contador:04d}",
                "ciclo": ciclo,
                "data_registro": (DATA_INICIAL + timedelta(days=ciclo - 1)).isoformat(),
                "modulo_id": mid,
                "modulo": nome,
                "tipo": tipo,
                "essencial": essencial,
                "codigo_sensor": codigo_sensor,
                "carga_rede_pct": round(carga, 1),
                "intensidade_sinal_dbm": round(sinal, 1),
                "tempestade_poeira": tempestade,
                "latencia_prevista_ms": round(lat_prev, 1),
                "latencia_observada_ms": round(lat_obs, 1),
                "perda_pacotes_pct": round(perda, 2),
                "tensao_v": round(tensao, 2),
                "corrente_a": round(corrente, 2),
                "potencia_w": round(potencia, 2),
                "distancia_terra_mkm": distancia,
                "status": status,
                "prioridade": prioridade,
                "mensagem_alerta": mensagem,
            })
            contador += 1

    return pd.DataFrame(registros)


if __name__ == "__main__":
    df = gerar()
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dados_aurora_siger.csv")
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"Base gerada: {destino} ({len(df)} registros)")
    print(df["status"].value_counts().to_string())
