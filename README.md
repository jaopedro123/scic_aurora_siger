# SCIC — Sistema de Comunicação Interplanetária da Colônia

**Missão Aurora Siger · Fase 6 · Atividade integradora**

Protótipo em Python que organiza os dados operacionais e de comunicação da colônia, avalia a confiabilidade das previsões de latência, prioriza alertas críticos com **heap**, faz busca por prefixo com **trie** e gera um relatório de apoio à decisão.

## Objetivo

A colônia já registra latência, sinal, energia e alertas de cada módulo, mas a previsão de latência usada hoje erra muito (principalmente em tempestades de poeira) e os alertas não são atendidos na ordem certa. O SCIC:

- calcula indicadores de comunicação (latência, P95, jitter, perda, disponibilidade, atraso de luz Terra–Marte);
- mede erro absoluto e relativo da previsão atual e discute ponto flutuante;
- simula a recuperação do enlace com **Euler** e estima o tempo de normalização com **Newton**;
- treina um modelo de regressão (scikit-learn) e o avalia com **MAE, MSE, RMSE, R², AIC e BIC** e **Grid Search**;
- prioriza alertas com **heap máximo** (heapify-up / heapify-down / Heap Sort) implementado do zero;
- faz busca e autocomplete por prefixo com **trie** implementada do zero;
- relaciona o sistema com COA: dispositivos de E/S, conversão hex/decimal/binário, IEEE 754, tensão, corrente, potência e Lei de Ohm;
- gera um painel de gerenciamento inteligente (redundância, manutenção preditiva, recalibração).

## Arquivos da entrega

| Arquivo | Conteúdo |
|---|---|
| `codigo_fonte.py` | **Arquivo principal.** Menu no terminal e todas as funcionalidades |
| `dados_aurora_siger.csv` | Base simulada: 150 registros (10 módulos × 15 ciclos) |
| `gerar_dados.py` | Gera a base de novo, de forma reprodutível (semente 42). Opcional |
| `relatorio_tecnico.md` | Relatório técnico completo com os resultados |
| `roteiro_video.md` | Roteiro do vídeo de apresentação (até 5 min) |
| `link_video.txt` | Link do vídeo no YouTube (não listado) |
| `graficos_ou_imagens/` | Gráficos PNG gerados pelo sistema |
| `README.md` | Este arquivo |

## Dependências

Somente bibliotecas trabalhadas na fase:

- Python 3.10 ou superior
- numpy, pandas, matplotlib, scikit-learn

```bash
pip install numpy pandas matplotlib scikit-learn
```

Heap e trie **não** usam bibliotecas prontas (`heapq` não é utilizado): foram implementados no próprio código.

## Como executar

Na pasta do projeto:

```bash
# Menu interativo
python codigo_fonte.py

# Executa todas as funcionalidades em sequência (bom para conferir a saída)
python codigo_fonte.py --demo

# (Opcional) recriar a base simulada
python gerar_dados.py
```

### Menu

```
  1. Carregar dados da colônia (CSV)
  2. Cadastrar novo registro
  3. Consultar registros
  4. Indicadores de comunicação
  5. Erros numéricos (absoluto, relativo, ponto flutuante)
  6. Simulação numérica (Euler e Newton)
  7. Modelo de previsão e métricas (MAE, MSE, RMSE, R², AIC, BIC)
  8. Priorizar alertas (heap)
  9. Busca por prefixo (trie)
 10. COA: dispositivos, bases numéricas e eletricidade
 11. Painel de gerenciamento inteligente
 12. Analisar alertas de comunicação (fluxo completo)
 13. Análise final + gerar gráficos
  0. Sair
```

A base é carregada automaticamente ao abrir o menu. A opção 2 grava o novo registro no CSV.

### Exemplos de execução

**Busca por prefixo (opção 9):**
```
Digite um prefixo (Enter para sair): 0x04
> prefixo '0x04': 2 resultado(s)
   - 0x0401                                           [CTL-01]
   - 0x0402                                           [CMD-01]
```

**Top alertas (opção 8):**
```
 1. [105.06] ciclo  9 CTL-01 Controle Ambiental   obs 124.1 ms | prev 71.5 | perda 5.03%
     -> Tempestade de poeira: perda de pacotes elevada
 2. [103.74] ciclo 10 MED-01 Suporte Médico       obs 106.2 ms | prev 56.6 | perda 4.37%
...
Extrair os 8 mais urgentes: heap = 64 comparações | lista simples = 260 comparações
```

**Modelo (opção 7):**
```
                              MAE       MSE    RMSE     R2
Baseline (sistema legado)  12.627   387.092  19.675  0.745
B: Linear múltipla          4.455    28.588   5.347  0.981
```

## Principais resultados

- A tempestade de poeira (ciclos 9–11) eleva a latência média de 98,9 para 143,3 ms e a perda de 0,69 % para 4,58 %.
- A previsão legada tem erro relativo médio acima de 10 % em 8 dos 10 módulos.
- O modelo do SCIC reduz o MAE de 12,6 para 4,5 ms (R² 0,75 → 0,98).
- O alerta mais urgente é o Controle Ambiental no ciclo 9.
- Após a tempestade, a Comunicação Central leva ~4,7 h para voltar ao normal (Newton).

Detalhes e discussão em `relatorio_tecnico.md`.
# scic_aurora_siger
