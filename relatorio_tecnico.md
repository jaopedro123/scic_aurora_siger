# Relatório Técnico — SCIC
## Sistema de Comunicação Interplanetária da Colônia · Missão Aurora Siger · Fase 6

> Todos os números deste relatório foram gerados pelo próprio protótipo (`python codigo_fonte.py --demo`) sobre a base `dados_aurora_siger.csv`. Os gráficos citados estão na pasta `graficos_ou_imagens/`.

---

## 1. Contexto da solução

A colônia Aurora Siger já tem módulos conectados, registros operacionais e um núcleo cognitivo. O problema desta fase é outro: **confiar nos dados**. Os registros mostravam latências que variavam sem explicação, previsões que não batiam com o observado e alertas críticos analisados fora de ordem.

O SCIC é uma camada de **avaliação, organização e interpretação** dos dados de comunicação. Ele:

1. organiza a base simulada (CSV → Pandas, dicionários e listas);
2. calcula indicadores de comunicação e o atraso de luz Terra–Marte;
3. mede o erro absoluto e relativo da previsão de latência usada hoje (sistema legado);
4. simula a recuperação do enlace com Euler e estima o tempo de normalização com Newton;
5. treina um modelo de regressão melhor que o legado e o avalia com MAE, MSE, RMSE, R², AIC e BIC;
6. prioriza alertas com um **heap máximo** e faz busca por prefixo com uma **trie**, ambos implementados do zero;
7. relaciona tudo com COA (E/S, bases numéricas, IEEE 754, Lei de Ohm, potência);
8. gera um painel de gerenciamento inteligente com recomendações (redundância, manutenção preditiva, recalibração).

É uma **simulação computacional**: não há sensores físicos, APIs externas nem telemetria real, conforme as regras da atividade.

---

## 2. Descrição dos dados

**Arquivo:** `dados_aurora_siger.csv` — 150 registros = 10 módulos × 15 ciclos (sóis), de 01/09/2026 a 15/09/2026. Gerado de forma reprodutível por `gerar_dados.py` (semente 42).

| Campo | Significado |
|---|---|
| `id_registro`, `ciclo`, `data_registro` | identificação e momento do registro |
| `modulo_id`, `modulo`, `tipo`, `essencial` | módulo (habitação, agricultura, comunicação, controle, laboratório, suporte médico, armazenamento) e se é crítico para a vida/operação |
| `codigo_sensor` | código hexadecimal `0xTTNN` (TT = tipo, NN = ordem no tipo) |
| `carga_rede_pct`, `intensidade_sinal_dbm` | carga da rede local e força do sinal recebido |
| `tempestade_poeira` | 1 nos ciclos 9, 10 e 11 (evento global) |
| `latencia_prevista_ms` | estimativa do **sistema legado** (usa carga e sinal médios, ignora tempestades) |
| `latencia_observada_ms` | latência medida |
| `perda_pacotes_pct` | perda de pacotes no enlace |
| `tensao_v`, `corrente_a`, `potencia_w` | elétrica do transmissor/rádio do módulo (P = V·I) |
| `distancia_terra_mkm` | distância Terra–Marte simulada (milhões de km) |
| `status`, `prioridade`, `mensagem_alerta` | ativo / alerta / manutenção, prioridade 1–5 e resumo do alerta |

Composição: **113 ativos, 34 alertas, 3 manutenções programadas**. Além da tempestade, foram simulados três picos isolados (reinício do roteador do relé orbital, cabo de fibra danificado no laboratório, congestionamento de switch na Habitação Beta) para testar se as métricas e o heap tratam bem eventos raros.

**Limpeza e organização:** o sistema valida as 21 colunas obrigatórias, remove duplicados e linhas sem latência, converte datas e tipos e cria as colunas derivadas `erro_abs_ms` e `erro_rel`. Os módulos também ficam num **dicionário Python** (`catalogo['COM-01'] → nome, tipo, essencial, sensor`), e os alertas viram objetos `Alerta` (dataclass) dentro do heap.

---

## 3. Indicadores de comunicação

| Indicador | Valor |
|---|---|
| Latência média da rede interna | 107,8 ms |
| Mediana / P95 | 95,2 ms / 202,9 ms |
| Jitter (desvio-padrão) | 48,8 ms |
| Perda média de pacotes | 1,47 % |
| Disponibilidade (status ativo) | 75,3 % |
| Atraso de luz Terra–Marte (só ida) | 11,42 min (ciclo 1) → 10,60 min (ciclo 15) |

O atraso de luz é calculado por `distância / c`. Com ~21 minutos de ida e volta, **nenhuma decisão crítica pode esperar a Terra**: a colônia precisa de um sistema local que interprete os dados e priorize ações.

**Impacto da tempestade de poeira** (gráfico `01_latencia_por_ciclo.png`):

| | Latência média | Perda média | Sinal médio |
|---|---|---|---|
| Sem tempestade | 98,9 ms | 0,69 % | −63,6 dBm |
| Com tempestade | 143,3 ms | 4,58 % | −77,1 dBm |

A previsão legada fica plana nos ciclos 9–11, enquanto a latência real sobe ~45%. É aí que nasce a maior parte dos erros.

---

## 4. Análise numérica e avaliação de erros

**Definições usadas**

- Erro absoluto: `|latência observada − latência prevista|` (ms)
- Erro relativo: `erro absoluto / |latência observada|` (adimensional)

**Critério da colônia:** erro relativo ≤ 10 % aceitável · 10–25 % atenção · > 25 % crítico. Os 10 % vêm da folga que os protocolos de retransmissão toleram sem perder janela; acima de 25 % a previsão já não serve para agendar transmissões.

**Exemplo:** registro R0095 (Comunicação Orbital Phobos, ciclo 10): previsto 199,7 ms, observado 267,1 ms → erro absoluto **67,4 ms**, relativo **25,2 % (crítico)**.

**Por que o erro relativo é necessário:**

| Módulo | Latência típica | Erro abs. médio | Erro rel. médio |
|---|---|---|---|
| COM-01 Comunicação Central | ~75 ms | 11,2 ms | 12,6 % |
| COM-02 Relé Orbital Phobos | ~215 ms | 19,5 ms | 8,0 % |

Pelo erro absoluto, o relé orbital parece o pior. Pelo relativo, a Comunicação Central erra proporcionalmente mais. O erro relativo põe módulos de escalas diferentes na mesma régua (gráfico `02_erro_relativo_modulos.png`).

**Resultado:** dos 150 registros, 88 têm erro aceitável, 31 de atenção e 31 críticos. Oito dos dez módulos têm erro relativo médio acima de 10 % — o sistema legado não é confiável.

**Precisão numérica e ponto flutuante**

- `0.1 + 0.2 = 0.30000000000000004`: decimais não têm representação binária exata. O SCIC compara valores com tolerância (`math.isclose`), nunca com `==`.
- Somar 300 mil leituras em `float32` de forma ingênua gerou erro de **4.484 ms** (relativo 1,39·10⁻⁴) contra a soma de referência (`math.fsum`). Em sensores embarcados que usam precisão simples, totais longos precisam de float64 ou soma compensada.
- O CSV guarda 1 casa decimal: erro máximo de arredondamento de 0,05 ms, no pior caso 0,117 % — desprezível perto dos erros de previsão. **Conclusão:** o problema da colônia não é aritmético; é de **modelo**.

**Quando o erro é preocupante:** em módulo essencial (suporte médico, controle ambiental, comunicação), mesmo "atenção" merece revisão; em módulos não essenciais, só "crítico" gera ação imediata.

### 4.1 Métodos numéricos: Euler e Newton

Após a tempestade, a latência do COM-01 volta ao normal. Modelamos isso com a lei de relaxação

`dL/dt = −k (L − L_eq)`, solução exata `L(t) = L_eq + (L0 − L_eq)·e^(−kt)`

com L0 = 103,63 ms (média na tempestade), L_eq = 68,42 ms (média normal) e k = 0,35 h⁻¹.

| Passo h | Erro máximo de Euler vs exata |
|---|---|
| 2,0 h | 6,92 ms |
| 1,0 h | 2,65 ms |
| 0,5 h | 1,22 ms |
| 0,1 h | 0,23 ms |

Reduzir o passo pela metade reduz o erro ~pela metade: Euler é de **1ª ordem**. Discretização grossa subestima a latência (gráfico `05_euler_newton.png`), o que poderia liberar o enlace cedo demais.

**Newton–Raphson** em `f(t) = L(t) − limite`, com limite = L_eq + 10 % = 75,27 ms, convergiu em 7 iterações (variação abaixo de 10⁻⁸ já na 6ª) para **t = 4,680370 h**, igual à solução analítica (diferença 2,7·10⁻¹⁵ h). Na prática: o enlace redundante deve permanecer ativo por ~4,7 h após a tempestade.

---

## 5. Modelo simples e avaliação de performance

**Problema:** prever `latencia_observada_ms` (regressão).
**Variáveis:** carga da rede, intensidade do sinal, tempestade e módulo (one-hot).
**Divisão:** 80 % treino (120) / 20 % teste (30), `random_state=42`, estratificada pela tempestade para que o teste também tenha ciclos de tempestade.

**Métricas no conjunto de teste**

| Modelo | MAE (ms) | MSE | RMSE (ms) | R² |
|---|---|---|---|---|
| Baseline — sistema legado | 12,63 | 387,09 | 19,68 | 0,745 |
| A — Linear simples (só carga) | 33,30 | 1660,66 | 40,75 | −0,095 |
| **B — Linear múltipla** | **4,46** | **28,59** | **5,35** | **0,981** |
| C — Ridge (Grid Search, α = 0,01) | 4,26 | 26,59 | 5,16 | 0,982 |

**Comparação no treino (AIC/BIC, menor é melhor)**

| Modelo | Parâmetros | AIC | BIC |
|---|---|---|---|
| A — Linear simples | 2 | 943,2 | 948,8 |
| B — Linear múltipla | 13 | 536,8 | 573,1 |

**Coeficientes do modelo B (fazem sentido físico):** +0,68 ms por ponto de carga; −0,52 ms por dBm (sinal mais forte, menos latência); **+37,3 ms durante tempestade**.

**Interpretação**

- O modelo B reduz o MAE de 12,6 para 4,5 ms (−65 %). O ganho vem principalmente de incluir a tempestade e o sinal, que o legado ignora.
- **R² alto não significa modelo perfeito.** R² = 0,981 diz que o modelo explica a variação *média* da latência; ainda há resíduos de até ~12 ms (gráfico `03_modelo_previsao.png`) e o modelo não prevê picos causados por falhas físicas.
- **MAE × RMSE:** a razão RMSE/MAE é 1,56 no legado e 1,20 no modelo B. Quando o RMSE se afasta muito do MAE, poucos registros concentram erros grandes — no legado, os ciclos de tempestade. No modelo B os erros são pequenos e espalhados.
- O modelo A tem R² negativo: só a carga da rede é **pior do que prever a média**. Um único número (R²) já mostra isso, mas é a comparação conjunta das métricas que explica o porquê.
- AIC e BIC são bem menores no B: a complexidade extra (13 parâmetros) é compensada pelo ganho de ajuste, mesmo com a penalização mais forte do BIC.
- O Grid Search escolheu α = 0,01 (regularização quase nula); o Ridge ficou praticamente igual ao B, sinal de que **não há sobreajuste relevante**. Mantivemos o B como modelo do SCIC por ser mais simples de explicar.

**Uso do modelo no sistema:** as previsões do modelo B entram no score do heap — se a latência observada passa 20 % do que o próprio modelo espera, o alerta ganha pontos extras de anomalia.

---

## 6. Priorização de alertas com heap

**Representação:** cada alerta é um objeto `Alerta(score, ciclo, id_registro, modulo_id, modulo, mensagem, detalhes)`.

**Critério de prioridade (score):**

```
score = prioridade_operacional (1–5) × 10
      + 15 se o módulo é essencial
      + excesso de latência sobre o previsto (até 30 pontos)
      + perda de pacotes × 2 (até 20 pontos)
      + 5 se a latência passa 20 % do previsto pelo modelo B (anomalia)
      − 10 se for manutenção programada
```
Empate → o alerta mais recente vem antes (chave `(score, ciclo)`).

**Estrutura:** `HeapMaxAlertas` em vetor, árvore binária quase completa: `pai(i) = (i−1)//2`, filhos `2i+1` e `2i+2`. A inserção usa **heapify-up**; a remoção do máximo troca raiz e último elemento e aplica **heapify-down**. Também implementamos **Heap Sort** in-place.

**Resultado:** 37 alertas (34 alertas + 3 manutenções) entram no heap com 68 comparações. O mais urgente é **Controle Ambiental, ciclo 9 (score 105,06)**: módulo essencial, latência 124,1 ms contra 71,5 ms previstos e 5 % de perda. Em seguida vêm três registros do Suporte Médico na tempestade. As manutenções programadas ficam no fim da fila — são esperadas.

**Vantagem sobre lista simples:** para extrair os 8 mais urgentes, o heap fez **64 comparações** e a lista simples (varredura do máximo a cada vez) **260**. Heap: inserir e remover em O(log n), ver o topo em O(1). Lista: encontrar o máximo em O(n) a cada consulta ou reordenar tudo em O(n log n) a cada alerta novo. Numa colônia que recebe alertas continuamente, o heap escala melhor.

---

## 7. Busca de registros com trie

**Conteúdo da trie:** 67 chaves e 675 nós — nomes de módulos (completos e cada palavra), IDs (`COM-01`), códigos de sensores (`0x0301`), mensagens de alerta e 11 comandos cadastrados (`comando_reiniciar_enlace`, `controle_termico`, `ativar_enlace_redundante`…). Cada nó terminal guarda o texto original e as referências (módulos ou registros).

**Operações:** `inserir` e `buscar` em O(m) (m = tamanho da chave); `autocompletar` desce até o nó do prefixo e percorre só aquela subárvore. A normalização remove acentos e maiúsculas, então `comunic` encontra "Comunicação".

**Exemplos reais da execução:**

| Prefixo | Resultados |
|---|---|
| `co` | COM-01, COM-02, Comando, Comando de Missão, Comunicação…, Controle, Controle Ambiental, comandos `controle_*`, `consultar_status` |
| `com` | COM-01, COM-02, Comando de Missão, Comunicação Central, Comunicação Orbital Phobos, `comunicar_terra`… |
| `0x04` | 0x0401 (CTL-01), 0x0402 (CMD-01) — todos os sensores do tipo "controle" |
| `temp` | as duas mensagens de tempestade, com os IDs dos registros afetados |

> Observação: o enunciado cita "com" retornando "Controle"; como "controle" começa com "con", na trie ele aparece pelo prefixo **"co"**, que é o comportamento correto da estrutura.

**Por que trie:** o custo da busca depende do tamanho do prefixo, não da quantidade de registros. Uma lista exigiria comparar o prefixo com todas as chaves (O(n·m)). Para o operador digitando um código de sensor parcial durante uma emergência, o autocomplete encontra o equipamento em poucos caracteres. A codificação `0xTTNN` foi pensada para isso: o prefixo do tipo agrupa os sensores na mesma subárvore.

---

## 8. Dispositivos, bases numéricas e eletricidade básica (COA)

**Entrada (conceitual):** sensores de latência (eco/ping), medidores de intensidade de sinal em dBm, sensores de tensão e corrente dos transmissores, teclado do operador e o arquivo CSV, que simula o barramento de telemetria.
**Saída:** terminal (menu e relatórios), gráficos PNG, CSV atualizado; numa versão real, painel na sala de controle e alarme sonoro.
**Interfaces (conceitual):** fibra/cabo entre módulos, Wi-Fi nas habitações, Bluetooth para sensores portáteis, USB para manutenção, rádio UHF até o relé em Phobos e banda X/Ka até a Terra.

**Bases numéricas — códigos de sensor `0xTTNN`:**

| Módulo | Hex | Decimal | Binário (16 bits) | Tipo | Nº |
|---|---|---|---|---|---|
| HAB-01 | 0x0101 | 257 | 00000001 00000001 | 1 | 1 |
| COM-02 | 0x0302 | 770 | 00000011 00000010 | 3 | 2 |
| CMD-01 | 0x0402 | 1026 | 00000100 00000010 | 4 | 2 |
| DAT-01 | 0x0701 | 1793 | 00000111 00000001 | 7 | 1 |

Conversão manual (passo a passo de 0x0302): `0·16³ + 3·16² + 0·16¹ + 2·16⁰ = 770`; 770 em binário por divisões sucessivas = `1100000010`. O tipo sai por deslocamento de bits (`código >> 8`) e o número por máscara (`código & 0xFF`). O código confere as funções manuais com `int()`/`format()` do Python.

**IEEE 754:** a latência 64,4 ms em float32 fica `sinal 0 | expoente 10000101 (+6) | mantissa 00000001100110011001101` e é armazenada como 64,4000015258789 (erro 1,5·10⁻⁶ ms).

**Eletricidade — transmissor COM-01 (ciclo 1):** V = 47,85 V, I = 2,72 A

- Potência: P = V·I = **130,15 W**
- Lei de Ohm: R = V/I = **17,59 Ω**; conferência P = I²·R = 130,15 W
- Em dBm: 10·log₁₀(130.150 mW) = **51,1 dBm**
- Energia por sol (24,62 h): ~3,2 kWh

No relé orbital COM-02, a potência média sobe de **178,7 W para 203,4 W (+14 %)** na tempestade, porque o transmissor aumenta a corrente para vencer a atenuação da poeira — justamente quando a microrrede solar gera menos. O gráfico `04_potencia_modulos.png` mostra que os dois módulos de comunicação são os maiores consumidores.

---

## 9. Gerenciamento inteligente da comunicação

O painel (opção 11 do menu) resume cada módulo com P95 de latência, tendência fora da tempestade (inclinação por regressão, em ms/ciclo), perda na tempestade, erro relativo em dias normais e recomendação automática.

| Achado do SCIC | Conceito | Ação recomendada |
|---|---|---|
| 9 de 10 módulos passam de 4 % de perda na tempestade | **Redundância** | migrar automaticamente para o enlace secundário (fibra ou rota via COM-01) enquanto a perda estiver alta |
| COM-01 recupera a latência em ~4,7 h (Newton) | **Automação + monitoramento contínuo** | manter a rota redundante até o indicador voltar ao limite, e não só até a poeira baixar |
| COM-02 com tendência de +1,32 ms/ciclo e pico no ciclo 13 | **Manutenção preditiva** | inspecionar roteador e antena do relé antes da próxima janela de transmissão para a Terra |
| MED-01 e HAB-02 erram >10 % mesmo sem tempestade | **Sensores/medidores inteligentes** | recalibrar a previsão desses módulos ou substituir o legado pelo modelo B |
| Comunicação = maior consumo e +14 % na tempestade | **Microrredes** | coordenar potência de transmissão com a geração solar; adiar tráfego não essencial (dados científicos) em tempestade |
| Atraso de ~10,6 min até a Terra | **Rede inteligente de comunicação** | decisão local com heap; Terra recebe relatório consolidado, não dados brutos |

**Armazenamento de dados:** o CSV histórico é o que permite calcular tendência, treinar o modelo e comparar com o baseline. Sem histórico não haveria manutenção preditiva.

**Automação com supervisão humana:** o SCIC sugere a ação e ordena os alertas; quem executa a troca de rota ou o desligamento de carga é o operador.

---

## 10. Reflexão social, cultural e sustentável

**Uso eficiente da comunicação e sustentabilidade.** Em Marte, cada watt de transmissão sai de uma microrrede limitada. Priorizar com o heap significa transmitir primeiro o que protege vidas (suporte médico, controle ambiental) e adiar o resto, em vez de aumentar a potência de todos os rádios na tempestade. Mandar à Terra um relatório consolidado em vez de dados brutos reduz o uso do enlace mais caro da colônia.

**Conhecimentos tradicionais e respeito à natureza.** Povos indígenas brasileiros organizam o uso de recursos pelos ciclos do ambiente — plantar, pescar e se deslocar conforme as estações, as chuvas e os sinais da natureza, sem tirar mais do que o território repõe. O SCIC segue essa lógica: em vez de "forçar" o sinal contra a tempestade, ele reconhece o ciclo da poeira, reduz o tráfego e espera a recuperação, que o próprio sistema calcula (~4,7 h). Adaptar-se ao ambiente marciano é mais sustentável do que lutar contra ele.

**Diversidade e sistemas não enviesados.** O critério de prioridade foi escrito de forma explícita e dá peso à função do módulo (suporte à vida), não a quem o usa. Um sistema que priorizasse, por exemplo, os módulos da liderança ou de uma equipe específica reproduziria desigualdades. A valorização de diferentes saberes — inclusive tradições afro-brasileiras de organização comunitária e decisão coletiva — lembra que regras de prioridade devem ser discutidas pela comunidade, e não impostas por quem escreve o código.

**Transparência.** Todo score pode ser recalculado à mão a partir do CSV; as métricas mostram onde o modelo erra (picos de falha física) e os limites de erro aceitável estão declarados. Uma decisão baseada em dados só é legítima se a comunidade consegue entender por que ela foi tomada.

**Responsabilidade humana.** O modelo tem R² de 0,98 e mesmo assim não previu nenhum dos três picos de falha. Por isso o SCIC **recomenda**, e o operador decide. Automação sem supervisão em um ambiente onde a ajuda da Terra está a 20 minutos de distância é um risco, não uma eficiência.

**Linguagem e inclusão.** Mensagens do sistema descrevem o problema técnico ("perda de pacotes elevada") sem termos ambíguos ou que culpem pessoas, e a busca ignora acentos e maiúsculas para não excluir quem digita de forma diferente.

---

## 11. Limitações

- Os dados são simulados: os coeficientes "corretos" foram definidos no gerador, então o modelo B tem vantagem natural. Com dados reais o R² seria menor.
- O modelo é linear e não prevê falhas físicas (picos); elas aparecem como alertas, não como previsões.
- A recuperação após a tempestade usa um k fixo (0,35 h⁻¹), adotado como parâmetro de engenharia e não estimado dos dados (a base tem resolução de um registro por ciclo).
- Os pesos do score do heap foram definidos pela equipe; outros pesos mudariam a ordem dos alertas do meio da fila.
- A tendência do COM-02 é influenciada pelo pico do ciclo 13; com 12 pontos normais por módulo, a manutenção preditiva é indicativa.
- O atraso de luz usa distâncias simuladas e desconsidera processamento nos relés.

## 12. Possíveis melhorias

- Estimar k da recuperação com dados de maior frequência (por hora) e comparar Euler com métodos de ordem maior.
- Validação cruzada com séries temporais (treinar no passado, testar no futuro) em vez de divisão aleatória.
- Incluir variáveis de hardware (temperatura do rádio, idade do equipamento) para prever picos.
- Heap indexado para atualizar o score de um alerta já na fila sem reinseri-lo.
- Automatizar o fluxo "detectar → priorizar → notificar operador" com um agente low-code/webhook (conceitual nesta fase).
- Painel visual na sala de controle a partir dos mesmos indicadores.
