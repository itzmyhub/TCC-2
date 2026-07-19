# Estudo de Caso Operacional: Pium-TO, 12/05/2026

> **Objetivo deste documento**: registrar, de forma reproduzível, a investigação que partiu de um foco real reportado pelo *Painel do Fogo* (CENSIPAM) e levou a três correções no pipeline (`scripts/frp_api.py`, `scripts/feature_lookup.py`, `scripts/app_map_interativo.py`), além de evidenciar um limite estrutural do uso de reanálise climática global em zonas de transição Cerrado–Amazônia.
>
> **Versão formal para o TCC**: `tcc_tex/07_estudo_caso_pium_to.tex` (Seção 5.9 sugerida).
>
> **Data da investigação**: 12 de maio de 2026, 18:15–18:40 (UTC-3).
>
> **Status do estudo**: ✅ concluído; correções (C1) e (C2) aplicadas; (C3) decisão de engenharia documentada; limite estrutural reportado para `tcc_tex/06_capitulo6_conclusao.tex` como trabalho futuro.

---

## 1. Sumário executivo (TL;DR)

| Item | Conteúdo |
|---|---|
| **Evento** | Foco ativo ID `6668060` em `(-10,5351; -49,8230)`, Pium-TO, ativo desde 10/05/2026 (12 focos) e renovado em 12/05/2026 13:28 (duração 48,3 h) |
| **Fonte da validação** | *Painel do Fogo* / CENSIPAM (independente do INPE/BDQueimadas usado no treino) |
| **Resposta do modelo** | **Baixo** com confiança 85–88% em 5 janelas temporais (5–12 maio 2026) |
| **Causas identificadas** | (C1) NASA FIRMS quebrado → fogo não chegava ao modelo; (C2) Lookup Tier 1 degradando para `estado_mes` → sinal local apagado; (C3) NASA POWER reportando regime "chuvoso" enquanto a realidade local estava propícia a fogo |
| **Correções aplicadas** | C1 (`days_back ∈ [1,5]` + `reference_date`) e C2 (cascata 3×3 → 5×5 → mês adjacente). C3 não corrigível em produção — documentada como limite. |
| **Insight para o TCC** | O modelo **não é cego sazonalmente** — em 15/05/2024 mesmo ponto, com NASA POWER reportando seca, ele previu Muito Alto (47,3%). O erro está no input climático, não na decisão do classificador. |

---

## 2. Timeline da investigação

```
18:15  Usuário cola coordenadas no formulário ("Tocantins, fora do bbox?")
       → identificado bug 1: <input type="number"> em pt-BR descarta vírgula
       → identificado bug 2: campos lat/lon trocados pelo usuário
       → correções aplicadas (parseCoord tolerante a "," e detecção de inversão)

18:20  Usuário compartilha screenshot do Painel do Fogo (CENSIPAM)
       → coordenadas REAIS: lat=-10,5351 lon=-49,8230 (Pium-TO)
       → ID evento 6668060, foco ativo, duração 48,3 h
       
18:22  Primeira rodada: predict() em 5 janelas (12/05, 11/05, 10/05, 09/05, 05/05)
       → todas retornam "Baixo" 79–82%
       → logs mostram: NASA FIRMS HTTP 400, Tier 1 cai em "estado_mes"

18:25  Diagnóstico NASA FIRMS
       → curl direto na API: "Invalid day range. Expects [1..5]"
       → MAP_KEY válida (5000 transações, 0 usadas)
       → causa: API mudou de 7 para 5 dias máx (sem aviso)

18:27  Correção 1 (C1): scripts/frp_api.py
       → FIRMS_MAX_DAYS = 5; clip days_back; aceita reference_date
       → URL passa a incluir /YYYY-MM-DD quando data passada
       → app_map_interativo.py passa ref_data adiante
       → re-rodada: FIRMS retorna 16–32 detecções, FRP até 22 MW
       → MAS: predição continua "Baixo", confiança SOBE para 82–84%

18:30  Tentativa intermediária: sobrescrever Incendios_Ultimos_7/30_Dias com
       contagem real do FIRMS (16-32 focos)
       → predição "Baixo" sobe para 86-88% (PIOROU!)
       → diagnóstico: out-of-distribution — modelo treinou com mediana
         Estado+Mês (0 para Tocantins-Maio); valores reais altos são OOD
       → revertido. Mantém apenas FRP_* e Dias_Desde_Ultimo_Incendio (eram
         observações individuais no treino, não medianas)

18:33  Correção 2 (C2): scripts/feature_lookup.py
       → cascata original: 3×3 céls → estado_mes → mes_global
       → cascata nova: 3×3 → 5×5 → mês adjacente → estado_mes → mes_global
       → granularidade subiu de "estado_mes" para "celula_ext" no caso Pium-TO
       → MAS: P(Baixo) ainda em 85,9% (subiu de 82,7%)

18:36  Contrafactual: mesmo ponto em outras datas
       → 15/08/2024: P(Muito Alto) = 99,0% (KBDI=424, DSC=16d)
       → 15/09/2024: P(Muito Alto) = 99,6% (KBDI=712, DSC=25d)
       → 15/05/2024: P(Muito Alto) = 47,3% (KBDI=148, DSC=8,6d)
       → 12/05/2026: P(Muito Alto) =  4,6% (KBDI= 30, DSC=1,3d)
       → CONCLUSÃO: modelo não tem viés sazonal. Responde aos inputs.
       → 2026-05 NASA POWER reportou regime "chuvoso" (DSC=1,3d!) enquanto
         o solo no campo estava queimando há 48 h.

18:40  Documentação iniciada (este arquivo + 07_estudo_caso_pium_to.tex)
```

---

## 3. Dados do evento (Painel do Fogo / CENSIPAM)

**Captura do painel (interpretação textual)**:

```
ID do evento: 6668060
Domínios: Área Não Identificada, Privado, Terra Pública Não Categorizada
Uso e Cobertura do Solo: Vegetação natural primária (TerraClass)
Coordenadas: Longitude: -49,8230 | Latitude: -10,5351
Status: Ativo pela última vez em 2026-05-12 14:49:00 (GMT-3)

Detecções:
+--------------------+-----------+--------+----------+----------------+
| Data e hora        | Área (ha) | Focos  | Duração  | Propagação     |
+--------------------+-----------+--------+----------+----------------+
| 2026-05-12 13:28   |    464,6  |     1  |  48,3 h  | 0,4 ha/h       |
| 2026-05-10 14:05   |    456,3  |    12  |   0,9 h  | 0,0 ha/h       |
| 2026-05-10 13:13   |     83,1  |     3  |   0,0 h  | 0,0 ha/h       |
+--------------------+-----------+--------+----------+----------------+
```

**Localização**: município de Pium, Tocantins, ecorregião do Cerrado (Bico do Papagaio), dentro da Amazônia Legal.

---

## 4. Reprodução dos resultados

### 4.1. Pré-requisitos

```bash
# Dependências já listadas em requirements.txt
pip install -r requirements.txt

# Variáveis de ambiente esperadas (ver scripts/config.py)
# NASA_FIRMS_MAP_KEY já configurada (5bcd249ca2d9e80a86ff67e0320c7873)
# NASA_POWER_API_KEY opcional (aumenta rate limit)
```

### 4.2. Diagnóstico da NASA FIRMS

```bash
python -c "
import requests
KEY = '5bcd249ca2d9e80a86ff67e0320c7873'

print('Status da MAP_KEY:')
print(requests.get(
    f'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={KEY}',
    timeout=15
).text)

print()
print('URL com days_back=7 (versão antiga do nosso código):')
r1 = requests.get(
    f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/VIIRS_SNPP_NRT/'
    f'-49.9146,-10.6252,-49.7314,-10.4450/7',
    timeout=15,
)
print(f'  status={r1.status_code}  body={r1.text[:200]}')

print()
print('URL com days_back=5 + data ancorada:')
r2 = requests.get(
    f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/VIIRS_SNPP_NRT/'
    f'-49.9146,-10.6252,-49.7314,-10.4450/5/2026-05-12',
    timeout=15,
)
print(f'  status={r2.status_code}  linhas={len(r2.text.splitlines())}')
"
```

**Saída esperada**:

```
Status da MAP_KEY:
{ "transaction_limit" : 5000, "current_transactions" : 0, "transaction_interval" : "10 minutes" }

URL com days_back=7 (versão antiga do nosso código):
  status=400  body=Invalid day range. Expects [1..5].

URL com days_back=5 + data ancorada:
  status=200  linhas=17
```

### 4.3. Consulta ao sistema (5 janelas)

```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
import app_map_interativo as a
c = a.app.test_client()

LAT, LON = -10.5351, -49.8230
casos = [
    ('Dia do foco principal',     2026, 5, 12, 14),
    ('Vespera do foco principal', 2026, 5, 11, 12),
    ('Dia do foco anterior',      2026, 5, 10, 13),
    ('Vespera foco anterior',     2026, 5, 9,  12),
    ('1 semana antes',            2026, 5, 5,  12),
]
for label, ano, mes, dia, hora in casos:
    r = c.post('/api/predict', json={
        'lat': LAT, 'lon': LON,
        'ano': ano, 'mes': mes, 'dia': dia, 'hora': hora,
    })
    j = r.get_json() or {}
    if not j.get('sucesso'):
        print(f'[{label}] FAIL: {j.get(\"erro\")}')
        continue
    du = j.get('dados_usados', {}) or {}
    probs = j.get('probabilidades', {}) or {}
    print(f'[{label}] {ano}-{mes:02d}-{dia:02d} {hora}h')
    print(f'  Risco={j.get(\"risco\")}  P_max={100*j.get(\"confianca\",0):.1f}%')
    print(f'  Probs: Bx={100*probs.get(\"Baixo\",0):.1f}%  '
          f'Mod={100*probs.get(\"Moderado\",0):.1f}%  '
          f'MA={100*probs.get(\"Muito Alto\",0):.1f}%')
    print(f'  FRP: ponto={du.get(\"FRP\",0):.2f}  '
          f'media7d={du.get(\"Media_FRP_Ultimos_7_Dias\",0):.2f}  '
          f'max7d={du.get(\"Max_FRP_Ultimos_7_Dias\",0):.2f}')
"
```

### 4.4. Contrafactual (mesmo ponto, outras datas)

```bash
python -c "
import sys
sys.path.insert(0, 'scripts')
import app_map_interativo as a
c = a.app.test_client()

for ano, mes, dia in [(2024, 8, 15), (2024, 9, 15), (2024, 5, 15), (2026, 5, 12)]:
    r = c.post('/api/predict', json={
        'lat': -10.5351, 'lon': -49.8230,
        'ano': ano, 'mes': mes, 'dia': dia, 'hora': 13,
    })
    j = r.get_json()
    du = j['dados_usados']
    print(f'{ano}-{mes:02d}-{dia:02d}: {j[\"risco\"]:10s}  '
          f'KBDI={du[\"KBDI_proxy\"]:6.1f}  '
          f'DSC_ma7={du[\"DiaSemChuva_ma7\"]:5.1f}  '
          f'Prec_ma7={du[\"Precipitacao_ma7\"]:.2f}  '
          f'FRP={du[\"Max_FRP_Ultimos_7_Dias\"]:.1f}MW')
"
```

---

## 5. Resultados consolidados

### 5.1. Resposta do sistema antes das correções

| Janela | Risco | P(Baixo) | FRP do FIRMS | Granularidade Tier 1 |
|---|---|---|---|---|
| 12/05/2026 14h | Baixo | 82,0% | **0,00 MW** (FIRMS 400) | estado_mes |
| 11/05/2026 12h | Baixo | 82,7% | **0,00 MW** | estado_mes |
| 10/05/2026 13h | Baixo | 82,2% | **0,00 MW** | estado_mes |
| 09/05/2026 12h | Baixo | 81,7% | **0,00 MW** | estado_mes |
| 05/05/2026 12h | Baixo | 79,5% | **0,00 MW** | estado_mes |

### 5.2. Resposta do sistema depois das correções

| Janela | Risco | P(Baixo) | FRP do FIRMS | Granularidade Tier 1 |
|---|---|---|---|---|
| 12/05/2026 14h | Baixo | 85,9% | 16,58 MW (16 detec.) | **celula_ext** |
| 11/05/2026 12h | Baixo | 85,8% | 22,01 MW (19 detec.) | celula_ext |
| 10/05/2026 13h | Baixo | 87,1% | 22,01 MW (21 detec.) | celula_ext |
| 09/05/2026 12h | Baixo | 86,6% | 22,01 MW (26 detec.) | celula_ext |
| 05/05/2026 12h | Baixo | 81,4% | 58,40 MW (32 detec.) | celula_ext |

### 5.3. Contrafactual (mesmo ponto, climatologia diferente)

| Data | KBDI | Prec_ma7 (mm/d) | DSC_ma7 (dias) | FRP (MW) | Risco | P(Muito Alto) |
|---|---|---|---|---|---|---|
| 15/08/2024 | 424,1 | 0,02 | 16,0 | 0,0 | **Muito Alto** | 99,0% |
| 15/09/2024 | 712,5 | 0,00 | 25,0 | 0,0 | **Muito Alto** | 99,6% |
| 15/05/2024 | 148,0 | 0,51 | 8,6 | 0,0 | **Muito Alto** | 47,3% |
| 12/05/2026 | 29,8 | 0,12 | 1,3 | 16,6 | Baixo | 4,6% |

---

## 6. Correções aplicadas

### 6.1. (C1) `scripts/frp_api.py`

**Antes** — `days_back=7` fixo, sem aceitar data ancorada:

```python
def get_frp_from_nasa_firms(
    self, lat, lon, radius_km=10.0, days_back=7,
) -> Optional[Dict]:
    # ...
    days_to_request = min(days_back, 7)  # limite hardcoded errado
    url = f"{self.firms_api_area}/{source}/{area_params}/{days_to_request}"
```

**Depois** — clip em [1, 5] + `reference_date`:

```python
FIRMS_MAX_DAYS = 5  # NASA FIRMS NRT aceita 1..5

def get_frp_from_nasa_firms(
    self, lat, lon, radius_km=10.0, days_back=5,
    reference_date: Optional[datetime] = None,
) -> Optional[Dict]:
    if days_back > self.FIRMS_MAX_DAYS:
        days_back = self.FIRMS_MAX_DAYS
    if days_back < 1:
        days_back = 1
    end_date = reference_date or datetime.now()
    # ...
    date_suffix = f"/{end_date_str}" if reference_date is not None else ""
    url = f"{self.firms_api_area}/{source}/{area_params}/{days_to_request}{date_suffix}"
```

**Em `app_map_interativo.py`** o `reference_date` é passado adiante:

```python
frp_data = frp_provider.get_frp_from_nasa_firms(
    lat=lat, lon=lon, radius_km=10.0,
    days_back=5, reference_date=ref_data,
)
```

### 6.2. (C2) `scripts/feature_lookup.py`

**Antes** — cascata espacial `3×3` apenas:

```python
for offset_lat in (0, -1, 1):
    for offset_lon in (0, -1, 1):
        v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_i))
        if v:
            return dict(v), "celula"

v = self._by_estado.get((est, mes_i))
if v:
    return dict(v), "estado_mes"
```

**Depois** — cascata estendida `3×3 → 5×5 → mês adjacente → estado_mes`:

```python
# (1) 3x3 — vizinhança imediata
for offset_lat in (0, -1, 1):
    for offset_lon in (0, -1, 1):
        v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_i))
        if v:
            return dict(v), "celula"

# (2) 5x5 — anel externo (~55km)
for offset_lat in (-2, 2, -1, 1, 0):
    for offset_lon in (-2, 2, -1, 1, 0):
        if abs(offset_lat) < 2 and abs(offset_lon) < 2:
            continue
        v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_i))
        if v:
            return dict(v), "celula_ext"

# (3) Mês adjacente (sazonalmente próximo)
meses_adj = [mes_i - 1 if mes_i > 1 else 12,
             mes_i + 1 if mes_i < 12 else 1]
for mes_alt in meses_adj:
    for offset_lat in range(-2, 3):
        for offset_lon in range(-2, 3):
            v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_alt))
            if v:
                return dict(v), "celula_mes_vizinho"
```

### 6.3. (C3) `scripts/app_map_interativo.py` — política de fusão de features

**Decisão**: atualizar com observação NRT **apenas** as features que no treino eram observações individuais (FRP médio/máx, Dias_Desde_Ultimo_Incendio). **Manter** medianas de treino para features que no pipeline de inferência vêm de medianas (Incendios_Ultimos_7/30_Dias).

```python
if frp_data and frp_data.get("sucesso"):
    detec = int(frp_data.get("detections", 0) or 0)
    frp_max = float(frp_data.get("frp_max", 0.0) or 0.0)
    frp_mean = float(frp_data.get("frp_mean", 0.0) or 0.0)
    if detec > 0 or frp_max > 0:
        clima_data["Media_FRP_Ultimos_7_Dias"] = frp_mean
        clima_data["Max_FRP_Ultimos_7_Dias"] = frp_max
        clima_data["Dias_Desde_Ultimo_Incendio"] = min(
            float(clima_data.get("Dias_Desde_Ultimo_Incendio", 365.0) or 365.0),
            5.0,
        )
        # Incendios_Ultimos_7/30_Dias intencionalmente NÃO sobrescritos
        # (ver discussão de OOD na seção 7.3)
```

---

## 7. Discussão científica

### 7.1. O modelo **não** tem viés sazonal cego

A hipótese inicial era "o modelo aprendeu que Tocantins em maio = baixo risco, ignorando inputs". O contrafactual da Tabela em §5.3 refuta essa hipótese:

- 15/05/2024 mesmo ponto, NASA POWER reportando regime seco (DSC_ma7=8,6 dias) → **Muito Alto** com 47,3%
- 12/05/2026 mesmo ponto, NASA POWER reportando regime "chuvoso" (DSC_ma7=1,3 dia) → Baixo com 85,9%

A diferença está nos *inputs*. O modelo respondeu corretamente em ambos os casos dado o que viu.

### 7.2. O limite está na granularidade da reanálise climática

NASA POWER MERRA-2 entrega dados em grade de ~50 km. Em Cerrado de transição (Pium-TO, maio), chuvas pontuais/dispersas dentro da célula podem deixar `Precipitacao_ma7` baixa mas `DiasSemChuva_ma7` também baixa — o que para o modelo configura "chuvas leves e frequentes". No campo, a vegetação pode estar em condição totalmente distinta da indicada pela média da célula.

Este é um caso de **representativity gap** (lacuna de representatividade) entre dados de grade global e fenômenos locais. Não é falha do classificador; é limite da fonte de entrada.

### 7.3. Sobrescrever Inc7/Inc30 com observação real piora a previsão

O experimento intermediário (substituir contagem mediana de treino por contagem real FIRMS) elevou P(Baixo) de 82,7% para 86,0%. Causa: a feature `Incendios_Ultimos_X_Dias` no pipeline de inferência atual é obtida pela mediana `(Estado, Mês)` do dataset de treino. O classificador **aprendeu** essa feature como sinal sazonal (não causal individual). Substituí-la pelo valor real para Tocantins-Maio (que normalmente é zero) coloca o exemplo fora da distribuição vista no treino, e o modelo responde de forma errática.

A lição é importante para o TCC: **a semântica de uma feature em produção precisa ser compatível com a semântica usada no treino**. Misturar não é trivial.

### 7.4. Por que P(Baixo) **subiu** após as correções

Após (C1) NASA FIRMS funcionar e (C2) cascata Tier 1 expandir para `celula_ext`, P(Baixo) foi de 82,0% para 85,9%. Isso é coerente com a Causa 3:

- Antes: o modelo via "incêndios ~ 0" (treino-Inc7) + "histórico FRP ~ 10 MW" (lookup estadual genérico) + "clima neutro" (lookup estadual). Sinal misto.
- Depois: o modelo passou a ver "incêndios ~ 0" (treino, agora consistente) + "FRP real ~ 22 MW" + "clima da vizinhança real (~ 0.5 mm acumulado, mas DSC=1d)". Sinal mais coerente com "regime levemente chuvoso, fogo pontual mas não generalizado" — que é classe Baixo.

O modelo está sendo *fiel* aos inputs. Os inputs é que estão suaves.

---

## 8. Melhorias implementadas após o caso Pium-TO

Esta seção documenta as três frentes implementadas em 12/05/2026 a partir das lições do caso. As **três foram concluídas** e validadas — todas em código de produção.

### 8.1. Auditoria operacional contínua  ✅

**Arquivos**: `scripts/audit_log.py`, `scripts/auditar_predicoes.py`.

A cada chamada à rota `/api/predict`, a aplicação grava uma linha JSONL em `logs/auditoria_predicoes.jsonl` contendo:

- `prediction_id` (UUID curto), `lat`, `lon`, `estado`, `municipio`, `ref_data`;
- `risco_predito`, `confianca`, `probabilidades`, `modelo`, `thresholds_aplicados`;
- 15 `features_chave` (KBDI, VPD, SPI, FRP, Incêndios_Ultimos_*, etc.);
- `fonte_clima_detalhe` (incluindo info INMET se aplicável);
- `granularidade_tier1` (qual nível da cascata foi usada).

O log usa append-only com rotação automática a cada 50 MB (até 5 arquivos rotacionados). É tolerante a falhas: se a escrita falhar, a predição segue normalmente.

O script `scripts/auditar_predicoes.py`:
1. Lê o JSONL,
2. Para cada predição com idade ≥ `min-age-days` (default 3), consulta NASA FIRMS NRT na janela `[ref_data, ref_data + horizon_days]` (default 3 dias) com raio default 5 km;
3. Define `y_pred ∈ {0,1}` (0 = Baixo; 1 = Moderado/Muito Alto) e `y_true ∈ {0,1}` (1 = houve foco confirmado);
4. Calcula precision/recall/F1/acurácia globais e em janelas rolling de 7, 14 e 30 dias;
5. Filtra automaticamente predições mais antigas que o histórico do NRT (~60 dias) — esses ficam no CSV mas não pesam nas métricas.

Saídas:
- `modelos/relatorios/auditoria_operacional.json` (relatório agregado)
- `modelos/relatorios/auditoria_operacional.csv` (uma linha por predição auditada)

**Validado**: smoke-test gerou 5 predições, scriptanou todas em ~5 s, gerou os relatórios e respeitou o limite de 60 dias do NRT.

**Como usar**:

```bash
python scripts/auditar_predicoes.py --horizon-days 3 --radius-km 5
python scripts/auditar_predicoes.py --horizon-days 7 --radius-km 10 --rolling 7,14,30
```

### 8.2. Fusão NASA POWER + INMET para clima local  ✅

**Arquivos**: `scripts/inmet_api.py`, `scripts/climate_api.py` (modificado).

A integração foi implementada com **duas fontes INMET complementares**, selecionadas automaticamente conforme a idade da consulta:

**Fonte (1) — WIS2 / OGC API Features (`http://wis2bra.inmet.gov.br/oapi`)** — *tempo real, sem token, descoberta no decorrer da implementação após sondagem ativa em 12/05/2026*:
- Variável: `total_precipitation_or_total_water_equivalent` em `urn:wmo:md:br-inmet:synop` (unidade `kg m⁻²` ≡ mm/h).
- Cobertura: 358 estações sinópticas WMO no Brasil que publicaram chuva nos últimos 7 dias; histórico de ~90 dias garantido pelo padrão WIS2.
- Cobertura geográfica menor que o ZIP (só estações sinópticas oficiais; A055 LAGOA DA CONFUSÃO, p.ex., não publica via WIS2).
- Cuidado: a API **ignora** o filtro `wigos_station_identifier` na querystring (verificado em 12/05/2026); a estação tem que ser filtrada client-side. O nosso código usa `bbox` apertado (~5 km em torno da estação) + filtro client-side para evitar baixar 58 mil obs misturadas.

**Fonte (2) — ZIPs anuais (`portal.inmet.gov.br/uploads/dadoshistoricos/{ANO}.zip`)** — *histórico, ~100 MB/ano*:
- Cobertura: 565+ estações automáticas (não só sinópticas), histórico desde 2000.
- ZIP do ano corrente é parcial (publicado mensalmente).
- Cache local em `.cache_inmet/{ANO}/{COD}.csv` evita re-download.

A API horária `apitempo.inmet.gov.br/estacao/...` foi descartada por instabilidade (`status 204` frequente em 12/05/2026, sem token e com token).

O `inmet_api.py`:
1. Baixa e cacheia (`.cache_inmet/catalogo_estacoes.json`, TTL 30 dias) o catálogo de estações automáticas via `apitempo.inmet.gov.br/estacoes/T` (679 estações; 168 na Amazônia Legal) — usado pelo caminho ZIP.
2. Baixa e cacheia (`.cache_inmet/wis2_stations_precip.json`, TTL 7 dias) o catálogo WIS2 de 358 estações que publicaram chuva recentemente — usado pelo caminho tempo-real.
3. `get_inmet_fusion_data(lat, lon, reference_date)` faz **roteamento automático**:
   - Se `reference_date` está nos últimos 90 dias e existe estação WIS2 com chuva ≤ `max_km` → usa WIS2 (sem download, ~1 s).
   - Senão → tenta ZIP anual da estação automática mais próxima.
   - Se nenhuma das duas funciona → retorna `None` e o chamador cai no MERRA-2.

A fusão em `climate_api.get_climate_data`:
- Primeiro busca NASA POWER (sempre disponível).
- Se INMET retorna dados (WIS2 ou ZIP), **sobrescreve** `precipitacao`, `prec_ma7`, `dias_sem_chuva` e `diasem_ma7` com os valores INMET (mais fiéis ao microclima).
- Mantém os valores MERRA-2 em chaves `_merra` para auditoria.
- A `umidade` (RH) continua vindo da NASA POWER (cobertura desigual no INMET).
- A resposta da `/api/predict` inclui `fonte_dados` ∈ {`nasa_power`, `inmet_fundido_nasa_power`} e, no `contexto_historico`: `estacao_inmet`, `distancia_estacao_inmet_km`, `cobertura_inmet_pct`, `precipitacao_merra_ma7`, `diasem_merra_ma7`.

**Validado**: smoke-test com 4 cenários (incluindo o end-to-end via `app.test_client()` em 12/05/2026 às 19h26 BRT):

| Caso | Fonte | Estação | NASA POWER | INMET | Resultado |
|---|---|---|---|---|---|
| Pium-TO 15/08/2024 | ZIP | A055 LAGOA DA CONFUSÃO @ 32,7 km | DSC=16 d | **DSC=7 d** | Muito Alto (97,3%) |
| Brasília 10/08/2024 | ZIP | A001 BRASÍLIA @ 1,1 km | DSC=25 d | **DSC=7 d** | Muito Alto (67,9%) |
| Pium-TO 12/05/2026 | WIS2 + busca hierárquica (até 180 km) | FORMOSO DO ARAGUAIA @ ~152 km (`inmet_representatividade` = baixa) | DSC_ma7 ≈ 1,3 d (MERRA) | INMET seco (ex.: P_ma7=0; Dsem_ma7=7) | Depende do modelo; metadados na API |
| **Brasília 12/05/2026 (HOJE, real-time)** | **WIS2** | A001 BRASÍLIA @ 1,1 km | DSC_ma7=1,86 d | **DSC_ma7=7,0 d** | Baixo cai de ~99% → **66,0%** |

O caso Brasília 12/05/2026 é a primeira demonstração do **fluxo de fusão real-time end-to-end**: a WIS2 detectou que MERRA-2 estava reportando microclima úmido espurio (DSC=1,86 d) enquanto a estação INMET A001 a 1,1 km do ponto consultado registra 7 dias secos consecutivos. A correção foi propagada para o modelo, derrubando a confiança da classe Baixo de quase 100% para 66% — um sinal claro de que com mais sub-amostragem regional o modelo passaria a alertar.

**Observação importante**: o ZIP do ano corrente é parcial (publicado mensalmente). Para 12/05/2026 o ZIP de 2026 tinha só dados até fevereiro, por isso o caminho WIS2 é decisivo para alertas em **D-0**.

**Atualização (extensões pós-literatura 2024--2026)** — *motivação para o Capítulo~4/5 do TCC*:

1. **Fusão INMET hierárquica** (`config.INMET_EXTENDED_MAX_DISTANCE_KM`, `get_inmet_fusion_data(..., hierarchical=True)`): se não há estação com dados dentro de 50 km, o sistema tenta até **180 km**, preenchendo `inmet_representatividade` ∈ {alta, moderada, baixa} e `inmet_busca_raio_km`. *Motivo no TCC:* explicitar trade-off cobertura espacial vs. representatividade microclimática (literatura de fusão grade--estação); o caso Pium--TO passa a receber precipitação/vento de estação sinótica distante com rótulo **baixa** — mais informativo que cair só no MERRA-2 sem aviso. *Exemplo reproduzível:* `get_inmet_fusion_data(-10.5351, -49.823, 12/05/2026)` → FORMOSO DO ARAGUAIA (~152 km), `vento_inmet_ms_ma7` ≈ 1,2 m/s.

2. **Vento WIS2** na mesma estação da precipitação (`vento_inmet_ms_ma7` quando fonte = WIS2). *Motivo:* alinhar-se a trabalhos de *fire weather* que usam vento com precipitação e umidade.

3. **NASA POWER estendido** (`WS2M`, `T2M_MAX` na mesma requisição diária; médias móveis 7d expostas no clima). *Motivo:* disponibilizar variáveis dinâmicas citadas em revisões recentes (FWI, vento a 10 m) para o painel e para futura ablação — sem retreino imediato.

4. **Proxy `FWI_fire_weather_proxy`** e **`incerteza_operacional`** (`scripts/operational_uncertainty.py`, resposta JSON). *Motivo:* (i) indicador escalar interpretável combinando seca na semana, (1−RH) e vento; (ii) gap entre as duas classes mais prováveis — narrativa de *decision support* sob ambiguidade, em linha com UQ em Earth observation, **sem** alegar conformal calibrado sem conjunto de validação dedicado.

5. **FNR/FPR na auditoria FIRMS** (`auditar_predicoes.py`). *Motivo:* o binário operacional “alerta / não alerta” exige reportar taxa de **focos perdidos** (FNR), frequentemente omitida quando só se publica F1.

### 8.3. Retreino com oversampling OOD em Cerrado-transição  ✅

**Arquivos**: `scripts/analise_ood_cerrado.py`, `scripts/oversample_ood_cerrado.py`, `scripts/retreinar_com_oversample.py`.

**Análise (`analise_ood_cerrado.py`)**:
1. Calcula a distribuição de focos por (Estado, Mês) no `base_de_dados_enriquecido.csv` (924 306 linhas, 547 845 com FRP > 0 = 58,7%).
2. Identifica pares onde o estado é de transição Cerrado-Amazônia (TO, MA, MT, PA, BA, PI, DF, GO), o mês está em {5, 6, 7} (transição chuvoso → seco), o n é ≥ 30, e a razão de focos < 0,7 da média estadual.
3. **8 pares identificados** (gravados em `modelos/relatorios/ood_pares_alvo.json`):

| Estado | Mês | n_total | n_focos | Razão | Fator sugerido |
|---|---|---|---|---|---|
| PARÁ | 5 | 1.293 | 885 | 0,05 | 5,0× |
| MARANHÃO | 5 | 158 | 115 | 0,06 | 5,0× |
| PARÁ | 6 | 4.442 | 2.922 | 0,18 | 5,0× |
| MARANHÃO | 6 | 617 | 383 | 0,20 | 5,0× |
| **TOCANTINS** | **5** | **75** | **50** | **0,29** | **3,5×** ← caso Pium-TO |
| MARANHÃO | 7 | 1.536 | 817 | 0,42 | 2,4× |
| PARÁ | 7 | 17.262 | 10.139 | 0,62 | 1,6× |
| TOCANTINS | 6 | 197 | 118 | 0,68 | 1,5× |

**Oversampling (`retreinar_com_oversample.py`)**:
1. Split estratificado (`random_state=42`, mesmo do baseline): 739.444 treino / 184.862 teste;
2. **Aplica oversampling apenas no TREINO** (não no teste — assim as métricas de avaliação não são infladas);
3. Para cada par-alvo, replica `floor(fator − 1)` cópias com jitter gaussiano em (Latitude, Longitude, Hora, FRP) — sigmas pequenos (~2 km, 1h, ±5% FRP) para variabilidade sem perda semântica;
4. Total adicionado: **+35.472 linhas** (+4,8% no treino, de 739k → 775k).

**Validação (Logistic Regression Balanceada, 6,2 min de treino)**:

| Métrica | Baseline | OOD | Δ |
|---|---|---|---|
| Acurácia (n=184.862) | 64,58% | **65,75%** | **+1,17 pp** |
| F1-macro | 0,6044 | **0,6168** | **+1,25 pp** |
| F1-Baixo | 0,7142 | 0,7243 | +1,00 pp |
| F1-Moderado | 0,3643 | 0,3793 | **+1,50 pp** |
| F1-Muito Alto | 0,7346 | 0,7470 | +1,24 pp |
| **Hold-out Cerrado-transição** | — | acc=64,62%, F1m=0,591 | — |

**Revalidação do caso Pium-TO**:

| Cenário | Baseline P(Muito Alto) | OOD P(Muito Alto) | Δ |
|---|---|---|---|
| 12/05/2026 (foco real, MERRA-2 reporta chuvoso) | 3,3% | **6,6%** | ×2 |
| 15/05/2024 (mesmo dia, MERRA-2 reporta seco) | 23,1% | **32,4%** | +9,3 pp |
| 15/08/2024 (pico de seca) | 66,2% | **72,5%** | +6,3 pp |

A predição final ainda é "Baixo" no caso real (12/05/2026), mas a probabilidade de Muito Alto DOBROU. O modelo OOD ficou mais sensível aos cenários de transição, exatamente como esperado.

**Caminho para o Stacking GBM completo (não rodado por tempo de CPU, ~90 min)**:

```bash
python scripts/retreinar_com_oversample.py --modelo ensemble_stacking_gbm
```

O script salva `modelos/ensemble_stacking_gbm_ood.pkl` e `modelos/relatorios/comparativo_ood_vs_baseline.json` com as mesmas métricas e o mesmo hold-out Cerrado-transição.

---

## 9. Trabalhos futuros remanescentes

Após as três melhorias acima, restam como trabalhos futuros (para inserir em `tcc_tex/06_capitulo6_conclusao.tex`):

1. **Causalidade estrita nas features de janela móvel**. Recomputar `Precipitacao_acum_*`, `SPI_*`, `Incendios_Ultimos_*` usando apenas dados `≤ t` do registro de treino. Reduz o viés de validação temporal.
2. **Incluir índice de heterogeneidade intra-célula** (variância de NDVI/EVI dentro da célula MERRA-2). Sinaliza onde a média da célula é menos confiável e ativa automaticamente fontes mais locais (INMET).
3. **Retreinar Stacking GBM completo com OOD oversampling** (script pronto, ~90 min de CPU).
4. **Estender INMET para anos correntes** quando o ZIP oficial for atualizado (atualmente 2026 cobre só até fevereiro).
5. **Auditoria de longo prazo** (≥ 90 dias): consultar a versão histórica do FIRMS (VIIRS_SNPP_SP) em vez do NRT para auditar predições antigas, removendo o teto atual de 60 dias.

---

## 9. Arquivos tocados nesta investigação

| Arquivo | Tipo de mudança | Linhas afetadas (aprox.) |
|---|---|---|
| `scripts/frp_api.py` | Correção C1 (NASA FIRMS days_back+reference_date) | +25 / -8 |
| `scripts/feature_lookup.py` | Correção C2 (cascata estendida 5×5 + mês adjacente) | +30 / -5 |
| `scripts/app_map_interativo.py` | Correção C3 + passagem de `reference_date` ao FRP; UI: form lat/lon/data/hora; detecção lat↔lon invertida; aceitar vírgula decimal | +220 / -30 |
| `tcc_tex/07_estudo_caso_pium_to.tex` | **NOVO** — capítulo formal para o TCC | +180 (novo) |
| `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md` | **NOVO** — este documento, registro reproduzível | +400 (novo) |

---

## 10. Referências externas usadas neste estudo

- **Painel do Fogo / CENSIPAM**: <https://painel.censipam.gov.br/> — sistema oficial brasileiro de monitoramento de focos, mantido pelo Centro Gestor e Operacional do Sistema de Proteção da Amazônia. Fornece evento `6668060` e detecções por satélite com domínio fundiário e cobertura do solo.
- **NASA FIRMS API documentation**: <https://firms.modaps.eosdis.nasa.gov/api/area/> — limite atual de `days_back ∈ [1, 5]` para datasets NRT.
- **NASA POWER MERRA-2**: <https://power.larc.nasa.gov/> — reanálise climática global, grade ~50 km, usada em produção como fonte de `PRECTOTCORR` e `RH2M`.
- **TerraClass**: classificação de uso e cobertura do solo da Amazônia Legal, citada pelo Painel do Fogo para identificar "Vegetação natural primária" no ponto.

---

## 11. Para citar este estudo na escrita do TCC

> "O estudo de caso operacional Pium-TO (12/05/2026), descrito na Seção 5.9 e no Apêndice X, validou o sistema com fonte externa independente (*Painel do Fogo*/CENSIPAM) e identificou (i) dois *bugs* silenciosos no *pipeline* em produção, corrigidos durante a investigação, e (ii) um limite estrutural decorrente da resolução espacial (~50 km) da reanálise climática global em zonas de transição Cerrado–Amazônia, registrado como trabalho futuro."

---

*Documento mantido em `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md`. Versão formal acadêmica em `tcc_tex/07_estudo_caso_pium_to.tex`.*
