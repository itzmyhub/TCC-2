"""Validação do RÓTULO contra área queimada independente (MapBiomas Fogo).

O rótulo do pipeline (fogo=1 se ≥1 foco INPE na célula-mês) vem de DETECÇÃO DE FOGO
ATIVO (focos), que tem omissão (nuvem, fogos pequenos, hora de passagem). O MapBiomas
Fogo mapeia **cicatriz de área queimada** (Landsat 30 m, deep learning) — produto
INDEPENDENTE e de modalidade diferente. Se ambos concordam, o rótulo é corroborado.

Acesso: o raster do MapBiomas exige Earth Engine/rasterio (indisponível aqui); usa-se
a estatística oficial **por estado/ano** (download estático, Coleção 3, 1985–2023):
``MB-Fogo-3-Biome-State.xlsx``. Validação AGREGADA: correlação entre
contagem de focos INPE e área queimada MapBiomas por (estado da Amazônia Legal, ano).

Limitação: granularidade estado×ano (não célula); a validação cel-a-cel exigiria o
raster. Ainda assim, concordância forte entre dois produtos independentes valida o
sinal que o rótulo captura.
"""
from __future__ import annotations

import glob
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / ".cache_umidade" / "MB-Fogo-3-Biome-State.xlsx"
URL = ("https://storage.googleapis.com/mapbiomas-public/brasil/fire/"
       "collection_3_stats/MB-Fogo-3-Biome-State.xlsx")
REL_DIR = ROOT / "modelos" / "relatorios"
ESTADOS_AL = ["ACRE", "AMAPA", "AMAZONAS", "MARANHAO", "MATO GROSSO", "PARA",
              "RONDONIA", "RORAIMA", "TOCANTINS"]
ANOS = list(range(2014, 2024))


def norm(s):
    s = str(s).strip().upper()
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def carregar_mapbiomas() -> pd.DataFrame:
    if not XLSX.exists():
        XLSX.write_bytes(requests.get(URL, timeout=180).content)
    df = pd.read_excel(XLSX, sheet_name="a_ANNUAL")
    df["uf"] = df["Estados"].map(norm)
    anos_cols = {c: int(float(c)) for c in df.columns if str(c).replace(".0", "").isdigit()}
    # soma área queimada (ha) por (estado, ano) — todos os biomas/classes
    linhas = []
    for uf in ESTADOS_AL:
        sub = df[df["uf"] == uf]
        for col, ano in anos_cols.items():
            if ano in ANOS:
                linhas.append({"uf": uf, "ano": ano,
                               "area_queimada_ha": float(pd.to_numeric(sub[col], errors="coerce").sum())})
    return pd.DataFrame(linhas)


def contar_focos() -> pd.DataFrame:
    parts = []
    for a in sorted(glob.glob(str(ROOT / "focos_qmd_inpe_*.csv"))):
        d = pd.read_csv(a, usecols=["DataHora", "Estado"])
        d["ano"] = pd.to_datetime(d["DataHora"], format="%Y/%m/%d %H:%M:%S", errors="coerce").dt.year
        d["uf"] = d["Estado"].map(norm)
        parts.append(d[["uf", "ano"]])
    f = pd.concat(parts, ignore_index=True)
    f = f[(f["uf"].isin(ESTADOS_AL)) & (f["ano"].isin(ANOS))]
    return f.groupby(["uf", "ano"]).size().rename("focos_inpe").reset_index()


def main():
    print("Carregando MapBiomas Fogo (área queimada estado/ano) e focos INPE...")
    mb = carregar_mapbiomas()
    fo = contar_focos()
    m = mb.merge(fo, on=["uf", "ano"], how="inner")
    m = m[(m["area_queimada_ha"] > 0) & (m["focos_inpe"] > 0)]
    print(f"  pontos (estado×ano): {len(m)} | estados: {m['uf'].nunique()} | anos: {m['ano'].min()}–{m['ano'].max()}")

    # Correlações (escala log p/ Pearson, dado heavy-tail; Spearman robusto a monotonia)
    la = np.log1p(m["area_queimada_ha"]); lf = np.log1p(m["focos_inpe"])
    pear = pearsonr(la, lf)
    spear = spearmanr(m["area_queimada_ha"], m["focos_inpe"])

    # Correlação interanual média DENTRO de cada estado (o padrão ano-a-ano bate?)
    por_estado = {}
    for uf, g in m.groupby("uf"):
        if len(g) >= 4 and g["focos_inpe"].nunique() > 1 and g["area_queimada_ha"].nunique() > 1:
            por_estado[uf] = float(spearmanr(g["area_queimada_ha"], g["focos_inpe"]).statistic)
    spear_intra = float(np.nanmean(list(por_estado.values())))

    res = {
        "fonte_independente": "MapBiomas Fogo Coleção 3 (área queimada, ha), estado×ano",
        "rotulo": "focos INPE (define fogo=1 na célula-mês)",
        "n_pontos": int(len(m)),
        "pearson_log": {"r": float(pear.statistic), "p": float(pear.pvalue)},
        "spearman_global": {"rho": float(spear.statistic), "p": float(spear.pvalue)},
        "spearman_interanual_intra_estado_media": spear_intra,
        "por_estado": por_estado,
    }
    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "validacao_rotulo_mapbiomas.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== Concordância focos INPE × área queimada MapBiomas ==")
    print(f"  Pearson(log)  r = {pear.statistic:.3f} (p={pear.pvalue:.2e})")
    print(f"  Spearman global rho = {spear.statistic:.3f} (p={spear.pvalue:.2e})")
    print(f"  Spearman interanual médio (dentro de cada estado) = {spear_intra:.3f}")
    print("  por estado (rho interanual):")
    for uf, r in sorted(por_estado.items(), key=lambda kv: -kv[1]):
        print(f"    {uf:12s} {r:+.3f}")
    print(f"\n  salvo em: {REL_DIR / 'validacao_rotulo_mapbiomas.json'}")


if __name__ == "__main__":
    main()
