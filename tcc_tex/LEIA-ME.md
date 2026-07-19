# Pasta `tcc_tex/` — Conteúdo pronto para inserção no `.tex` do TCC

> **Atualizado em:** 12/05/2026
> **Objetivo:** entregar todo o material textual, bibliográfico e gráfico em formato perfeito para colagem direta no arquivo LaTeX do TCC original (`Projeto_de_graduação_II_do_curso_de_ciência_da_computação.pdf` — fevereiro/2024), sem que o aluno precise refazer formatação.

---

## 1. O que está nesta pasta

| Arquivo | Conteúdo | Onde inserir no TCC |
|---|---|---|
| `references.bib` | 42 entradas BibTeX consolidadas (modelos, otimização, interpretabilidade, domínio, infraestrutura) | Pasta do projeto `.tex` (substitui/complementa o `.bib` existente) |
| `00_resumo_abstract.tex` | Resumo (PT) e Abstract (EN) atualizados em ABNT | Substituir `\begin{resumo}...\end{resumo}` e `\begin{abstract}...\end{abstract}` |
| `01_capitulo1_complementos.tex` | (A) parágrafo de âncora científica para Justificativa; (B) nova redação dos 6 objetivos específicos; (C) nova redação da estrutura da monografia | Capítulo 1 (Introdução) |
| `02_capitulo2_estado_arte_complemento.tex` | Nova subseção "Literatura recente (2024--2025)" | Final do Capítulo 2 (Estado da Arte), antes do parágrafo de fechamento |
| `03_capitulo3_subsecoes_novas.tex` | 10 novas subseções: XGBoost, LightGBM, CatBoost, Ensembles, Optuna, Interpretabilidade, Desbalanceamento, Calibração, Índices físico-climáticos, Validação temporal | Capítulo 3 (Fundamentação Teórica) |
| `04_capitulo4_metodologia.tex` | (A) APIs externas; (B) bibliotecas; (C) Tier 1 e Calibração; (D) Treinamento (reescrita); (E) Avaliação (expansão); (F) Aplicação web; (G) Pseudo-labeling | Capítulo 4 (Metodologia) |
| `05_capitulo5_resultados.tex` | **Capítulo 5 inteiro** (o original está vazio) — 9 seções, 11 tabelas, 6 figuras | Substituir todo o Capítulo 5 |
| `06_capitulo6_conclusao.tex` | **Capítulo 6 inteiro** — 8 parágrafos numerados de achados + Limitações + Trabalhos futuros + Considerações finais | Substituir todo o Capítulo 6 |
| `INSTRUCOES_INSERCAO.md` | Passo-a-passo detalhado para o `.tex` original | Use como guia operacional |

---

## 2. Como compilar

### 2.1 No arquivo principal `.tex` (preâmbulo)

**UFTeX:** não repita `babel` (a classe já usa `[english,brazil]`). `\usepackage[portuguese]{babel}` causa *Option clash for package babel*.

```latex
% Pacotes recomendados (se ainda não estiverem incluídos)
% \usepackage[brazil]{babel}   % omitir com uftex / classe que já carrega babel
\usepackage[utf8]{inputenc}
\usepackage{siunitx}
\sisetup{
  output-decimal-marker = {,},
  group-separator = {.},
  group-minimum-digits = 4
}
\usepackage{booktabs}      % \toprule, \midrule, \bottomrule nas tabelas
\usepackage{graphicx}      % para \includegraphics
\usepackage{mhchem}        % para \ce{CO2} no Resumo

% Bibliografia (ABNT)
\usepackage[alf,abnt-emphasize=bf]{abntex2cite}
% ... no fim do arquivo:
\bibliography{references}
```

### 2.2 Sequência de compilação

```bash
pdflatex tcc.tex
bibtex tcc            # processa references.bib
pdflatex tcc.tex
pdflatex tcc.tex
```

---

## 3. Figuras geradas (alta resolução, 300 DPI)

Todas em `modelos/relatorios/`:

| Arquivo | Citado em |
|---|---|
| `distribuicao_classes.png` | Cap. 5 §5.1, Figura `fig:res_distribuicao` |
| `matriz_confusao_stacking_gbm.png` | Cap. 5 §5.3, Figura `fig:res_matriz_confusao` |
| `shap_per_classe_barras.png` | Cap. 5 §5.4, Figura `fig:res_shap_per_classe` |
| `permutation_importance_por_classe.png` | Cap. 5 §5.4, Figura `fig:res_perm_importance` |
| `validacao_temporal_comparativo.png` | Cap. 5 §5.5, Figura `fig:res_temporal_barras` |
| `evolucao_modelos.png` (já existia) | Cap. 5 §5.7, Figura `fig:res_evolucao` |
| `threshold_precision_recall.png` (já existia) | Cap. 5 §5.3, Figura `fig:res_calibracao` |
| `shap_feature_importance.png` (já existia) | usar como apoio em §5.4 (importância global Gini) |

### 3.1 Regerar as figuras

```bash
python scripts/gerar_figuras_tcc.py
```

Reprodutível: 100% dos JSON-fonte em `modelos/relatorios/*.json` são lidos pelo script.

---

## 4. Compatibilidade com o TCC original

- O original está em ABNT (provavelmente `abntex2`), com pacote `abntex2cite`. As entradas BibTeX deste arquivo são compatíveis.
- Chaves duplicadas no `.bib` (por exemplo, colar o arquivo inteiro duas vezes no Overleaf) quebram o BibTeX. O `references.bib` do repositório mantém **uma** entrada por chave; a conclusão usa `\cite{INPE24}` para o BDQueimadas (equivalente ao antigo `inpequeimadas`). A chave `wmo2021` permanece para o resumo.
- Substituições no texto original também devem trocar formatos `(Autor, ano)` por `\cite{chave_bibtex}` — ver `INSTRUCOES_INSERCAO.md` para o mapeamento completo.

---

## 5. Verificação rápida pré-entrega

```bash
# Conta entradas BibTeX (número varia conforme inclusões; evite duplicar chaves)
grep -c "^@" tcc_tex/references.bib

# Verifica que as figuras existem
ls -la modelos/relatorios/*.png

# Compila um documento de teste mínimo
cd tcc_tex
cat > _teste.tex << 'EOF'
\documentclass[a4paper,12pt]{article}
\usepackage[brazil]{babel}
\usepackage[utf8]{inputenc}
\usepackage{siunitx}
\usepackage{booktabs}
\usepackage{graphicx}
\sisetup{output-decimal-marker={,}, group-separator={.}}
\begin{document}
\input{05_capitulo5_resultados.tex}
\bibliographystyle{plain}
\bibliography{references}
\end{document}
EOF
pdflatex _teste.tex
bibtex _teste
pdflatex _teste.tex
pdflatex _teste.tex
# Confirma que _teste.pdf compilou sem erros e que as citações estão resolvidas.
```

---

## 6. Suporte

Esta entrega foi produzida em paralelo com a documentação científica em `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md`, `REFERENCIAS_TCC.md` e `MAPEAMENTO_TCC.md`. Os números reportados nos `.tex` deste pacote conferem com os artefatos JSON gerados pelos scripts em `scripts/` e validados na versão de 12/05/2026.
