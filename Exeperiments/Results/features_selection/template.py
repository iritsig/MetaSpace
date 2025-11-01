from pathlib import Path
from string import Template

# paramètres
scenario = "s_s_bis" ## put here the scenario you want for the feature selection
models = ['albert','bart','bert','distilbert','roberta','minilm']
base_root = Path("/Users/nourkired/Volumes/My Passport for Mac/feature_selection/") / scenario 

# --- template brut (avec $model, $base_path, $scenario) ---
TEMPLATE = Template(r'''# -*- coding: utf-8 -*-
"""
Script auto-généré pour le modèle: $model
Analyse R sur le fichier filtré (df_without_correlated_features.csv)
"""
from pathlib import Path
import warnings
import pandas as pd
import sys

try:
    import rpy2.robjects as robjects
    import rpy2.rinterface_lib.callbacks
    HAS_RPY2 = True
except Exception as e:
    HAS_RPY2 = False
    warnings.warn(f"rpy2 non disponible: {e}", RuntimeWarning)

warnings.filterwarnings("ignore")

BASE = Path("$base_path")
INPUT_CSV = BASE / "df_without_correlated_features.csv"
OUT_DIR   = BASE / "script"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_txt(path: Path, title: str, content: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\\n{'='*80}\\n")
        f.write(f"🔹 {title}\\n")
        f.write(f"{'='*80}\\n")
        f.write(content)
        f.write("\\n\\n")
        f.flush()

def run_r_analysis(input_csv: Path, out_dir: Path):
    out_txt = out_dir / "resultats_experimentation.txt"
    out_feats = out_dir / "metriques_sorties.txt"
    out_png = out_dir / "odds_ratios_plot.png"

    if not HAS_RPY2:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("rpy2 non disponible: analyse R non exécutée.\\n")
            f.flush()
        return

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("📄 **Résultats de l'Analyse**\\n")
        f.write("="*80 + "\\n")
        f.flush()

    rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: print(x, end="")
    df = pd.read_csv(input_csv).drop(columns=["ExecutionTime"])

    numeric_predictors = [c for c in df.columns if c != "true_match" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_predictors:
        write_txt(out_txt, "Alerte", "Aucun prédicteur numérique trouvé (hors true_match).")
        return

    predictors = "+".join(numeric_predictors)
    formula_full = f"true_match ~ {predictors}"

    in_r = str(input_csv).replace("\\\\", "/")
    robjects.r(f'df <- read.csv("{in_r}", sep=",")')
    robjects.r('suppressPackageStartupMessages({library(GGally); library(broom.helpers); library(ggplot2); library(car)})')
    robjects.r(f"formula_full <- as.formula('{formula_full}')")
    robjects.r('m.complet <- glm(formula_full, data = df, family = "binomial")')
    robjects.r('m.back <- step(m.complet, direction = "backward")')

    odds_ratios = robjects.r('capture.output(exp(cbind(OR = coef(m.back), confint(m.back))))')
    write_txt(out_txt, "Odds Ratios et Intervalles de Confiance", "\\n".join(odds_ratios))

    out_png_r = str(out_png).replace("\\\\", "/")
    robjects.r(f"""
        g <- ggcoef_model(m.back, exponentiate = TRUE)
        ggsave("{out_png_r}", plot = g, width = 10, height = 12, dpi = 300)
    """)

    selected_features = robjects.r('names(coef(m.back))[-1]')
    with open(out_feats, "w", encoding="utf-8") as f:
        f.write(str(list(selected_features)))
        f.flush()

def main():
    out_txt = OUT_DIR / "resultats_experimentation.txt"
    write_txt(out_txt, "DEBUG", "Je suis bien passé ici (avant l'analyse)")

    if not INPUT_CSV.exists():
        print(f"[err] Fichier introuvable: {INPUT_CSV}")
        sys.exit(1)

    run_r_analysis(INPUT_CSV, OUT_DIR)
    write_txt(out_txt, "DEBUG", "Analyse terminée ✅")

if __name__ == "__main__":
    main()
''')

# --- génération automatique ---
for m in models:
    base_path = base_root / m
    outdir = base_path / "script"
    outdir.mkdir(parents=True, exist_ok=True)

    code = TEMPLATE.substitute(model=m, scenario=scenario, base_path=str(base_path))

    outfile = outdir / "script_model.py"
    outfile.write_text(code, encoding="utf-8")
    print(f"Généré: {outfile}")
