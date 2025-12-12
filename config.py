# Fichier: config.py
DEFAULT_APPORT = 100000
DEFAULT_TAUX = {15: 3.2, 20: 3.4, 25: 3.6}
TAUX_PRELEVEMENTS_SOCIAUX = 17.2
TAUX_IR_PLUS_VALUE = 19.0

TAXONOMY_THRESHOLDS = {
    # “Optimisé” signals
    "min_yield_pct": 6.0,          # rendement net H (ou CoC) jugé “élevé”
    "min_cf_month": 150.0,         # CF net mensuel H jugé “confortable”
    "min_share_optimized_modes": 0.35,  # part (0–1) de biens Colocation/LCD pour “Optimisé”

    # “Patrimonial” signals
    "min_qual_score": 0.65,        # score qualitatif moyen (0–1)
    "max_yield_for_patrimonial": 7.0,   # Patrimonial reste modéré, pas “très haut rendement”
}
OPTIMIZED_MODES = {"coloc", "lcd", "lcd_pro"}  # codes déjà présents dans les archetypes
# === Taxonomy badge mapping (icon + tooltip) ===
TAXO_ICON = {"Optimisé": "⚡️", "Patrimonial": "🏛️", "Mix": "🔀"}
TAXO_TIP  = {
    "Optimisé": "Rendement/CF élevés ou part importante de Colocation/LCD",
    "Patrimonial": "Qualitatif fort avec rendement modéré",
    "Mix": "Équilibre entre rendement et qualité",
}