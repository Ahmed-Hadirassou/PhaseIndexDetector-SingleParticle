# PhaseIndex — reproductibilité

Tout ce qu'il faut pour reproduire chaque tableau et chaque figure de
*PhaseIndex : un potentiel bistable appris pour la détection des crises de
marché* (septembre 2026), avec les fichiers de résultats bruts tels qu'ils
ont été produits.

## Organisation

```
reproducibility/
├── scripts/     22 scripts Python — à placer à côté des 8 fichiers du modèle
├── results/     13 fichiers JSON — sortie brute de chaque expérience, par pli et par graine
└── paper/       source LaTeX et les 17 figures
```

Les 8 fichiers du modèle lui-même (`main_v304_soft_labels.py`, `features.py`,
`velocity_features.py`, `evaluation_metrics.py`, `visualization.py`,
`config.py`, `spectral_features.py`, et `walk_forward.py`) sont à la racine
de ce dépôt. **Attention** : `scripts/walk_forward.py` est une version
*patchée* de celui de la racine (ajout de l'ECE, de l'écart entraînement/test
et du temps d'avance) ; c'est elle qu'il faut utiliser pour reproduire le
tableau 1.

## Correspondance papier → script → résultat

| Papier | Ce qui est calculé | Script | Résultat |
|---|---|---|---|
| Tableau 1 | marche en avant, 6 plis, 5 graines | `walk_forward.py` (patché) | `walk_forward.json` |
| Tableau 2 | VIX, GARCH, Markov, tendance MM200 | `run_baseline_comparison.py` | `baseline_comparison.json` |
| §6, fig. 1 | permutation, sensibilité, courbe d'apprentissage, ablation par groupe | `run_overfitting_suite.py` | `overfitting_suite.json` |
| §6, fig. 2–3 | balayage des époques, perte train/val | `run_early_stopping_test.py` | `early_stopping_folds_*.json` |
| §7.1, fig. 4 | ablation du terme de brisure de symétrie, 4 branches | `run_tilt_ablation.py` | `tilt_ablation_v2_horizon_physics1to5.json` + `tilt_physics.json` + `tilt_coupled.json` |
| §7.2, tab. 3, fig. 5 | ablation des mécanismes, 2×2 | `run_mechanism_ablation.py` | `mechanism_ablation.json` |
| §7.3, fig. 6 | chaîne de soustraction (encodeur linéaire, volatilité seule) | `run_minimal_model.py` | `minimal_model.json` |
| §8.1–8.3, fig. 7 | régime de friction, taux de Kramers, Δ_FDT | `run_physics_signals.py` | `physics_signals.json` |
| §8.4 | vacillement (variance glissante de Ψ) | `run_flickering_signals.py` | `flickering_signals.json` |
| §8.5, tab. 4, fig. 8 | ensemble de rangs final, 6 plis | `run_final_ensemble.py` | `final_ensemble.json` |

Modules importés par les scripts ci-dessus (pas à lancer directement) :
`tilt_ablation.py`, `mechanism_ablation.py`, `minimal_model.py`,
`physics_signals.py`, `flickering_signals.py`, `extra_metrics.py`,
`baseline_models.py`, `calibrated_metrics.py`, `overfitting_diagnostics.py`.

Les scripts `run_tilt_physics_resume.py`, `run_tilt_coupled_resume.py` et
`run_mechanism_ff.py` relancent une seule branche d'une ablation, avec
sauvegarde après chaque pli. Ils existent parce que les runs complets
(2 à 4 h) ont été interrompus par des déconnexions ; le fichier
`tilt_ablation_v2_horizon_physics1to5.json` a été reconstruit à la main
depuis la sortie console après un tel crash, et ses cinq premiers plis de
la branche *physics* ont été recalculés dans `tilt_physics.json`, qui
coïncide à la quatrième décimale.

## Lancer une expérience

```bash
# les 8 fichiers du modèle + le contenu de scripts/ dans le même dossier
python run_tilt_ablation.py          # ~3.5–4 h, 4 branches × 6 plis × 5 graines
python run_mechanism_ablation.py     # idem
python run_physics_signals.py        # ~1 h
python run_final_ensemble.py         # ~2 h
```

Chaque script écrit dans `results/<nom>.json` après chaque pli (ou chaque
branche pour les plus anciens). Les données de prix sont téléchargées par
`yfinance` au lancement.

## Ce qui n'est pas reproductible au bit près

L'entraînement est stochastique ; les cinq graines (42, 123, 456, 789, 1011)
sont fixées, mais des différences à la troisième décimale entre machines
sont attendues. C'est précisément pour cela que chaque fichier de résultats
contient les scores par graine : un écart entre deux configurations qui ne
dépasse pas l'écart-type inter-graines du pli n'est pas un effet, et cette
règle est appliquée à chaque comparaison du papier.

## Usage d'outils d'IA

Les scripts de ce dossier ont été conçus et écrits avec l'assistance d'un
modèle de langage (Claude, Anthropic), qui a aussi participé à l'analyse des
résultats et à la rédaction du papier. Les hypothèses testées, l'exécution
de tous les calculs, la vérification des résultats et la responsabilité des
conclusions sont de l'auteur.
