# Archetype data sources (`data/archetypes_2026_v1.json`)

Recalibration of `data/archetypes_recale_2025_v2.json` (kept unchanged). Research date: 2026-11-13. Target accuracy about ±10%.

## Method and caveats

- Price benchmark: DVF 2025 apartment price per m² (Habity, which reprocesses DVF+ Cerema 2025), by commune or arrondissement. Habity gives two slightly different figures per page (text vs yearly table); the text figure is used.
- Rent benchmark: DHUP/Cerema "Carte des loyers" 2025 median apartment rent, via Habity, plus its size-based rent estimates for Grenoble-area communes.
- **Small-unit premium (assumption, not sourced):** studio price/m² is taken as commune median x 1.10, T2 x 1.05. Furnished rent is taken as nu rent x 1.15, colocation x 1.30.
- A value was changed only when the old value was more than 10% off the benchmark. Otherwise the old value is kept ("kept, consistent").
- Habity is an aggregator, not an official source. DVF raw files (data.gouv.fr) were not queried. To be replaced by the T66 pipeline.
- Reference rent caps (`loyer_m2_max`) are mostly **unverified**: the arrêtés (Grenoble: arrêté préfectoral du 6 janvier 2026; Lyon-Villeurbanne: nov. 2025 - oct. 2026) were not read line by line.
- `taxe_fonciere_m2_an` and `charges_m2_an`: **no per-m² public source found** for any archetype, so all kept ("unverified"). Habity gives only rates and amounts per cadastral rental value (VLC), not per m². Rates 2025: Lyon 38.01%, Grenoble 76.80%. Grenoble/Lyon ratio is about 2.0, versus about 1.3 in the file, so Grenoble TF is probably understated or Lyon overstated. A follow-up is needed with real TF notices.

## Review flags verified

| Flag | Finding |
|---|---|
| Grenoble Île Verte studio 3,800 | **Not too high**: Île Verte-Saint-Roch is 3,566 €/m² (DVF), x1.10 = about 3,900. Kept. |
| Saint-Martin-d'Hères studio 3,200 | Too high: commune 2,226, studio about 2,450. Changed. |
| Échirolles 2,700 | Too high: 1,944 for the commune. Changed to 2,050. |
| Chambéry 3,800-4,000 | Too high: 2,923 (DVF+ Cerema 2025). Changed. |
| Student vacancy 0.5-1.5% | Raised to 8%. No empirical source found; this is the T26b spec requirement (DEV_PLAN). Marked "spec". |
| Lyon TF 26-30 €/m² | Not verifiable (see above). Kept, unverified. |
| `soumis_encadrement` SMH, Échirolles, Fontaine, Meylan | **Flag is justified, no change.** Décret n° 2023-1046 du 16/11/2023; arrêté préfectoral du 11/12/2024 (in force 20/01/2025), 21 communes. Meylan is fully covered. Saint-Martin-d'Hères, Échirolles and Fontaine are only partly covered (by IRIS sector), so the flag holds only for covered sectors; whether Campus, Comboire and La Poya are inside is **unverified**. Chambéry and Annecy are not in the perimeter (False kept). |
| Rent equal to the cap | Grenoble-area caps are unverified, so the pairs were kept where the rent matched the benchmark. Caps are not independently checked. |

## Changed and checked values

| Archetype | Field | Old | New | Source | Date |
|---|---|---|---|---|---|
| Studio Lyon La Doua | prix_m2 | 5200 | 5000 | Lyon apts 4,519 €/m² x1.10 (assumption); https://habity.fr/ville/lyon-69/immobilier | DVF 2025 |
| Studio Lyon La Doua | loyer_m2 | 24.0 | 22.0 | Old rent exceeded the cap; Lyon median 16.6 €/m² (same URL) x1.15 x1.15 (assumptions) | Cerema 2025 |
| Studio Lyon La Doua | loyer_m2_max | 26.0 | 23.3 | Studio/T1 cap 23.3 €/m² (+20%), same URL; https://www.justice.fr/fiche/lyon-villeurbanne-montant-loyer-bail-habitation | 2025-26 |
| Studio Lyon La Doua | vacance_pct | 0.5 | 8.0 | Spec T26b (no empirical source) | n/a |
| T2 Lyon Croix-Rousse | prix_m2 | 6100 | 5000 | Lyon 4e 4,795 €/m² (2025) x1.05; https://habity.fr/ville/lyon-4e-arrondissement-69/immobilier | DVF 2025 |
| T3 Lyon Monplaisir | prix_m2 | 4900 | 3950 | Lyon 8e 3,941 €/m²; https://habity.fr/ville/lyon-8e-arrondissement-69/immobilier | DVF 2025 |
| T4 coloc Villeurbanne | prix_m2 | 4100 | 3600 | Villeurbanne 3,596 €/m²; https://habity.fr/ville/villeurbanne-69/immobilier | DVF 2025 |
| Studio Grenoble Île Verte | vacance_pct | 1.5 | 8.0 | Spec T26b | n/a |
| Studio Grenoble Île Verte | prix_m2 | 3800 | 3800 (kept) | Île Verte-Saint-Roch 3,566 €/m²; https://habity.fr/ville/grenoble-38/immobilier | 2025 |
| T2 Grenoble Caserne de Bonne | loyer_m2 | 18.5 | 16.1 | Grenoble 40-60 m² nu about 13.4-14.1 €/m² x1.15, same URL (median 13.4, Cerema 2025) | 2025 |
| T2 Grenoble Caserne de Bonne | prix_m2 | 4200 | 4200 (unverified) | Quartier not in source (recent-build district); city max 3,566 | n/a |
| T5 coloc Grenoble Berriat | prix_m2 | 2900 | 2550 | Grenoble average 2,531 €/m², same URL (Berriat not listed) | DVF 2025 |
| T5 coloc Grenoble Berriat | loyer_m2 | 17.0 | 13.7 | 100 m² nu 10.5 €/m² (1,046 €/mo) x1.30 | 2025 |
| T3 nu Grenoble Teisseire | prix_m2, loyer_m2 | 2300, 12.4 | kept | Consistent with Grenoble average and 70 m² rent about 12.9 | 2025 |
| Studio SMH Campus | prix_m2 | 3200 | 2450 | SMH 2,226 €/m² x1.10; https://habity.fr/ville/saint-martin-d-heres-38/immobilier | DVF 2025 |
| Studio SMH Campus | vacance_pct | 1.0 | 8.0 | Spec T26b | n/a |
| Studio SMH Campus | loyer_m2 | 19.4 | 19.4 (kept) | 25 m² nu 18.9 €/m², same URL | Cerema 2025 |
| T3 nu SMH | prix_m2 | 2900 | 2250 | SMH 2,226 €/m², same URL | DVF 2025 |
| T3 nu SMH | loyer_m2 | 14.0 | 12.4 | 60 m² 13.5, 80 m² 11.9 €/m² interpolated for 72 m², same URL | Cerema 2025 |
| T2 Échirolles | prix_m2 | 2700 | 2050 | Échirolles 1,944 €/m² x1.05; https://habity.fr/ville/echirolles-38/immobilier | DVF 2025 |
| T2 Échirolles | loyer_m2 | 16.0 | 16.0 (kept) | 13.5 median x1.15 | 2025 |
| T4 coloc Fontaine | prix_m2 | 2600 | 1850 | Fontaine 1,834 €/m²; https://habity.fr/ville/fontaine-38/immobilier | DVF 2025 |
| T4 coloc Fontaine | loyer_m2 | 14.2 | 14.2 (kept) | 80 m² nu 11.2 x1.30 | 2025 |
| T2 Meylan | prix_m2 | 4500 | 3600 | Meylan 3,440 €/m² x1.05; https://habity.fr/ville/meylan-38/immobilier | DVF 2025 |
| T2 Meylan | loyer_m2 | 17.2 | 17.2 (kept) | Median 15.5 x1.15 | 2025 |
| Studio Chambéry | prix_m2 | 3800 | 3200 | Chambéry 2,923 €/m² x1.10; https://habity.fr/ville/chambery-73/immobilier (also notaires des Savoie: 2,720 median, mars 2025) | DVF 2025 |
| Studio Chambéry | vacance_pct | 1.5 | 8.0 | Spec T26b | n/a |
| T2 Chambéry | prix_m2 | 4000 | 3100 | Chambéry 2,923 x1.05, same URL | DVF 2025 |
| Chambéry (both) | loyer_m2 | 19.5 / 17.5 | kept | Median 16.1 €/m² x premiums, same URL | Cerema 2025 |
| Studio Annecy | prix_m2 | 6200 | 5600 | Annecy 5,102 €/m² x1.10; https://habity.fr/ville/annecy-74/immobilier | DVF 2025 |
| Studio Annecy | vacance_pct | 0.5 | 8.0 | Spec T26b | n/a |
| T2 Annecy Galeries | prix_m2, loyer_m2 | 5600, 21.0 | kept | Within 10% of benchmark (5,357; median rent 20.4) | 2025 |
| T3 Annecy Parmelan | prix_m2 | 5200 | kept | Annecy 5,102 | 2025 |
| T3 Annecy Parmelan | loyer_m2 | 17.5 | 20.4 | Annecy median 20.4 €/m² (Cerema 2025), same URL; Grenoble curve shows 60-70 m² rent close to median | 2025 |

## Unverified (kept as is)

- All `taxe_fonciere_m2_an` and `charges_m2_an` values.
- All `loyer_m2_max` except Studio Lyon La Doua.
- Prix for Grenoble Caserne de Bonne and Teisseire; transport, liquidity, tension and delay fields (not in scope).
- Vacancy of non-student archetypes.
- `soumis_encadrement` coverage by IRIS sector for Saint-Martin-d'Hères, Échirolles and Fontaine.

## Other sources consulted

- Encadrement Grenoble: https://www.justice.fr/fiche/grenoble-alpes-metropole-montant-loyer-bail-habitation (page updated 2026-02-02); https://www.nousgerons.com/encadrement-de-loyers-a-grenoble.html (list of 21 communes, 13 full / 8 partial).
- Lyon encadrement: the Tribunal administratif de Lyon annulled the 2023-24 arrêté on 2025-10-14; a new arrêté covers nov. 2025 - oct. 2026 (justice.fr link above).
- TF rates: https://habity.fr/ville/lyon-69/finance and https://habity.fr/ville/grenoble-38/finance (DGFiP 2025).

## DPE

- Rental ban (loi Climat et Résilience 2021-1104 art. 160; loi 89-462 art. 6, logement décent): class G unrentable from 1 Jan 2025, F from 1 Jan 2028, E from 1 Jan 2034. Source cited in `src/domain/dpe.py`: ecologie.gouv.fr and service-public.fr, "Ce que vous devez savoir sur la location des logements mal isolés".
- Assumption (not sourced to a figure): renovation cost of 150 EUR/m2 per DPE class gained, an ADEME / Effinergie order of magnitude (150-300 EUR/m2 for 1-2 classes), adjustable in the sidebar.
- Assumption: the ban applies from the ban year unless renovated. In law it bites at a new or renewed lease; furnished leases turn over every 1-3 years, so this is realistic and conservative for an investor's horizon. See TAX_SPEC 11d.
