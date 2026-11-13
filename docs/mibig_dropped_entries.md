# MIBiG entries dropped before MIBiG 4.0

DiscERN's native homology source (`--cb_source diamond`) uses MIBiG 4.0 as published by default. These entries were in earlier MIBiG releases but are not in 4.0. They can be used as references, and are part of the background, with `--mibig_set 4.0_legacy` (build it once with `discern-setup-mibig --set 4.0_legacy`). Each one is taken from the most recent release that still contained it.

244 entries; last seen in 3.1 (88), 3.0 (14), 2.0 (62), 1.4 (80).

**Reason dropped** is MIBiG's retirement reason from the MIBiG 4.0 metadata (`mibig_json_4.0_all_jsons`), where there is one.

**Closest 4.0 entry** is the MIBiG 4.0 BGC with the highest DIAMOND blast score against the dropped entry, and the **ratio** is that score divided by the entry's score against itself. Entries with a ratio of 0.9 or more are marked as near-duplicates (76 entries): in the 4.0_legacy set they sit in the background as near-identical copies of a current entry, which tightens the blast-vec threshold and can take the top BLAST hit of a query away from that entry. If the current entry is in your reference family, add the near-duplicate to the references or exclude it with `-x`.

| ID | Compound(s) | Class | Last release | Reason dropped | Closest 4.0 entry (ratio) |
|---|---|---|---|---|---|
| BGC0000005 | aflatoxin | Polyketide | 2.0 | contains single non-core gene | **BGC0000008 (0.98)** near-duplicate |
| BGC0000015 | amphotericin | Polyketide | 1.4 | not in MIBiG releases after 1.4; no retirement record in the MIBiG 4.0 metadata | BGC0000115 (0.63) |
| BGC0000025 | avermectin | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001731 (0.41) |
| BGC0000049 | E-492, E-975 | Polyketide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001653 (0.35) |
| BGC0000061 | levorin A3, C06690, FR-008-III, +2 more | Polyketide | 2.0 | Duplicate of BGC0000034 (see also BGC0000034) | **BGC0000034 (1.00)** near-duplicate |
| BGC0000089 | lovastatin | Polyketide | 1.4 | Duplicate of BGC0000088 (see also BGC0000088) | **BGC0000088 (1.00)** near-duplicate |
| BGC0000095 | micromonolactam | Polyketide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001303 (0.70) |
| BGC0000108 | natamycin | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001690 (0.87) |
| BGC0000155 | T-toxin | Polyketide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001913 (0.20) |
| BGC0000183 | paenimacrolidin | Polyketide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001025 (0.74) |
| BGC0000206 | chartreusin | Polyketide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0000207 (0.43) |
| BGC0000228 | granaticin | Polyketide | 2.0 | Duplicate of BGC0000227 (see also BGC0000227) | **BGC0000227 (1.00)** near-duplicate |
| BGC0000230 | griseorhodin A | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000266 (0.42) |
| BGC0000246 | mithramycin | Polyketide | 3.0 | Duplicate of BGC0000247 (see also BGC0000247) | **BGC0000247 (1.00)** near-duplicate |
| BGC0000250 | nogalaviketone, nogalonic acid, nogalonic acid methyl ester, +6 more | Polyketide, Saccharide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0002982 (0.41) |
| BGC0000251 | nogalamycin | Polyketide | 1.4 | Missing publication | BGC0002982 (0.24) |
| BGC0000276 | Urdamycin A | Polyketide, Saccharide | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001384 (0.63) |
| BGC0000284 | phenolic lipids | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | – |
| BGC0000299 | alterochromide A, bromoalterochromide A, dibromoalterochromide A, +13 more | NRP | 2.0 | Duplicate of BGC0000314 (see also BGC0000314) | BGC0002949 (0.03) |
| BGC0000314 | bromoalterochromide A | NRP, Polyketide, Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002949 (0.03) |
| BGC0000317 | cephalosporin C | NRP | 2.0 | Entry is spread over multiple contigs | BGC0000404 (0.34) |
| BGC0000318 | cephalosporin, cefEFG genes | NRP | 1.4 | Missing publication | BGC0001190 (0.27) |
| BGC0000338 | Echinocandin B | NRP | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001371 (0.45) |
| BGC0000356 | verruculogen, fumitremorgin B, fumitremorgin c, +4 more | NRP, Alkaloid | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000818 (0.30) |
| BGC0000358 | fusaricidin | NRP | 1.4 | Retired in the 2.0 release for unknown reasons | **BGC0001152 (0.91)** near-duplicate |
| BGC0000361 | gliotoxin, bisdethiobis(methylthio)gliotoxin | NRP | 2.0 | Entry is spread over multiple contigs | BGC0001609 (0.21) |
| BGC0000370 | HC-toxin | NRP | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001166 (0.40) |
| BGC0000389 | massetolide A | NRP | 2.0 | Entry is spread over multiple contigs | BGC0001312 (0.84) |
| BGC0000406 | phosphinothricintripeptide | NRP | 3.1 | validation failed | BGC0001006 (0.04) |
| BGC0000456 | verlamelin | NRP | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0002188 (0.56) |
| BGC0000458 | viomycin | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000316 (0.41) |
| BGC0000462 | WLIP | NRP | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001838 (0.79) |
| BGC0000463 | xantholysin A, xantholysin B, xantholysin C | NRP | 2.0 | Entry is spread over multiple contigs | BGC0000344 (0.71) |
| BGC0000481 | trichamide | RiPP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002551 (0.45) |
| BGC0000596 | SLI-2138 | RiPP | 3.1 | validation failed | **BGC0000595 (0.98)** near-duplicate |
| BGC0000629 | aflatrem | Terpene | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0002149 (0.14) |
| BGC0000655 | siomycin | Terpene | 2.0 | Duplicate of BGC0000611 (see also BGC0000611) | **BGC0000611 (1.00)** near-duplicate |
| BGC0000661 | geosmin | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000657 (0.34) |
| BGC0000680 | austinol, dehydroaustinol | Terpene | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0001729 (0.39) |
| BGC0000691 | acarbose | Saccharide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000804 (0.21) |
| BGC0000695 | fortimicin | Saccharide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000700 (0.27) |
| BGC0000697 | gentamicin | Saccharide | 1.4 | Duplicate of BGC0000696 (see also BGC0000696) | **BGC0001603 (1.00)** near-duplicate |
| BGC0000710 | neomycin | Saccharide | 1.4 | Duplicate of BGC0000709 (see also BGC0000709) | BGC0000709 (0.66) |
| BGC0000717 | streptomycin | Saccharide | 1.4 | Duplicate of BGC0000724 (see also BGC0000724) | BGC0000724 (0.36) |
| BGC0000718 | streptomycin | Saccharide | 1.4 | Duplicate of BGC0000724 (see also BGC0000724) | **BGC0000724 (0.98)** near-duplicate |
| BGC0000726 | alginate | Saccharide | 1.4 | Low quality entry | BGC0000725 (0.69) |
| BGC0000750 | capsular polysaccharide | Saccharide | 1.4 | Low quality entry | BGC0000753 (0.77) |
| BGC0000771 | lipo-oligosaccharide | Saccharide | 1.4 | Low quality entry | BGC0000770 (0.81) |
| BGC0000805 | moenomycin | Saccharide | 2.0 | Entry is spread over multiple contigs | BGC0002497 (0.79) |
| BGC0000808 | Cepacian | Saccharide | 1.4 | not in MIBiG releases after 1.4; no retirement record in the MIBiG 4.0 metadata | BGC0001588 (0.09) |
| BGC0000822 | rebeccamycin | Alkaloid | 3.0 | Duplicate of BGC0000821 (see also BGC0000821) | **BGC0000821 (0.97)** near-duplicate |
| BGC0000823 | rebeccamycin | Alkaloid | 3.0 | Duplicate of BGC0000821 (see also BGC0000821) | **BGC0000821 (1.00)** near-duplicate |
| BGC0000824 | Scytonemin | Alkaloid | 1.4 | not in MIBiG releases after 1.4; no retirement record in the MIBiG 4.0 metadata | BGC0000668 (0.18) |
| BGC0000828 | violacein | Alkaloid | 1.4 | Duplicate of BGC0000829 (see also BGC0000829) | **BGC0000829 (1.00)** near-duplicate |
| BGC0000835 | simocyclinone | Polyketide | 3.0 | Duplicate of BGC0000270 (see also BGC0000270) | **BGC0000270 (1.00)** near-duplicate |
| BGC0000839 | flexirubin | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000838 (0.25) |
| BGC0000887 | saxitoxin, gonyautoxin 2, gonyautoxin 3, +6 more | Alkaloid | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000188 (0.74) |
| BGC0000915 | mitomycin | Other | 3.1 | validation failed | BGC0003049 (0.09) |
| BGC0000928 | T3 toxin | Other | 2.0 | Duplicate of BGC0000188 (see also BGC0000188) | **BGC0000188 (1.00)** near-duplicate |
| BGC0000931 | trichothecene | Other | 1.4 | Duplicate of BGC0000930 (see also BGC0000930) | **BGC0001277 (1.00)** near-duplicate |
| BGC0000952 | pristinamycin | NRP, Polyketide | 1.4 | Cluster split over multiple loci. | BGC0001116 (0.17) |
| BGC0000962 | barbamide | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002886 (0.26) |
| BGC0001009 | melithiazol A | NRP, Polyketide | 2.0 | Duplicate of BGC0001010 (see also BGC0001010) | **BGC0001010 (1.00)** near-duplicate |
| BGC0001068 | pyripyropene A | Terpene, Polyketide | 3.1 | validation failed | BGC0000129 (0.33) |
| BGC0001081 | lolitrem | Terpene, Alkaloid | 1.4 | Retired in the 2.0 release for unknown reasons | BGC0003163 (0.57) |
| BGC0001085 | teleocidin B1 | NRP, Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000391 (0.43) |
| BGC0001086 | astaxanthin dideoxyglycoside | Saccharide, Terpene | 1.4 | Entry is spread over multiple contigs | BGC0000634 (0.44) |
| BGC0001087 | sioxanthin | Saccharide, Terpene | 1.4 | Entry split over multiple genomic loci; Coordinates wrong | BGC0000664 (0.12) |
| BGC0001092 | diaphorin | NRP, Polyketide | 1.4 | Entry spread over multipe genomic loci. | BGC0001108 (0.10) |
| BGC0001096 | FR901464 | NRP, Polyketide | 2.0 | Duplicate of BGC0000995 (see also BGC0000995) | BGC0000995 (0.89) |
| BGC0001104 | myxovirescin A1 | NRP, Polyketide | 2.0 | Duplicate of BGC0001025 (see also BGC0001025) | **BGC0001025 (1.00)** near-duplicate |
| BGC0001107 | patellazole | NRP, Polyketide | 1.4 | Low quality entry | BGC0001299 (0.03) |
| BGC0001120 | burkholderic acid | NRP, Polyketide | 2.0 | Duplicate of BGC0001102 (see also BGC0001102) | BGC0001368 (0.02) |
| BGC0001121 | orsellinic acid | Polyketide | 1.4 | Duplicate of BGC0000057 (see also BGC0000057) | **BGC0000057 (1.00)** near-duplicate |
| BGC0001158 | bikaverin | Polyketide | 1.4 | Entry is spread over multiple contigs | **BGC0000030 (0.93)** near-duplicate |
| BGC0001180 | isopropylstilbene, 2,5-dialkylresorcinols | Other | 1.4 | Entry spread over multiple genomic loci | BGC0000383 (0.49) |
| BGC0001182 | chaetoglobosins | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002546 (0.70) |
| BGC0001191 | A-102395 | Other | 3.1 | validation failed | BGC0000288 (0.72) |
| BGC0001196 | desotamide | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001814 (0.51) |
| BGC0001208 | poaeamide A | NRP | 1.4 | Entry split over multiple genomic loci | BGC0002779 (0.60) |
| BGC0001232 | leucanicidin | Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0000028 (0.84) |
| BGC0001234 | 4,4'-piperazine-2,5-diyldimethyl-bis-phenol | NRP | 1.4 | Low quality entry | BGC0002167 (0.61) |
| BGC0001236 | thiotetroamide | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001237 (0.41) |
| BGC0001243 | botryenalol, botrydial, dihydrobotrydial, +4 more | Polyketide | 1.4 | Entry is split over multiple genomic loci | BGC0001784 (0.02) |
| BGC0001247 | cercosporin | Polyketide | 1.4 | Reference sequence contains only a single gene | BGC0001541 (0.87) |
| BGC0001253 | ACT-Toxin I | Polyketide | 1.4 | Reference sequence contains only a single gene | BGC0001254 (0.60) |
| BGC0001256 | atromentin | Other | 1.4 | Reference sequence contains only a single gene | BGC0002277 (0.51) |
| BGC0001259 | elsinochrome A | Polyketide | 1.4 | Reference sequence contains only a single gene | BGC0002161 (0.20) |
| BGC0001263 | ergometrine | NRP | 1.4 | Reference sequence contains only a single gene | **BGC0002232 (0.97)** near-duplicate |
| BGC0001265 | melanin | Polyketide | 2.0 | Entry is spread over multiple contigs | BGC0001258 (0.47) |
| BGC0001269 | gibberellin A3, gibberellin A1, gibberellin A7, +2 more | Polyketide, Terpene | 1.4 | Reference sequence contains only a single gene | BGC0001604 (0.27) |
| BGC0001270 | gibberellin A3, gibberellin A1, gibberellin A7, +2 more | Polyketide, Terpene | 1.4 | Reference sequence contains only a single gene | **BGC0001604 (0.94)** near-duplicate |
| BGC0001271 | emericellin | Polyketide | 1.4 | Entry is split over multiple genomic loci | BGC0000101 (0.71) |
| BGC0001272 | elsinochrome B, elsinochrome C | Polyketide | 1.4 | Duplicate of BGC0001259 (see also BGC0001259) | BGC0002161 (0.20) |
| BGC0001274 | non-acetylated open-chain sophorolipid, diacetylated lactonic sophorolipid | Other | 1.4 | Reference sequence contains only a single gene | – |
| BGC0001279 | deoxysambucinol, sambucinol, roridin E | Terpene | 1.4 | Reference sequence contains only a single gene | BGC0001278 (0.53) |
| BGC0001280 | betaenone C, betaenone A | Polyketide | 2.0 | Duplicate of BGC0001264 (see also BGC0001264) | **BGC0001264 (1.00)** near-duplicate |
| BGC0001282 | lolitriol | Other | 1.4 | Entry is spread over multiple contigs | BGC0000815 (0.78) |
| BGC0001308 | simocyclinone D9, simocyclinone D10, simocyclinone D11 | Polyketide | 1.4 | Low quality entry | BGC0001072 (0.74) |
| BGC0001309 | usnic acid | Polyketide | 1.4 | Reference sequence contains only a single gene | **BGC0002483 (0.98)** near-duplicate |
| BGC0001319 | alpha-tomatine | Terpene | 1.4 | Low quality entry | BGC0002722 (0.22) |
| BGC0001320 | alpha-chaconine, alpha-solanine | Terpene | 1.4 | Low quality entry | – |
| BGC0001323 | monoterpenes-diterpenes | Terpene | 3.1 | validation failed | **BGC0001322 (0.95)** near-duplicate |
| BGC0001347 | Poaeamide B | NRP | 1.4 | Entry is split over multiple genomic loci | BGC0002779 (0.67) |
| BGC0001350 | phormidolide | Polyketide | 3.1 | validation failed | BGC0001032 (0.14) |
| BGC0001352 | Tu 3010 | NRP, Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0001237 (0.23) |
| BGC0001353 | thiolactomycin | NRP, Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0002642 (0.40) |
| BGC0001354 | U-68204 | NRP, Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0001237 (0.40) |
| BGC0001355 | U-68204 | NRP, Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0001237 (0.41) |
| BGC0001359 | PM100117, PM100118 | Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0001856 (0.76) |
| BGC0001362 | tropodithietic acid | Other | 2.0 | Duplicate of BGC0000932 (see also BGC0000932) | **BGC0000932 (1.00)** near-duplicate |
| BGC0001378 | tiancimycin | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001377 (0.45) |
| BGC0001382 | svaricin | NRP, Polyketide | 1.4 | Low quality entry | BGC0003160 (0.11) |
| BGC0001385 | byssochlamic acid, agnestadrides | Polyketide | 1.4 | Duplicate of BGC0001340 (see also BGC0001340) | **BGC0001340 (1.00)** near-duplicate |
| BGC0001401 | melleolide F, 6'-Chloromelleolide F, 6'-Bromomelleolide F | Polyketide | 1.4 | Reference sequence contains only a single gene | BGC0001923 (0.43) |
| BGC0001402 | cyclochlorotine, hydroxycyclochlorotine A, cyclotine, +2 more | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002264 (0.03) |
| BGC0001436 | cornexistin | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1); listed as active again in MIBiG 5.0rc1 | BGC0002222 (0.20) |
| BGC0001450 | macrophorin A, 4'-oxomacrophorin A, 4'-oxomacrophorin D, +1 more | Terpene | 1.4 | Reference sequence contains only a single gene | **BGC0002615 (1.00)** near-duplicate |
| BGC0001466 | Wortmanamide | Polyketide | 1.4 | Reference sequence contains only a single gene | **BGC0001954 (1.00)** near-duplicate |
| BGC0001476 | pseudouridimycin | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000283 (0.27) |
| BGC0001479 | anabaenopeptin NZ857, nostamide A | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000302 (0.68) |
| BGC0001482 | – | – | 1.4 | not in MIBiG releases after 1.4; no retirement record in the MIBiG 4.0 metadata | **BGC0001483 (1.00)** near-duplicate |
| BGC0001487 | 3,7-dihydroxytropolone | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001008 (0.11) |
| BGC0001494 | acinetodin | RiPP | 3.1 | validation failed | BGC0000864 (0.03) |
| BGC0001497 | actinonin | Polyketide | 2.0 | Duplicate of BGC0001442 (see also BGC0001442) | **BGC0001442 (1.00)** near-duplicate |
| BGC0001530 | bicyclomycin | Other | 2.0 | Duplicate of BGC0001468 (see also BGC0001468) | **BGC0001468 (1.00)** near-duplicate |
| BGC0001538 | caboxamycin | NRP, Polyketide | 2.0 | Duplicate of BGC0001444 (see also BGC0001444) | BGC0001444 (0.88) |
| BGC0001546 | chuangxinmycin | Alkaloid | 2.0 | Duplicate of BGC0001485 (see also BGC0001485) | **BGC0001485 (1.00)** near-duplicate |
| BGC0001557 | cornexistin | Polyketide | 2.0 | Duplicate of BGC0001436 (see also BGC0001436) | BGC0002222 (0.20) |
| BGC0001559 | crocagin A, crocagin B | RiPP | 3.1 | validation failed | BGC0002962 (0.08) |
| BGC0001577 | divamide B | RiPP | 3.1 | validation failed | **BGC0001576 (0.97)** near-duplicate |
| BGC0001581 | ECO-02301 | Polyketide | 1.4 | Low quality entry | **BGC0000052 (0.95)** near-duplicate |
| BGC0001583 | emodin | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001886 (0.55) |
| BGC0001598 | foxicin A, foxicin B, foxicin C, +1 more | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001065 (0.06) |
| BGC0001599 | fragin | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | – |
| BGC0001613 | harzianum B | Terpene | 1.4 | Reference sequence contains only a single gene | BGC0001278 (0.33) |
| BGC0001614 | hassallidin E | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000369 (0.66) |
| BGC0001643 | lactomycins A-C | Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0000123 (0.86) |
| BGC0001646 | lagriamide | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002810 (0.02) |
| BGC0001664 | merosterol | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001595 (0.41) |
| BGC0001669 | microvionin | RiPP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000528 (0.16) |
| BGC0001673 | moomysin | RiPP | 3.0 | Duplicate of BGC0001539 (see also BGC0001539) | **BGC0001539 (0.98)** near-duplicate |
| BGC0001676 | mycemycin A, mycemycin B, mycemycin C, +2 more | Alkaloid | 1.4 | Entry is spread over multiple contigs | BGC0002443 (0.74) |
| BGC0001678 | mycosporine-like amino acids (MAAs) | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | – |
| BGC0001686 | N-octanoyl-Met-Phe-H | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | – |
| BGC0001688 | N-tetradecanoyl tyrosine | Other | 2.0 | Duplicate of BGC0001591 (see also BGC0001591) | **BGC0001591 (1.00)** near-duplicate |
| BGC0001689 | naphthgeranine A | Polyketide | 1.4 | Missing literature reference | BGC0003029 (0.54) |
| BGC0001691 | nectrisine | Other | 1.4 | Reference sequence contains only a single gene | – |
| BGC0001697 | neurosporin A | Polyketide | 2.0 | pubmed:30908040 hints that this locus produces sordarial, covered by BGC0002729 (see also BGC0002729) | BGC0002729 (0.58) |
| BGC0001698 | nevaltophin A, nevaltophin B, nevaltophin C, +1 more | NRP | 2.0 | Duplicate of BGC0001434 (see also BGC0001434) | **BGC0001434 (1.00)** near-duplicate |
| BGC0001710 | obafluorin | NRP | 2.0 | Duplicate of BGC0001437 (see also BGC0001437) | **BGC0001437 (1.00)** near-duplicate |
| BGC0001714 | octacosamicin | NRP, Polyketide | 1.4 | Entry is spread over multiple contigs | BGC0001662 (0.31) |
| BGC0001730 | paramagnetoquinone 1, paramagnetoquinone 2 | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000279 (0.20) |
| BGC0001734 | penilumamide | RiPP | 1.4 | GenBank sequence is retracted due to reason:unconfirmed sequence | BGC0001281 (0.02) |
| BGC0001745 | PlpA1, PlpA2, PlpA3 | RiPP | 3.1 | validation failed | BGC0001733 (0.22) |
| BGC0001754 | ralsolamycin | NRP, Polyketide | 2.0 | Duplicate of BGC0001363 (see also BGC0001363) | **BGC0001363 (1.00)** near-duplicate |
| BGC0001766 | salinamide A | NRP | 2.0 | Duplicate of BGC0001230 (see also BGC0001230) | **BGC0001230 (1.00)** near-duplicate |
| BGC0001767 | salinichelins | NRP | 3.1 | validation failed | BGC0002466 (0.70) |
| BGC0001777 | shearinine D | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001260 (0.52) |
| BGC0001798 | tetrapetalone A, tetrapetalone B, tetrapetalone C, +1 more | Polyketide | 1.4 | Missing literature reference | BGC0001666 (0.26) |
| BGC0001803 | thioviridamide | RiPP | 2.0 | Duplicate of BGC0000625 (see also BGC0000625) | BGC0000625 (0.65) |
| BGC0001804 | tiancimycin | Polyketide | 2.0 | Duplicate of BGC0001378 (see also BGC0001378) | BGC0001377 (0.45) |
| BGC0001805 | tilivalline | NRP | 2.0 | Duplicate of BGC0000446 (see also BGC0000446) | **BGC0000446 (1.00)** near-duplicate |
| BGC0001815 | uncialamycin | Polyketide | 2.0 | Duplicate of BGC0001377 (see also BGC0001377) | **BGC0001377 (1.00)** near-duplicate |
| BGC0001821 | verticillins | NRP | 1.4 | Duplicate of BGC0001820 (see also BGC0001820) | **BGC0001820 (1.00)** near-duplicate |
| BGC0001829 | α,β-epoxyketone | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000345 (0.81) |
| BGC0001837 | leptolyngbyalide | Polyketide | 3.1 | validation failed | BGC0001031 (0.15) |
| BGC0001883 | porphyra-334 | Other | 2.0 | Duplicate of BGC0001678 (see also BGC0001678) | – |
| BGC0001892 | botcinic acid | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001784 (0.01) |
| BGC0001893 | abscisic acid | Other | 3.1 | validation failed | BGC0001899 (0.48) |
| BGC0001898 | thioplatensimycin, thioplatencin | Terpene | 2.0 | Duplicate of BGC0001140 (see also BGC0001140) | BGC0001140 (0.89) |
| BGC0001900 | fragin | NRP | 2.0 | Duplicate of BGC0001599 (see also BGC0001599) | – |
| BGC0001901 | tricholignan A | Polyketide | 2.0 | Duplicate of BGC0001854 (see also BGC0001854) | **BGC0001854 (1.00)** near-duplicate |
| BGC0001908 | sceliphrolactam | Polyketide | 2.0 | Duplicate of BGC0001770 (see also BGC0001770) | **BGC0001770 (1.00)** near-duplicate |
| BGC0001912 | (2S,6R)-diamino-(5R,7)-dihydroxy-heptanoic acid | NRP | 2.0 | Duplicate of BGC0001818 (see also BGC0001818) | **BGC0001818 (1.00)** near-duplicate |
| BGC0001914 | benzastatin J, 7-hydroxyl benzastatin B, 7-hydroxyl O-demethylbenzastatin A, +8 more | Other | 2.0 | Duplicate of BGC0001529 (see also BGC0001529) | **BGC0001529 (1.00)** near-duplicate |
| BGC0001915 | betaestacin I, betaestacin II | Terpene | 2.0 | Duplicate of BGC0001528 (see also BGC0001528) | **BGC0001528 (1.00)** near-duplicate |
| BGC0001916 | lavendiol | Polyketide | 2.0 | Duplicate of BGC0001649 (see also BGC0001649) | **BGC0001649 (1.00)** near-duplicate |
| BGC0001921 | MS-271 | Other | 2.0 | Duplicate of BGC0001920 (see also BGC0001920) | **BGC0002986 (1.00)** near-duplicate |
| BGC0001930 | pallidocin | RiPP | 2.0 | Duplicate of BGC0001862 (see also BGC0001862) | **BGC0001862 (1.00)** near-duplicate |
| BGC0001932 | mediomiycin A | Polyketide | 2.0 | Duplicate of BGC0001662 (see also BGC0001662) | **BGC0001662 (1.00)** near-duplicate |
| BGC0001933 | sphaerocyclamide | RiPP | 2.0 | Duplicate of BGC0001505 (see also BGC0001505) | **BGC0001505 (1.00)** near-duplicate |
| BGC0001934 | armeniaspirols | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | **BGC0002022 (0.99)** near-duplicate |
| BGC0001935 | 3,7-dihydroxytropolone | Other | 2.0 | Duplicate of BGC0001487 (see also BGC0001487) | BGC0001008 (0.11) |
| BGC0001936 | nodulisporic acid F | Terpene | 2.0 | Duplicate of BGC0001706 (see also BGC0001706) | **BGC0001706 (1.00)** near-duplicate |
| BGC0001938 | entotheonella | Polyketide | 3.1 | validation failed | **BGC0001186 (0.99)** near-duplicate |
| BGC0001939 | streptoseomycin | Polyketide | 2.0 | Duplicate of BGC0001784 (see also BGC0001784) | **BGC0001784 (1.00)** near-duplicate |
| BGC0001943 | vioprolides A, vioprolides B, vioprolides C, +1 more | NRP | 2.0 | Duplicate of BGC0001822 (see also BGC0001822) | **BGC0001822 (1.00)** near-duplicate |
| BGC0001946 | lagriamide | NRP, Polyketide | 2.0 | Duplicate of BGC0001646 (see also BGC0001646) | BGC0002810 (0.02) |
| BGC0001949 | domoic acid | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001974 (0.31) |
| BGC0001958 | amipurimycin | Other | 2.0 | Duplicate of BGC0001957 (see also BGC0001957) | **BGC0001957 (1.00)** near-duplicate |
| BGC0001961 | ashimides | NRP | 3.0 | Duplicate of BGC0002288 (see also BGC0002288) | BGC0002288 (0.86) |
| BGC0001973 | fervenulin | Other | 2.0 | Duplicate of BGC0001972 (see also BGC0001972) | **BGC0001972 (1.00)** near-duplicate |
| BGC0001977 | eupenifeldin | Other | 2.0 | Duplicate of BGC0001976 (see also BGC0001976) | **BGC0001976 (1.00)** near-duplicate |
| BGC0001978 | burnettramic acid | NRP, Polyketide | 2.0 | Duplicate of BGC0001857 (see also BGC0001857) | **BGC0001857 (1.00)** near-duplicate |
| BGC0001982 | pyrichalasin H | NRP, Polyketide | 2.0 | Duplicate of BGC0001881 (see also BGC0001881) | **BGC0001881 (1.00)** near-duplicate |
| BGC0001990 | BU-4704, xanthocillin-X monomethylether, melanocin E, +2 more | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000010 (0.07) |
| BGC0001993 | paenilipoheptin | NRP | 2.0 | Duplicate of BGC0001728 (see also BGC0001728) | **BGC0001728 (1.00)** near-duplicate |
| BGC0002003 | glidopeptin A | NRP | 2.0 | Duplicate of BGC0001608 (see also BGC0001608) | **BGC0001608 (1.00)** near-duplicate |
| BGC0002016 | lugdunomycin | Polyketide | 3.1 | validation failed | BGC0002477 (0.53) |
| BGC0002023 | saalfelduracin | RiPP | 2.0 | Duplicate of BGC0001471 (see also BGC0001471) | **BGC0001471 (0.93)** near-duplicate |
| BGC0002027 | TglA-thiaGlu | RiPP | 2.0 | Duplicate of BGC0001486 (see also BGC0001486) | **BGC0001486 (1.00)** near-duplicate |
| BGC0002029 | bagremycin A, bagremycin B, bagremycin C, +6 more | Other | 2.0 | Duplicate of BGC0001592 (see also BGC0001592) | **BGC0001592 (1.00)** near-duplicate |
| BGC0002030 | viriditoxin | Polyketide | 2.0 | Duplicate of BGC0001866 (see also BGC0001866) | BGC0001866 (0.78) |
| BGC0002041 | pyrazofurin A | Other | 3.0 | Duplicate of BGC0001985 (see also BGC0001985) | BGC0001985 (0.80) |
| BGC0002061 | nostopeptolide A1, nostopeptolide 1052 | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001028 (0.85) |
| BGC0002069 | styrolide A, styrolide B | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000186 (0.16) |
| BGC0002076 | 4,5-didehydroacaterin, acaterin | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | – |
| BGC0002094 | B-ethynylserine | Other | 3.0 | Duplicate of BGC0001994 (see also BGC0001994) | **BGC0001994 (0.98)** near-duplicate |
| BGC0002105 | lucensomycin | Polyketide | 3.1 | validation failed | BGC0002333 (0.86) |
| BGC0002116 | darobactin A | RiPP | 3.1 | listed as 'active' in the MIBiG 4.0 metadata, but missing from the MIBiG 4.0 protein FASTA; listed as active again in MIBiG 5.0rc1 | BGC0002442 (0.06) |
| BGC0002160 | aspergillicin A, aspergillicin F, aspergillicin G | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001372 (0.02) |
| BGC0002166 | peramine, intermediate 1, intermediate 2 | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002164 (0.52) |
| BGC0002178 | gramillin A, gramillin B | NRP | 3.1 | validation failed | BGC0001724 (0.02) |
| BGC0002217 | HEx-tc3 terpene | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001571 (0.06) |
| BGC0002223 | scytalidin | Polyketide | 3.1 | validation failed | BGC0002222 (0.55) |
| BGC0002224 | zopfiellin | Polyketide | 3.1 | validation failed; listed as active again in MIBiG 5.0rc1 | **BGC0002222 (0.90)** near-duplicate |
| BGC0002292 | pyrazofurin | Other | 3.0 | Duplicate of BGC0001985 (see also BGC0001985) | **BGC0001985 (1.00)** near-duplicate |
| BGC0002325 | mycetohabin-16 | RiPP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000571 (0.71) |
| BGC0002353 | sch40832, Sch 40832 | RiPP | 3.1 | validation failed | BGC0000611 (0.38) |
| BGC0002380 | microviridin 1777 | RiPP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000592 (0.83) |
| BGC0002388 | oryzalides, phytocassanes | Terpene | 3.0 | Duplicate of BGC0000672 (see also BGC0000672) | BGC0000672 (0.82) |
| BGC0002389 | β-bisabolene, β-macrocarpene | Terpene | 3.1 | validation failed | – |
| BGC0002396 | tanshinones | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | – |
| BGC0002399 | caprutriene | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001756 (0.84) |
| BGC0002428 | DDP-13, DDP-14 | Terpene, Polyketide | 3.1 | validation failed | – |
| BGC0002437 | thermoactinoamide A | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001532 (0.43) |
| BGC0002444 | epoxy-Janthitrems | Terpene | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001706 (0.69) |
| BGC0002457 | alazopeptin | NRP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002536 (0.66) |
| BGC0002463 | staphyloferrin B | Other | 3.0 | Duplicate of BGC0000943 (see also BGC0000943) | **BGC0000943 (1.00)** near-duplicate |
| BGC0002486 | tolypamide | RiPP | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000480 (0.70) |
| BGC0002566 | [S,S]-EDDS | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002568 (0.43) |
| BGC0002585 | ubericin K | Other | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001602 (0.16) |
| BGC0002590 | colibromycin A, colibromycin B, colibromycin C | NRPS, PKS | 3.0 | Duplicate of BGC0002100 (see also BGC0002100) | **BGC0002100 (1.00)** near-duplicate |
| BGC0002618 | penitremane | Terpene | 3.1 | validation failed | BGC0001375 (0.15) |
| BGC0002624 | fischerazole A, fischerazole B, fischerazole C | NRP, Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0001884 (0.31) |
| BGC0002627 | kamptornamide | RiPP | 3.1 | validation failed | BGC0002699 (0.07) |
| BGC0002630 | thatisin | RiPP | 3.1 | validation failed | BGC0002005 (0.23) |
| BGC0002632 | varlaxin 1046A, varlaxin 1022A | NRP | 3.1 | validation failed | BGC0003044 (0.71) |
| BGC0002636 | streptophenazine 1, streptophenazine 2 | Polyketide | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0002010 (0.60) |
| BGC0002647 | massinidine | Alkaloid | 3.1 | not in MIBiG 4.0; no retirement record in the MIBiG 4.0 metadata (it was 'active' in 3.1) | BGC0000380 (0.12) |
| BGC0002662 | petrichorin A, petrichorin B | NRP | 3.1 | validation failed | BGC0001370 (0.03) |
| BGC0002714 | luminmycin A, glidobactin A, cepafungin | NRP, Polyketide | 3.0 | Duplicate of BGC0000383 (see also BGC0000383) | **BGC0000383 (1.00)** near-duplicate |
| BGC0002728 | aculene A, aculene B, aculene C, +1 more | NRP, Terpene | 3.0 | Duplicate of BGC0002225 (see also BGC0002225) | **BGC0002225 (1.00)** near-duplicate |
