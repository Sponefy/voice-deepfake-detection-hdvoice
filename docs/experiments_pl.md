# Detekcja deepfake audio w warunkach HD Voice — podsumowanie eksperymentów OOD

> Stan na 2026-06-10. Szkielet rozdziału wynikowego pracy inżynierskiej: diagnoza → interwencje →
> wyniki → wnioski, z mapą reprodukcji (które skrypty dają które liczby).
>
> **Wynik finalny: OOD test AUC 0.904 / EER 0.171** (model E6 — pełny fine-tuning + label
> smoothing + uczciwa selekcja epoki po val_ood). Progresja: 0.43 → 0.755 → 0.878 → **0.904**.

---

## 1. Problem: AUC ≈ 1.0 było iluzją (shortcut learning)

Pierwotny model (Wav2Vec2 XLS-R-300M, fine-tuning na V1/V2) osiągał AUC ≈ 1.0 jednocześnie na
train/val/test — co nie było przeuczeniem (byłoby train ≫ val), tylko **shortcut learningiem**:
model uczył się sygnału stałego per klasa, niezależnego od mówcy i splitu.

**Dowody (wstępna diagnoza):**

| skrót | dowód | naprawa |
|---|---|---|
| długość nagrania | deepfake mediana 12.8 s vs bonafide 3.6 s; regresja logistyczna na samej długości: AUC 0.93 | krótkie okno bez nadmiarowego paddingu |
| odcisk kanału/vocodera | 6 cech spektralnych bez długości: AUC 0.82 | symetryczna augmentacja, wspólny pipeline HD Voice |
| domena bonafide | bonafide wyłącznie studyjne (VCTK, LibriTTS-R) | różnorodne źródła bonafide |

**Miarą realnej generalizacji jest ewaluacja OOD** (out-of-distribution) na zbiorze
**In-the-Wild** (Müller i in. 2022; nagrania celebrytów z YouTube, 54 mówców, bonafide+spoof tych
samych osób): pierwotny model osiągał tam **AUC 0.43** — gorzej niż losowo (kolaps na jedną klasę).

---

## 2. Interwencje (chronologicznie)

### Etap I — naprawa potoku (na małym podzbiorze):
okno 3 s bez paddingu, symetryczna augmentacja kanału, split po mówcy, różnorodne dane.
Progres OOD AUC: **0.43 → 0.60 → 0.66 → 0.755** — każdy skok dała dodana różnorodność DANYCH,
nie zmiana modelu (ablacja rozmiaru: base 0.744 vs XLS-R-300M 0.759).

### Etap II — pełny zbiór V2 z balansem źródeł (ta praca):

1. **Rozszerzenie bonafide o różnorodne korpusy:**
   - **Common Voice 25.0 (en)**: selekcja 150 h od **49 625 różnych mówców** (round-robin po
     mówcach, maks. różnorodność) → `selectCommonVoiceSubset.py`;
   - **People's Speech**: ~15 GiB streaming z limitem na źródło (≥120 źródeł) →
     `huggingface_download.py`;
   - wszystkie dane przepuszczone przez **ten sam pipeline HD Voice** (16 kHz mono, bandpass
     50–7000 Hz, Opus VoIP 6.6–23.85 kbps losowo) → `compressDataset.py` — brak skrótu kanałowego
     między klasami.
2. **Balans źródeł bonafide PO CZASIE trwania** (nie liczbie plików): po **77.1 h z każdego z 4
   źródeł** ≈ czas deepfake; split po mówcy → `createBalancedOodSplit.py`.
3. **Ewaluacja = OOD**: split `test` to In-the-Wild (zamiast testu in-domain).
4. **Label smoothing 0.05** przeciw overconfidence → `detector.py`/`train.py`.
5. **Selekcja modelu po `val_ood` AUC** (a nie po in-domain val): in-domain val nasyca się
   (AUC→1.0 po ~2 epokach) i jest ślepy na generalizację; OOD val ma zapas i koreluje z
   tym, co mierzymy. Best-model i scheduler idą po `val_ood` AUC; `test` (rozłączni mówcy)
   pozostaje nietknięty → `detector.py` `train(select_loader=...)`.
6. **Early stopping** po `val_ood` AUC (`patience=3`) — pełny fine-tuning szczytuje OOD wcześnie.
7. **Kalibracja progu**: In-the-Wild dzielone po mówcach 27/27 na `val_ood` (selekcja epoki +
   dobór progu, Youden J = max balanced accuracy) i `test` (raport) → `evalOodFromCsv.py`.
8. **Temperature scaling**: dopasowanie skalara `T` na `val_ood` (min. NLL) — prostuje nasycony
   softmax, daje zdrowy próg i kalibrowane prawdopodobieństwa (ECE). Nie zmienia AUC/EER.

---

## 3. Dane

### Źródła (po kompresji HD Voice, `V2/compressed/`)

| klasa | źródło | pliki | dostępne godziny | użyte w train+val |
|---|---|---:|---:|---:|
| bonafide | CommonVoice | 101 670 | 150.0 h | 77.1 h |
| bonafide | LibriTTS-R | 160 231 | 261.4 h | 77.1 h |
| bonafide | PeoplesSpeech | 64 528 | 244.0 h | 77.1 h |
| bonafide | VCTK_Corpus | 88 328 | 82.7 h | 77.1 h |
| **bonafide razem** | | **414 757** | **738.1 h** | **308.4 h** |
| deepfake | AUDETER (10 generatorów TTS) | 75 760 | 308.4 h | 308.4 h (całość) |
| OOD | In-The-Wild | 31 779 | — | tylko ewaluacja |

### Splity (`dataset_split_balanced_ood.csv`, wersja 5, seed 42)

| split | bonafide | deepfake | rola |
|---|---:|---:|---|
| train | 169 009 | 64 396 | trening (sampler 50/50, okno 4 s) |
| val | 30 216 | 11 364 | in-domain, selekcja modelu po VAL AUC |
| val_ood | 7 144 | 3 625 | In-the-Wild, 27 mówców — **selekcja epoki + kalibracja progu** |
| test | 12 819 | 8 191 | In-the-Wild, 27 mówców — **finalny pomiar (nietknięty)** |

**Zasady bez przecieku:** bonafide dzielone po mówcy; In-the-Wild dzielone po mówcy (te same
osoby mają tam nagrania prawdziwe i fake — split po osobie). `val_ood` służy do selekcji epoki
ORAZ doboru progu/temperatury; `test` (rozłączni mówcy) jest używany wyłącznie do raportu.
Ponieważ `val_ood` wchodzi w pętlę selekcji, **nagłówek liczymy na `test`-only** (pełny
In-the-Wild = poglądowo, jest lekko optymistyczny). AUC/EER są threshold-free.

### Konfiguracja treningu

XLS-R-300M (`facebook/wav2vec2-xls-r-300m`, 315.7M parametrów; frozen: 0.296M trenowanych)
+ głowica MLP 1024→256→128→2, mean pooling, dropout 0.3. AdamW lr 5e-5, weight_decay 0.01,
batch 16 × grad-accum 2, FP16, WeightedRandomSampler (klasy 50/50), okno 4 s (64 000 sampli)
z attention mask, label smoothing 0.05 (od E3), 10 epok z early stoppingiem (`patience=3`),
selekcja best-modelu i ReduceLROnPlateau po **val_ood AUC** (od E6; wcześniej po in-domain VAL).

---

## 4. Wyniki

### Tabela główna (OOD In-the-Wild)

| # | konfiguracja | in-domain val AUC | pełny AUC (54 mów.) | pełny EER | test-half AUC (27 mów.) | próg kalib. | BalAcc test po kalibracji |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | oryginał (przed balansem) | ~1.0 | 0.43 | — | — | — | — |
| E1 | unfrozen, bez LS (ep. 1) | 0.9999 (nasycony) | 0.756 | —* | 0.733 | **1.000** | 0.689 |
| E2 | frozen, bez LS (ep. 1) | — | —* | —* | 0.774 | 0.860 | 0.726 |
| E3 | frozen + LS (ep. 1) | 0.892 | —* | —* | 0.790 | 0.818 | 0.735 |
| E4 | frozen + LS (10 epok) | 0.965 | 0.789 | 0.282 | 0.836 | 0.724 | 0.763 |
| E5 | unfrozen + LS (best = ep. 2, selekcja in-domain) | 1.0000 (nasycony) | 0.878 | 0.195 | 0.884 | 0.975 | 0.813 |
| **E6** | **unfrozen + LS + selekcja po val_ood (best = ep. 7)** | 1.0000 (nasycony) | 0.892 | 0.187 | **0.904** | 0.742 (po T) | 0.786 |

\* miara „pełny In-the-Wild" dodana do skryptu ewaluacji po E3 — wcześniejsze biegi mają tylko połowę `test`.
Od E6 nagłówkiem jest **test-half AUC** (val_ood wchodzi w selekcję — patrz sekcja 3). E5 vs E6: E5
wybrany trafem (in-domain val nasycony), E6 uczciwie po val_ood — i wyżej (0.904 vs 0.884 na test).

### Wynik finalny (E6: pełny fine-tuning + label smoothing + selekcja po val_ood)

```
★ TEST (held-out, 21 010 sampli, rozłączni mówcy — GŁÓWNA MIARA): AUC 0.904 | EER 0.171
  poglądowo — pełny In-the-Wild (31 779): AUC 0.892 | EER 0.187  (val_ood 0.876)
  kalibracja progu (dobrana na val_ood):
    @0.5 (surowy):        BalAcc 0.653  (Bona 0.991 / Spoof 0.315)   ← przepewnienie
    @temperature T=3.59:  próg 0.742, BalAcc 0.786  (Bona 0.661 / Spoof 0.912)
    ECE test: 0.238 → 0.156 (po temperaturze)
```

### Redukcja luki in-domain ↔ OOD (dowód naprawy shortcutu)

| model | in-domain | OOD | luka |
|---|---:|---:|---:|
| oryginał | ~1.0 | 0.43 | **0.57** |
| **E6** | ~1.0 (val nasycony) | **0.904** | ~0.10 |
| E5 | ~1.0 (val nasycony) | 0.878 | ~0.12 |
| E4 (frozen+LS) | 0.965 | 0.789 | ~0.18 |

---

## 5. Wnioski

1. **Dźwignią generalizacji są dane, nie architektura.** Balans źródeł bonafide po czasie
   (4 × 77.1 h) + różnorodność (49.6k mówców CV, 120+ źródeł PS) podniosły OOD z 0.43 do ~0.76
   zanim dotknięto modelu.
2. **Label smoothing odwrócił wynik frozen vs unfrozen.** Bez LS pełny fine-tuning kolapsował na
   przepewniony skrót (E1: próg kalibracji = 1.000, softmax nasycony; AUC 0.733 < frozen 0.774).
   Z LS enkoder adaptuje reprezentacje pod kontrolą i wygrywa wyraźnie (0.878 vs 0.789).
   Dodatkowy dowód jakości E5: zanik wariancji między połowami mówców (0.865/0.884 vs 0.686/0.836
   u frozen).
3. **Uczciwa selekcja epoki po val_ood.** In-domain val nasyca się (AUC→1.0000 po ~2 epokach) i
   jest ślepy — E5 trafił dobrą epokę przypadkiem. E6 wybiera epokę po `val_ood` AUC (rozłączni
   mówcy względem `test`): zapisuje się wielokrotnie, łapie szczyt OOD (epoka 7) i daje wyższy,
   powtarzalny wynik (test 0.904 vs 0.884). Koszt: `val_ood` jest „widziany" przez selekcję, więc
   nagłówek raportujemy na rozłącznym `test`.
4. **Overconfidence to problem progu/kalibracji, nie rankingu.** Przy argmax@0.5 model „wypuszcza"
   spoofy (Spoof Acc 0.03–0.32) mimo wysokiego AUC. Dwa narzędzia (oba dopasowane na `val_ood`,
   raportowane na `test`): (a) próg Youden J — naprawia punkt pracy; (b) temperature scaling
   (skalar `T`) — prostuje nasycony softmax: **AUC/EER bez zmian** (transformacja monotoniczna),
   ale próg ląduje w zdrowym zakresie (E6: 0.978→0.742) i prawdopodobieństwa stają się sensowne.
   **Tradeoff AUC↔kalibracja:** najlepszy ranker jest najbardziej przepewniony — E6 (epoka 7,
   AUC 0.904) ma `T=3.59`, ECE 0.238→0.156 (wciąż wysokie); model z epoki 2 kalibrował się do
   ECE 0.031. Więcej epok → lepszy AUC, gorsza kalibracja, której jeden skalar nie naprawi w pełni.
5. **Caveaty (do uczciwej dyskusji):**
   - kalibracja częściowa (ECE 0.156 po temperaturze) — model przepewniony mimo label smoothing;
   - duża wariancja per-mówca w In-the-Wild (E4: połowy 0.686 vs 0.836) — pełna kwantyfikacja:
     k-fold po mówcach (AUC ± std);
   - operating point po kalibracji bywa przesunięty (E6: Bona 0.66 / Spoof 0.91), bo `val_ood`
     to trudniejsza połowa mówców — dlatego nagłówkiem są AUC/EER (threshold-free, odporne);
   - wyniki z pojedynczych biegów (bez powtórzeń z różnymi seedami);
   - EER ~17% to wynik badawczy, nie wdrożeniowy (produkt wymaga EER < 5–10%).
6. **Kontekst:** In-the-Wild to trudny benchmark (publikowane modele często 0.6–0.8 AUC);
   **0.904** wyraźnie przekracza wynik referencyjny z diagnozy (0.755, frozen base na małym zbiorze).

### Możliwe rozszerzenia (poza zakresem)

dodanie CodecFake/DFADD jako spoof (foldery przygotowane, puste); k-fold po mówcach (AUC ± std);
głębsza kalibracja (isotonic/Platt — choć i tak nie ruszą AUC); architektury dedykowane
(AASIST/RawNet2). Temperature scaling jest już zintegrowany (sekcja 2, pkt 8).

---

## 6. Reprodukcja — mapa skryptów i artefaktów

| krok | skrypt / artefakt | wynik |
|---|---|---|
| selekcja Common Voice (150 h, round-robin po mówcach) | `Code/other/selectCommonVoiceSubset.py` | `V2/raw/bonafide/CommonVoice/spkNNNNN/` + `V2/commonvoice_selection.csv` |
| pobranie People's Speech (limit rozmiaru i źródła) | `Code/other/huggingface_download.py` (`download_peoples_speech_subset`) | `V2/raw/bonafide/PeoplesSpeech/<źródło>/` + manifest |
| kompresja HD Voice (idempotentna) | `Code/audio/compressDataset.py` | `V2/compressed/**/*.wav` |
| budowa splitu (balans po czasie, val_ood/test po mówcach) | `Code/other/createBalancedOodSplit.py` (seed 42) | `V2/dataset_split_balanced_ood.csv` (wersja 5 w `projectMethods.get_dataset_split_csv`) |
| trening (selekcja po val_ood, label smoothing, early stopping) | `Code/model_training/train.py` + `detector.py` (`train(select_loader=val_ood, patience=3)`) | **`Models/wav2vec2_300m_not_frozen_V2_balanced_ood_v2_2.pt` (E6, FINALNY)**; `..._not_frozen_V2_balanced_ood.pt` (E5); `..._frozen_...pt` (E4) |
| ewaluacja OOD + kalibracja progu + temperature scaling | `Code/model_testing/evalOodFromCsv.py` | tabela wyników (sekcja 4) |

Czasy: trening frozen ~29.5 min/epoka, unfrozen ~45.5 min/epoka (RTX 4080 SUPER, batch 16);
ewaluacja OOD ~2.5 min. E6 = unfrozen, 10 epok z early stoppingiem, best = epoka 7 (val_ood AUC 0.876).
