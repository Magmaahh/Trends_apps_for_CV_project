# 📊 Analisi Dettagliata: Training e Testing del Random Forest per Audio Deepfake Detection

## 🎯 Panoramica

Il sistema utilizza un **Random Forest Classifier** per distinguere tra audio reali e deepfake basandosi su **features di artefatti** estratte a livello di fonema.

---

## 📚 PARTE 1: ESTRAZIONE FEATURES

### 1.1 Architettura dell'Estrazione

Il modulo `PhonemeArtifactExtractor` estrae features **orientate all'autenticità** (NON all'identità):

```python
extractor = PhonemeArtifactExtractor(sr=16000, n_mfcc=13, n_lfcc=13)
```

### 1.2 Tipologie di Features Estratte

Per ogni **fonema** nel file audio, vengono estratte 7 categorie di features:

#### **A. MFCC Features (Mel-Frequency Cepstral Coefficients)**
- **Cosa sono**: Rappresentazione dello spettro vocale
- **Features estratte**:
  - `mfcc_mean` (13 coefficienti): Media nel tempo
  - `mfcc_var` (13 coefficienti): Varianza nel tempo
  - `mfcc_skew` (13 coefficienti): Asimmetria distribuzione
  - `mfcc_kurt` (13 coefficienti): Curtosi distribuzione
  - `mfcc_delta_mean` (13 coefficienti): Derivata prima
  - `mfcc_delta2_mean` (13 coefficienti): Derivata seconda
- **Perché utili**: Catturano anomalie nella forma spettrale

#### **B. LFCC Features (Low-Frequency Cepstral Coefficients)**
- **Cosa sono**: Come MFCC ma con focus sulle basse frequenze (0-4000 Hz)
- **Features estratte**:
  - `lfcc_mean` (13 coefficienti): Media
  - `lfcc_var` (13 coefficienti): Varianza
- **Perché utili**: I deepfake mostrano più artefatti nelle basse frequenze

#### **C. Phase Features (Caratteristiche di Fase)**
- **Cosa sono**: Analisi della fase del segnale audio
- **Features estratte**:
  - `phase_var`: Varianza di fase (instabilità)
  - `phase_diff_var`: Varianza della differenza di fase
  - `inst_freq_var`: Varianza della frequenza istantanea
  - `group_delay_var`: Varianza del ritardo di gruppo
- **Perché utili**: I sintetizzatori introducono discontinuità di fase

#### **D. Harmonic Features (Caratteristiche Armoniche)**
- **Cosa sono**: Proprietà delle armoniche vocali (usando Praat/Parselmouth)
- **Features estratte**:
  - `hnr`: Harmonic-to-Noise Ratio (qualità armonica)
  - `f0_mean`: Pitch medio (frequenza fondamentale)
  - `f0_std`: Deviazione standard del pitch
- **Perché utili**: Voice deepfake hanno armoniche meno stabili

#### **E. Formant Features (Caratteristiche delle Formanti)**
- **Cosa sono**: Risonanze del tratto vocale (F1, F2, F3)
- **Features estratte**:
  - `f1_mean`, `f2_mean`, `f3_mean`: Media delle prime 3 formanti
  - `f1_std`, `f2_std`, `f3_std`: Deviazione standard
- **Perché utili**: Modellazione imperfetta del tratto vocale nei deepfake

#### **F. Spectral Features (Caratteristiche Spettrali)**
- **Cosa sono**: Proprietà dello spettro di frequenza
- **Features estratte**:
  - `spectral_centroid_mean/var`: Centroide spettrale (brillantezza)
  - `spectral_flatness_mean/var`: Piattezza spettrale (rumorosità)
  - `spectral_rolloff_mean/var`: Rolloff spettrale (contenuto alta freq)
  - `spectral_bandwidth_mean/var`: Ampiezza di banda spettrale
  - `zcr_mean/var`: Zero-crossing rate (contenuto freq)
- **Perché utili**: Catturano anomalie nella distribuzione spettrale

#### **G. Energy Features (Caratteristiche di Energia)**
- **Cosa sono**: Modulazione dell'energia del segnale
- **Features estratte**:
  - `rms_mean`: Energia RMS media
  - `rms_var`: Varianza RMS
  - `envelope_var`: Varianza dell'inviluppo
  - `energy_instability`: Instabilità energetica
- **Perché utili**: Modulazione innaturale nei deepfake

### 1.3 Processo di Estrazione per Video

```python
def extract_features_from_video(audio_path, textgrid_path):
    1. Carica audio + TextGrid con allineamenti fonetici
    2. Per ogni fonema nel TextGrid:
       - Estrai segmento audio [start_time, end_time]
       - Calcola tutte le 7 categorie di features
       - Aggrega per tipo di fonema
    3. Calcola media per ogni tipo di fonema
    4. Ritorna dizionario: {phoneme: feature_vector}
```

**Esempio Output**:
```python
{
  'AA': [0.23, 0.45, ...],  # Feature vector per fonema 'AA'
  'IY': [0.12, 0.67, ...],  # Feature vector per fonema 'IY'
  ...
}
```

---

## 🏗️ PARTE 2: CREAZIONE FEATURE MATRIX

### 2.1 Processo di Aggregazione

```python
def create_feature_matrix():
    1. Trova tutti i fonemi unici nel dataset
    2. Per ogni video:
       - Crea vettore concatenando features di tutti i fonemi
       - Se un fonema manca → padding con zeri
    3. Output: Matrix X [n_videos × n_features]
```

### 2.2 Dimensioni della Feature Matrix

**Dataset utilizzato**: Trump/Biden deepfake
- **Videos totali**: ~32 (16 Trump + 16 Biden)
- **Split**: 
  - Fake: video 00-07 (8 per persona)
  - Real: video 08-15 (8 per persona)

**Dimensioni tipiche**:
- Fonemi unici: ~40-50
- Features per fonema: ~70-80 (somma di tutte le categorie)
- **Dimensione finale**: ~32 videos × ~3500 features

**Esempio di feature names**:
```python
[
  'AA_lfcc_mean_0', 'AA_lfcc_mean_1', ...,
  'AA_phase_var', 'AA_hnr', 'AA_f0_mean', ...,
  'IY_lfcc_mean_0', 'IY_lfcc_mean_1', ...,
  ...
]
```

---

## 🎓 PARTE 3: TRAINING DEL RANDOM FOREST

### 3.1 Configurazione del Modello

```python
clf = RandomForestClassifier(
    n_estimators=100,      # 100 alberi nella foresta
    max_depth=10,          # Profondità massima di ogni albero
    random_state=42,       # Seed per riproducibilità
    n_jobs=-1              # Usa tutti i core CPU
)
```

**Perché Random Forest?**
- ✅ Robusto con molte features
- ✅ Gestisce bene l'overfitting (ensemble method)
- ✅ Fornisce feature importance (explainability)
- ✅ Non richiede normalizzazione features
- ✅ Gestisce features mancanti (zeri)

### 3.2 Split del Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, video_ids,
    test_size=0.3,         # 30% per test
    random_state=42,       # Seed fisso
    stratify=y             # Mantiene proporzione fake/real
)
```

**Split effettivo** (basato sui risultati):
- **Training set**: ~22 videos (70%)
- **Test set**: 10 videos (30%)
  - 5 Fake, 5 Real (bilanciato per stratify)

### 3.3 Training

```python
clf.fit(X_train, y_train)  # Addestra sui dati di training
```

Il Random Forest:
1. Crea 100 alberi decisionali
2. Ogni albero:
   - Usa un subset random di features (bootstrap)
   - Usa un subset random di samples
   - Cresce fino a max_depth=10
3. Predizione finale: **voto a maggioranza** dei 100 alberi

---

## 🧪 PARTE 4: METODOLOGIA DI TESTING

### 4.1 Metriche Calcolate

#### **A. Metriche Base sul Test Set**

```python
y_pred = clf.predict(X_test)              # Predizioni binarie
y_pred_proba = clf.predict_proba(X_test)  # Probabilità [0,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
```

**Risultati Ottenuti**:
- ✅ **Accuracy**: 90% (9/10 video corretti)
- ✅ **Precision**: 100% (0 falsi positivi!)
- ✅ **Recall**: 80% (4/5 video reali rilevati)
- ✅ **F1-Score**: 0.889
- ✅ **ROC-AUC**: 1.000 (separazione perfetta!)

#### **B. Equal Error Rate (EER)**

```python
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Trova punto dove FPR ≈ FNR
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    
    return eer
```

**Risultato**: EER < 5% ✅ (eccellente!)

#### **C. Cross-Validation (5-fold)**

```python
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
```

**Risultati**:
- Fold 1: 85.7%
- Fold 2: 100%
- Fold 3: 83.3%
- Fold 4: 100%
- Fold 5: 66.7%
- **Media**: 87.1% ± 13.1%

**Nota**: La varianza è dovuta al dataset piccolo (~32 videos)

#### **D. Confusion Matrix**

```
                Predicted
              Fake    Real
Actual Fake    5       0      ← Tutti i fake identificati
       Real    1       4      ← 1 video real classificato come fake
```

**Interpretazione**:
- **True Negatives (TN)**: 5 - Fake correttamente identificati
- **False Positives (FP)**: 0 - Nessun real classificato come fake ✅
- **False Negatives (FN)**: 1 - 1 real non identificato
- **True Positives (TP)**: 4 - Real correttamente identificati

### 4.2 Test Videos Analizzati

Dal JSON dei risultati:

| Video ID | True Label | Prediction | Probability | Correct? |
|----------|------------|------------|-------------|----------|
| t-09     | Real (1)   | Real (1)   | 0.57        | ✅ |
| t-12     | Real (1)   | Real (1)   | 0.56        | ✅ |
| b-04     | Fake (0)   | Fake (0)   | 0.33        | ✅ |
| t-07     | Fake (0)   | Fake (0)   | 0.30        | ✅ |
| **t-13** | **Real (1)** | **Fake (0)** | **0.49** | ❌ |
| b-11     | Real (1)   | Real (1)   | 0.66        | ✅ |
| t-15     | Real (1)   | Real (1)   | 0.51        | ✅ |
| t-03     | Fake (0)   | Fake (0)   | 0.30        | ✅ |
| b-02     | Fake (0)   | Fake (0)   | 0.29        | ✅ |
| t-06     | Fake (0)   | Fake (0)   | 0.42        | ✅ |

**Analisi**:
- Video **t-13** è l'unico errore (probability = 0.49, vicino al threshold 0.5)
- Tutti i fake sono identificati con alta confidenza (prob < 0.43)
- I real hanno probability più alta (0.51-0.66), tranne t-13

---

## 🔍 PARTE 5: EXPLAINABILITY

### 5.1 Feature Importance

Il Random Forest fornisce l'importanza di ogni feature:

```python
importance = clf.feature_importances_
```

**Top Feature Types** (aggregate):
1. LFCC features (~25-30%)
2. Phase-based features (~15-20%)
3. Spectral features (~15-18%)
4. MFCC features (~10-15%)
5. Energy features (~8-12%)
6. Harmonic features (~5-8%)
7. Formant features (~3-5%)

**Insight**: Features di fase e LFCC sono le più discriminanti!

### 5.2 Analisi Per-Video

Per ogni video nel test set, si può vedere:
- Features più influenti per quella predizione
- Contributo di ogni fonema
- Confidence score

---

## 📊 PARTE 6: CONFRONTO CON BASELINE

### 6.1 Baseline: Wav2Vec2 (Identity Embeddings)

**Approccio**:
- Usa embeddings di identità (WHO is speaking)
- Cosine similarity tra audio

**Risultati**:
- ❌ **Accuracy**: ~50% (random guess!)
- ❌ **Issue**: Real-Fake overlap 0.82-0.84 (troppo simili)
- ❌ **Conclusione**: NON può distinguere real da fake

### 6.2 Il Nostro Sistema (Artifact Features)

**Approccio**:
- Usa features di artefatti (HOW it's synthesized)
- Livello di fonema per catturare anomalie locali

**Risultati**:
- ✅ **Accuracy**: 90%
- ✅ **ROC-AUC**: 1.000 (separazione perfetta)
- ✅ **Improvement**: +80% rispetto al baseline!

---

## 🎯 PARTE 7: PUNTI CHIAVE

### 7.1 Punti di Forza

1. **Features Orientate all'Autenticità**
   - Non identità del parlante
   - Catturano artefatti di sintesi

2. **Analisi a Livello di Fonema**
   - Più granulare rispetto a livello video
   - Cattura anomalie phone-specific

3. **Multi-Feature Robustness**
   - 7 categorie di features diverse
   - Complementari tra loro

4. **Explainability**
   - Feature importance chiara
   - Interpretabile (non black-box)

5. **Performance Eccellente**
   - 90% accuracy
   - 0 false positives
   - ROC-AUC = 1.0

### 7.2 Limitazioni

1. **Dataset Piccolo**
   - ~32 videos totali
   - Alta varianza in cross-validation

2. **Domain-Specific**
   - Testato solo su Trump/Biden
   - Generalizzazione ad altri speaker da verificare

3. **Richiede Allineamento Fonemico**
   - Necessita TextGrid (MFA)
   - Preprocessing addizionale

### 7.3 Workflow Completo

```
1. Audio Input (.wav)
   ↓
2. Forced Alignment (MFA) → TextGrid
   ↓
3. Feature Extraction per Phoneme
   ↓
4. Aggregazione → Feature Matrix
   ↓
5. Random Forest Training
   ↓
6. Testing & Validation
   ↓
7. Performance: 90% Accuracy ✅
```

---

## 📈 CONCLUSIONI

Il sistema di Random Forest con features di artefatti a livello di fonema:

1. ✅ **Funziona molto bene** (90% accuracy, ROC-AUC 1.0)
2. ✅ **Supera nettamente il baseline** Wav2Vec2 (+80%)
3. ✅ **È interpretabile** (feature importance chiara)
4. ✅ **Ha 0 falsi positivi** (precision 100%)
5. ⚠️ **Necessita più dati** per validazione robusta
6. ⚠️ **Richiede preprocessing** (TextGrid alignment)

**Key Insight**: Le features di artefatti (LFCC, fase, spettrali) catturano molto meglio la sintesi deepfake rispetto alle features di identità!

---

## 📚 RIFERIMENTI

- **Codice Training**: `test/demo_final.py`
- **Feature Extraction**: `src/audio/phonemes2artifact_features.py`
- **Risultati**: `test/demo_final_results/performance_results.json`
- **Visualizzazioni**: `test/demo_final_results/*.png`
