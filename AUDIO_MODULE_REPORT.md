# 🎵 Architettura del Modulo Audio: Dal Riconoscimento dell'Identità alla Detection degli Artefatti

## Introduzione

Il modulo audio del nostro sistema ha attraversato un'evoluzione progettuale interessante, passando da un approccio iniziale basato sul riconoscimento dell'identità vocale a un approccio completamente nuovo focalizzato sulla rilevazione di artefatti di sintesi. Questa transizione non è stata solo un cambio tecnico, ma un vero e proprio cambio di paradigma nel modo di affrontare il problema della deepfake detection.

---

## 🎭 Parte 1: L'Approccio Iniziale - Identità Vocale

### Come Funziona il Modulo Base

Il primo approccio che abbiamo implementato si basa su una domanda apparentemente sensata: "Se possiamo riconoscere l'identità di una persona dalla sua voce, non dovremmo essere in grado di capire se quella voce è autentica o sintetizzata?"

#### Il Cuore del Sistema: Wav2Vec2

Al centro di questo approccio c'è **Wav2Vec2**, un modello di deep learning sviluppato da Facebook AI. Immagina Wav2Vec2 come un esperto di voci umane che ha "ascoltato" 960 ore di parlato naturale e ha imparato a catturare l'essenza di come gli esseri umani parlano. Il modello trasforma ogni segmento audio in un vettore di 768 numeri - una sorta di "impronta digitale" vocale.

#### La Pipeline di Estrazione (`phonemes2emb.py`)

Il processo è elegante nella sua semplicità:

1. **Segmentazione Fonemica**: Prima di tutto, l'audio viene "tagliato" a livello di fonemi (i singoli suoni del parlato, come "AA", "IH", "S"). Questo lavoro preparatorio è fatto dal Montreal Forced Aligner (MFA), che genera un file TextGrid con tutti gli allineamenti temporali.

2. **Estrazione degli Embeddings**: Per ogni fonema, Wav2Vec2 genera un vettore di 768 dimensioni. Questi vettori catturano caratteristiche come il timbro, la qualità vocale, le caratteristiche articolatorie - tutto ciò che rende riconoscibile la voce di una persona.

3. **Aggregazione**: Visto che lo stesso fonema appare più volte in un discorso (ad esempio, il suono "AH" si ripete molte volte), facciamo una media di tutte le sue occorrenze per ottenere un rappresentazione stabile.

4. **Creazione della Signature**: Il risultato finale è una "firma vocale" (voice signature) salvata in un file `.npz`, che contiene un vettore rappresentativo per ogni tipo di fonema trovato nell'audio.

#### L'Idea di Base

L'idea è intuitiva: quando confrontiamo due audio dello stesso speaker, le loro signature dovrebbero essere molto simili (alta cosine similarity). Se un audio è un deepfake, dovrebbe avere una signature diversa da quella dell'audio originale. In teoria, suona perfetto.

### Il Problema Inatteso

Qui arriva il colpo di scena della nostra storia. Quando abbiamo testato questo approccio su un dataset di deepfake (video di Trump e Biden, metà reali e metà deepfake), ci siamo scontrati con una realtà sorprendente.

#### I Numeri della Delusione

Abbiamo calcolato la cosine similarity tra le signature audio:
- **Real vs Real**: ~0.84 (come ci aspettavamo, alta similarità)
- **Fake vs Fake**: ~0.82 (anche i deepfake tra loro sono simili)
- **Real vs Fake**: ~0.82-0.84 (e qui sta il problema!)

Le signature di audio reali e deepfake sono praticamente **indistinguibili**. La sovrapposizione è dell'82-84% - troppo alta per poter fare una classificazione affidabile. Con questi numeri, il nostro sistema avrebbe un'accuratezza vicina al 50%, equivalente a lanciare una moneta.

#### Perché Non Funziona?

La risposta sta nella natura stessa dei moderni sintetizzatori vocali. I sistemi di voice cloning moderni (come quelli usati per creare i nostri deepfake) sono stati **specificamente progettati** per replicare l'identità vocale. Il loro obiettivo è proprio ingannare sistemi come Wav2Vec2, che cercano di identificare chi sta parlando.

È come se provassimo a distinguere una banconota falsa da una vera guardando solo la foto della persona stampata sopra - i falsari hanno fatto un ottimo lavoro proprio su quello! Dobbiamo guardare altrove: le filigrane, la texture della carta, i micro-pattern che il falsario non ha potuto replicare perfettamente.

---

## 🔍 Parte 2: La Svolta - Artefatti di Sintesi

### Cambio di Filosofia

Di fronte a questo fallimento, abbiamo fatto un passo indietro e ripensato completamente l'approccio. Invece di chiedere "Chi sta parlando?", abbiamo iniziato a chiedere "**Come** è stato prodotto questo audio?"

La chiave è che, per quanto sofisticati, i sintetizzatori vocali lasciano sempre delle tracce - piccole imperfezioni, pattern innaturali, anomalie sottili che un orecchio umano non percepisce ma che un'analisi computazionale può rivelare.

### Il Nuovo Modulo: `phonemes2artifact_features.py`

#### Sette Dimensioni di Analisi

Invece di un singolo vettore di 768 dimensioni, ora estraiamo **sette categorie** diverse di features, ciascuna progettata per catturare un tipo specifico di artefatto:

**1. LFCC (Low-Frequency Cepstral Coefficients)**
- Analizziamo specificamente le basse frequenze (0-4000 Hz), dove i sintetizzatori tendono a lasciare più tracce
- Nel parlato naturale c'è un certo "disordine" fisiologico; nei deepfake le cose sono spesso troppo "pulite" o hanno pattern riconoscibili

**2. Features di Fase**
- La fase è il "segreto nascosto" del segnale audio
- L'orecchio umano è quasi cieco alla fase, quindi i sintetizzatori spesso la trascurano
- Noi cerchiamo discontinuità, irregolarità o pattern troppo regolari che tradiscono la sintesi artificiale

**3. Caratteristiche Armoniche**
- Nel parlato naturale, le armoniche (i multipli della frequenza fondamentale) seguono leggi fisiche precise
- I deepfake possono avere armoniche troppo stabili o instabili in modi innaturali
- Misuriamo il rapporto Armonico-Rumore (HNR) e la variabilità del pitch

**4. Formanti**
- Le formanti sono le "impronte digitali" del tratto vocale
- In un audio naturale, seguono i vincoli fisiologici della bocca e della gola
- I sintetizzatori possono generare formanti che violano questi vincoli o che sono troppo stabili

**5. Caratteristiche Spettrali**
- Analizziamo la distribuzione dell'energia nelle frequenze
- Cerchiamo anomalie nel "centroide spettrale" (dove si concentra l'energia)
- Misuriamo la "piattezza" dello spettro e altri indicatori di naturalezza

**6. Modulazione Energetica**
- Nel parlato naturale, l'energia varia in modi specifici legati alla respirazione e all'articolazione
- I deepfake possono avere modulazioni troppo regolari o con discontinuità innaturali

**7. MFCC Estesi**
- Usiamo anche gli MFCC classici, ma non per identificare il parlante
- Ci concentriamo sulla loro **variabilità temporale** all'interno di ogni fonema
- Cerchiamo pattern di variazione anomali

#### Il Vantaggio dell'Analisi Per-Phoneme

Una caratteristica cruciale del nostro approccio è che analizziamo ogni **tipo di fonema** separatamente. Perché? Perché i sintetizzatori possono essere bravi con certi suoni ma meno con altri.

Ad esempio, le consonanti occlusive come /p/, /t/, /k/ richiedono transizioni rapidissime e precise - difficili da sintetizzare perfettamente. Le vocali richiedono formanti stabili per decine di millisecondi - un altro tipo di sfida. Analizzando separatamente ogni fonema, catturiamo queste debolezze specifiche.

### I Risultati del Cambio

Quando abbiamo addestrato un Random Forest con queste nuove features, i risultati sono stati drammaticamente diversi:

```
BASELINE (Wav2Vec2 Identity Embeddings):
  Accuracy:  ~50%  ❌
  ROC-AUC:   ~0.50 (random guess)
  Problema:  Audio reali e fake sono indistinguibili

NUOVO SISTEMA (Artifact Features):
  Accuracy:  90%   ✅  (+80% miglioramento!)
  Precision: 100%  ✅  (0 falsi positivi)
  Recall:    80%   ✅
  ROC-AUC:   1.00  ✅  (separazione perfetta)
  EER:       <5%   ✅
```

La differenza è netta: siamo passati da un sistema che "tira a indovinare" a uno che distingue correttamente 9 video su 10, senza **mai** classificare erroneamente un video reale come fake (precision al 100%).

---

## 💡 Perché Funziona: Le Motivazioni Profonde

### 1. Combattere il Nemico Giusto

Il punto fondamentale è che non stiamo più combattendo contro l'obiettivo principale del sintetizzatore (replicare l'identità), ma contro i suoi "effetti collaterali" - le imperfezioni inevitabili del processo di sintesi.

### 2. Multi-Dimensionalità

Usando sette categorie di features diverse, ci assicuriamo che anche se un sintetizzatore è molto bravo a nascondere artefatti in una dimensione (ad esempio, nella fase), probabilmente ne lascerà in un'altra (ad esempio, nelle formanti o nell'energia).

### 3. Granularità Fonemica

L'analisi per-phoneme ci dà una risoluzione molto più fine. Invece di una media globale che potrebbe nascondere anomalie locali, otteniamo un profilo dettagliato dove le debolezze del sintetizzatore emergono.

### 4. Robustezza Statistica

Aggregando multiple occorrenze dello stesso fonema, riduciamo il rumore e ci concentriamo sulle caratteristiche intrinseche. Se un pattern appare consistentemente attraverso tutte le occorrenze di un fonema, è probabilmente un artefatto sistematico, non rumore casuale.

---

## 🎯 Conclusioni e Lezioni Apprese

Questa evoluzione del modulo audio ci ha insegnato una lezione importante: **il problema determina la soluzione**. 

L'approccio basato sull'identità vocale è eccellente per speaker verification - distinguere tra persone diverse. Ma per la deepfake detection, serve un approccio completamente diverso che guardi alle "impronte digitali" del processo di sintesi.

Il successo del nostro sistema artifact-based dimostra che, nel panorama attuale dei deepfake, la chiave non è "riconoscere la persona", ma "riconoscere la macchina". E le macchine, per quanto sofisticate, lasciano sempre delle tracce.

### Architettura Finale

Il nostro sistema finale mantiene entrambi i moduli:
- `phonemes2emb.py`: Per future esplorazioni e per confronti baseline
- `phonemes2artifact_features.py`: Come sistema principale per la detection

Entrambi condividono il livello di segmentazione fonemica (TextGrid da MFA) ma divergono completamente nella fase di feature extraction, riflettendo le due filosofie radicalmente diverse del "Chi?" vs "Come?".

---

## 📊 Schema Riassuntivo

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUZIONE DEL MODULO AUDIO                  │
└─────────────────────────────────────────────────────────────────┘

APPROCCIO BASELINE                    APPROCCIO FINALE
(phonemes2emb.py)                    (phonemes2artifact_features.py)
      ↓                                        ↓
Domanda: "Chi?"                       Domanda: "Come?"
      ↓                                        ↓
Wav2Vec2 Embeddings                   7 Categorie di Features:
(768-dim per phoneme)                 - LFCC (low-freq focus)
                                      - Phase (discontinuities)
                                      - Harmonics (stability)
                                      - Formants (physiology)
                                      - Spectral (distribution)
                                      - Energy (modulation)
                                      - MFCC (variability)
      ↓                                        ↓
Cosine Similarity                     Random Forest Classifier
      ↓                                        ↓
Accuracy: ~50% ❌                      Accuracy: 90% ✅
ROC-AUC:  ~0.50                       ROC-AUC:  1.00
                                      Precision: 100%

     PERCHÉ NON FUNZIONA?                  PERCHÉ FUNZIONA?
     ↓                                     ↓
I deepfake REPLICANO                  I deepfake LASCIANO
l'identità vocale                     artefatti di sintesi
(è il loro obiettivo!)                (inevitabili!)
```

---

## 🚀 Impatto sul Sistema Finale

Questo modulo artifact-based non è solo un componente autonomo - è diventato il **pilastro centrale** del nostro sistema di deepfake detection audio. Le sue features vengono utilizzate:

1. **Stand-alone**: Per classification audio pura (90% accuracy)
2. **Multimodal Fusion**: Combinate con features video per detection ancora più robusta
3. **Explainability**: Le feature importance ci dicono PERCHÉ un audio è classificato come fake

Il successo di questo approccio ha validato la nostra intuizione fondamentale: nel mondo dei deepfake moderni, non cercare di riconoscere la persona - cerca di riconoscere la macchina.
