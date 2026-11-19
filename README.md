# ASTRA_Glia

Experimenteller Chat-Agent mit einem **dauerhaft laufenden Spiking Neural Network (SNN)**, das parallel zu einem lokalen Sprachmodell arbeitet.

- **SNN:** Izhikevich-Neuronen mit Astrozyten-Modulation (Calcium-Kanäle auf drei Zeitskalen) und Stammzell-Neurogenese, die ausgebrannte Neuronen ersetzt. Eine Drei-Faktor-Lernregel koppelt Konsolidierungsereignisse an die Gewichte.
- **Kortex:** lokales LLM (GGUF, via llama.cpp) mit Tool-Loop. Es entscheidet pro Frage, ob es direkt antwortet oder ein Tool nutzt.
- **Gedächtnis:** Hippocampus-Modul mit Kurz- und Langzeitspeicher (ChromaDB, mehrsprachige Embeddings) für Wissen aus früheren Tool-Aufrufen.
- **Tools:** Wetter (Open-Meteo), Websuche (DuckDuckGo), Wikipedia (deutsch). Alle live und ohne API-Key.

Nutzereingaben lösen im SNN einen Salienz-Reiz aus. Das SNN läuft währenddessen ungestört weiter und meldet Konsolidierungen ans Gateway.

## Voraussetzungen

- Python 3.11
- Eine NVIDIA-GPU ist empfohlen, das SNN selbst läuft auf der CPU
- Ein GGUF-Sprachmodell (siehe unten)

## Installation (Windows / PowerShell)

```powershell
git clone https://github.com/Batwinx/ASTRA_Glia.git
cd ASTRA_Glia
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**PyTorch zuerst passend zur GPU installieren** (Anleitung auf pytorch.org). Karten der RTX-50-Serie brauchen einen Build mit CUDA 12.8 oder neuer. Danach:

```powershell
pip install -r requirements.txt
```

## Modell

Das Sprachmodell liegt **nicht** im Repository. Verwendet wird NVIDIA Nemotron 3.5 Bolt E4B als GGUF (Q4_K_M):

https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB

Lade die GGUF-Datei herunter und lege sie nach:

```
checkpoints/llm/
```

Der erwartete Dateiname steht in `main_chat.py` (`CORTEX_MODEL`, standardmäßig `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`).

Das Modell steht unter der Lizenz OpenMDW-1.1, die unabhängig von der Lizenz dieses Projekts gilt.

Beim ersten Start wird außerdem das Embedding-Modell `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` heruntergeladen.

## Starten

```powershell
python main_chat.py
```

Befehle im Chat: `/quit`, `/stats`, `/tools`, `/memory`

Das Gedächtnis wird beim Start im Ordner `memory/` angelegt und gehört nicht ins Repository.

## Projektstruktur

```
main_chat.py        Chat-Agent (SNN-Thread + Gateway)
core/               Cortex-Gateway (LLM, Tool-Loop) und Hippocampus
src/neurons/        Izhikevich-Neuronen
src/glia/           Astrozyten-Modulator
src/stemcell/       Neurogenese
src/tools/          Router, Wetter, Suche, Wikipedia
src/knowledge/      Grundlagenwissen
scripts/            Hilfsskripte (Wissen einlesen, Smoke-Test)
tests/              Tests
config/             Hardware-Profile und Astrozyten-Parameter
```

## Lizenz

[GNU AGPL-3.0](LICENSE). Wer den Code verwendet oder ändert, muss sein Projekt unter derselben Lizenz offenlegen, auch wenn es nur als Online-Dienst läuft.
