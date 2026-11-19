# ASTRA_Glia

*Bio-inspirierter KI-Agent mit laufendem Spiking Neural Network · Bio-inspired AI agent with a continuously running spiking neural network*

[🇩🇪 Deutsch](#was-es-kann) · [🇬🇧 English](#what-it-does)

## Was es kann

ASTRA_Glia ist ein bio-inspirierter KI-Agent, der ein adaptives Izhikevich-Spiking-Netz mit Astrozyten-Modulation (drei Calcium-Kanäle für Gain, Homöostase und Konsolidierungs-Gate) sowie einer zweistufigen Stammzell-Reparatur mit einem Sprachmodell als entscheidendem Kortex koppelt. Er beantwortet Fragen wahlweise direkt, aus dem Vektor-Gedächtnis (ChromaDB-Hippocampus mit RAG) oder über schlüssellose Live-Tools für Wetter, Wikipedia und Websuche, deren Ergebnisse automatisch zu einer Antwort synthetisiert und als Wissens-Episode abgelegt werden. Nur geerdete Antworten werden langfristig konsolidiert, und ein zweistufiger RAG-Short-Circuit beantwortet gut getroffene Fachfragen direkt aus frischer Erinnerung.

Nutzereingaben lösen im SNN einen Salienz-Reiz aus. Das Netz läuft währenddessen ungestört weiter und meldet Konsolidierungen ans Gateway.

## What it does

ASTRA_Glia is a bio-inspired AI agent that couples an adaptive Izhikevich spiking network — modulated by astrocytes across three calcium channels for gain, homeostasis and consolidation gating, plus a two-stage stem-cell repair mechanism — with a language model acting as its decision-making cortex. It answers questions either directly, from vector memory (a ChromaDB hippocampus with RAG), or via keyless live tools for weather, Wikipedia and web search, whose results are automatically synthesized into an answer and stored as a knowledge episode. Only grounded answers are consolidated into long-term memory, and a two-tier RAG short-circuit answers well-matched factual questions straight from fresh recall.

*The rest of this README is in German. Quick start: create a venv, install PyTorch for your GPU, run `pip install -r requirements.txt`, put a GGUF model into `checkpoints/llm/`, then run `python main_chat.py`.*

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
