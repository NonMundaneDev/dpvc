# Introduction

This repository provides a library for defining differentially private speaker anonymization systems using existing voice control models. The approach works for any voice control system that separates utterance information into constant-length speaker information (e.g. a speaker embedding) and time-varying content information (e.g. semantic features).

## Overview

The DPVC library provides a framework for performing differentially private speaker anonymization. The library's architecture is designed around a standardized "wrapper" for a voice control system that exposes methods for extracting speaker embeddings and performing inference. The library provides methods for performing the anonymization using a wrapped system, and also utilities for training the autoencoder used during anonymization.

## Installation

Install the library by cloning the repository and then running:

```
pip install .
```

## Current Research Notes

The active controllable-speaker research line is maintained in the research
fork, not upstream `main`. For the latest research interpretation, start with:

- [Metric and Collapse Guide](metric_collapse_guide.md)
- `EVIDENCE_DEMO_PACKET.md` at the repository root
- `results/listening_evidence_demo_index.html` for local listening review
