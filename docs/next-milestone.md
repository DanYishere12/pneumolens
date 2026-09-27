# Portfolio milestone status

The project is ready for local portfolio review and packaging. The custom inference app runs at http://127.0.0.1:8502/ with `python serve.py`. A clearly labeled recorded demo runs at http://127.0.0.1:8503/ with `python3 demo.py`, using only the standard library and included assets.

Completed: responsive viewer, synchronized comparison/zoom/pan, opacity and class controls, uploads in the full app, annotated PNG exports, all 624 live test cases, a five-scan recorded gallery, clickable confusion matrix, two preserved evaluated models, parameter checks, artifact-integrity tests, reviewer-facing README, reproduction guide, attribution and a 90-second presentation walkthrough.

Verification: 16 Python tests and six JavaScript tests pass. Actual mobile/desktop screens, divider dragging, pan/zoom, matrix navigation and downloaded export were checked. See `docs/verification.md` for evidence and tool limitations.

The model result remains 78.8% test accuracy with 131 false positives and one false negative; Brier score worsened after fine-tuning. Test reuse is disclosed. Future research should prioritize independent-source evaluation and development-set artifact/error analysis. No further training was performed for this packaging pass.

Next publication step: choose a repository/hosting destination. No public repository or site has been published. The source archive excludes full data, trained weights, virtual environments and temporary files; it includes the small real-output demo. Neural Canvas and Chroma Lab were not changed.
