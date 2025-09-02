---
title: PersonaMate Pro (OAuth + Simplified UI)
emoji: "🤖"
colorFrom: "purple"
colorTo: "blue"
sdk: "gradio"
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

This Space uses FastAPI to serve the Gradio UI mounted at `/`.  
The entrypoint is `app.py`, which defines both the API server and the Gradio interface.  
Dependencies are listed in `requirements.txt`.
