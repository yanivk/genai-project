"""Streamlit UI package.

    streamlit_main  The app itself.
    utils           Rendering and session helpers.

UI only. This layer collects input, calls ``app/``, and renders the result. No
prompts, no SQL, no retrieval logic lives here — if you are tempted to add some,
it belongs in ``app/modules/``. See CLAUDE.md section 2, rule 4.
"""
