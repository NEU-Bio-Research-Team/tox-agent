"""The ToxAgent comparison study (RETHINK §5.4, backlog Wave 5).

Runs the same pre-registered investigation cases through ToxAgent arms (the
current product, the case-based investigator and its skill ablations), a
predictor-only template, and general platforms (OpenAI, Anthropic, Google
models, bare and with the predictor snapshot); logs every prompt, output,
resolved model id, timing and product trace; and produces a blinded packet for
the chemistry lab that grades it. Nothing here scores quality: the lab does.
"""
