"""Application modules.

    main_agent   Orchestrator, turn router, action literals.
    advisors     Exit, Scheduling and Info advisors + their output schemas.
    database     SQLAlchemy engine, Schedule queries, SQLite seeder.
    embedding    PDF -> chunks -> Chroma, and the retrieval helper.
    finetuning   JSONL builder, job launcher, status poller.
    evaluation   Dataset loader, prediction runner, metrics.

Sub-packages are imported explicitly by the caller; nothing is re-exported here,
so that importing one module never drags in the whole dependency tree.
"""
