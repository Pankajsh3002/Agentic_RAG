CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    TEXT NOT NULL,
    name          TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_workspaces_session ON workspaces (session_id);

CREATE TABLE IF NOT EXISTS documents (
    doc_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    content_hash  TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    stage         TEXT,
    error         TEXT,
    error_detail  TEXT,
    markdown      TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents (workspace_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_ws_hash ON documents (workspace_id, content_hash);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id        UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    heading_path  TEXT,
    content       TEXT NOT NULL,
    embedding     halfvec(3072),
    UNIQUE (doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON chunks (workspace_id);


CREATE TABLE IF NOT EXISTS document_branches (
    doc_id        UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    branch        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'running', 'done', 'failed')),
    error         TEXT,
    error_detail  TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, branch)
);

CREATE TABLE IF NOT EXISTS summaries (
    doc_id        UUID PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    summary       TEXT NOT NULL,
    model         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_summaries_workspace ON summaries (workspace_id);

ALTER TABLE document_branches ADD COLUMN IF NOT EXISTS raw_output JSONB;

CREATE TABLE IF NOT EXISTS events (
    event_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id    UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    description     TEXT,
    event_type      TEXT NOT NULL DEFAULT 'other',
    date_start      DATE,
    date_end        DATE,
    is_ongoing      BOOLEAN NOT NULL DEFAULT false,
    date_precision  TEXT NOT NULL DEFAULT 'unknown',
    date_text       TEXT,
    quote           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_ws_date ON events (workspace_id, date_start);
CREATE INDEX IF NOT EXISTS idx_events_doc ON events (doc_id);

CREATE TABLE IF NOT EXISTS entities (
    entity_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id        UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    entity_key    TEXT NOT NULL,
    entity_type   TEXT NOT NULL DEFAULT 'other',
    description   TEXT,
    UNIQUE (doc_id, entity_key)
);
CREATE INDEX IF NOT EXISTS idx_entities_ws_key ON entities (workspace_id, entity_key);

CREATE TABLE IF NOT EXISTS relations (
    relation_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id        UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source_id     BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id     BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation      TEXT NOT NULL,
    quote         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_relations_ws ON relations (workspace_id);
CREATE INDEX IF NOT EXISTS idx_relations_doc ON relations (doc_id);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations (source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations (target_id);
ALTER TABLE document_branches ADD COLUMN IF NOT EXISTS raw_output JSONB;

CREATE TABLE IF NOT EXISTS events (
    event_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id    UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    description     TEXT,
    event_type      TEXT NOT NULL DEFAULT 'other',
    date_start      DATE,
    date_end        DATE,
    is_ongoing      BOOLEAN NOT NULL DEFAULT false,
    date_precision  TEXT NOT NULL DEFAULT 'unknown',
    date_text       TEXT,
    quote           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_ws_date ON events (workspace_id, date_start);
CREATE INDEX IF NOT EXISTS idx_events_doc ON events (doc_id);

CREATE TABLE IF NOT EXISTS entities (
    entity_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id        UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    entity_key    TEXT NOT NULL,
    entity_type   TEXT NOT NULL DEFAULT 'other',
    description   TEXT,
    UNIQUE (doc_id, entity_key)
);
CREATE INDEX IF NOT EXISTS idx_entities_ws_key ON entities (workspace_id, entity_key);

CREATE TABLE IF NOT EXISTS relations (
    relation_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  UUID NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    doc_id        UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source_id     BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id     BIGINT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation      TEXT NOT NULL,
    quote         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_relations_ws ON relations (workspace_id);
CREATE INDEX IF NOT EXISTS idx_relations_doc ON relations (doc_id);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations (source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations (target_id);
