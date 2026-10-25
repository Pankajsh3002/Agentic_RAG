from . import chunks, events, graph, summary

BRANCHES = {
    "chunks": chunks.run,
    "summary": summary.run,
    "events": events.run,
    "graph": graph.run,
}

# If one of these fails, the whole document is marked failed.
# Everything else is optional: the document stays "done" and the branch shows as failed.
ESSENTIAL = {"chunks"}