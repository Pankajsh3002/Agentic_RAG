import logging
import os
import re

from Pipeline.extraction import (
    MAX_TOTAL_CHARS, RULES, ask_json, clean_text, norm, normalize_key, quote_in_doc, split_windows,
)
from db.database import save_branch_output, save_graph

logger = logging.getLogger(__name__)

MAX_ENTITIES = 200
MAX_RELATIONS = 300
MAX_TOKENS = 10000   # lower this if your model rejects the limit

ENTITY_TYPES = {"person", "organization", "location", "technology",
                "document", "money", "concept", "other"}

PROMPT = """You extract a knowledge graph (entities and relationships) from a document.
<<RULES>>

Return ONLY a JSON object with two keys, "entities" and "relations".

Each item in "entities" is an object with:
- "name": the entity's name as written in the document
- "type": exactly one of person, organization, location, technology, document, money, concept, other
- "description": one short line about it based only on the document, or null

Each item in "relations" is an object with:
- "source": the name of the source entity, copied exactly from an entity you listed
- "relation": a short snake_case verb phrase such as works_at, signed_by, located_in, uses, owns, pays, employs, party_to, issued_by, reports_to, part_of
- "target": the name of the target entity, copied exactly from an entity you listed
- "quote": one sentence or line copied from the document that states this link, character for character, as one contiguous span with no ellipses

Rules:
- Entities are specific named things: people, companies, places, products or technologies, agreements, amounts. Do not create entities for generic words.
- Only include relationships that the document states. Never infer or guess.
- At most 60 entities and 80 relations. If nothing qualifies, return {"entities": [], "relations": []}.

<document>
<<TEXT>>
</document>"""


def _clean_entity(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    name = clean_text(raw.get("name"), 120)
    key = normalize_key(name) if name else ""
    if not name or not key:
        return None
    entity_type = str(raw.get("type", "")).lower().strip()
    return {
        "name": name,
        "key": key,
        "type": entity_type if entity_type in ENTITY_TYPES else "other",
        "description": clean_text(raw.get("description"), 300),
    }


def _clean_relation(raw, entities: dict, norm_doc: str) -> dict | None:
    if not isinstance(raw, dict):
        return None
    source, target = raw.get("source"), raw.get("target")
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    source_key, target_key = normalize_key(source), normalize_key(target)
    if source_key == target_key or source_key not in entities or target_key not in entities:
        return None

    relation = re.sub(r"[^a-z0-9]+", "_", str(raw.get("relation", "")).lower()).strip("_")[:40]
    quote = clean_text(raw.get("quote"), 600)
    if not relation or not quote or not quote_in_doc(quote, norm_doc):
        return None          # no evidence in the document: drop it

    return {"source_key": source_key, "target_key": target_key, "relation": relation, "quote": quote}


def run(doc_id: str, markdown: str) -> None:
    model = os.getenv("GRAPH_MODEL") or None   # None -> MAIN_MODEL
    text = markdown[:MAX_TOTAL_CHARS]
    norm_doc = norm(text)

    raw_outputs: list[dict] = []
    entities: dict[str, dict] = {}
    relations: dict[tuple, dict] = {}
    dropped = 0

    for window in split_windows(text):
        prompt = PROMPT.replace("<<RULES>>", RULES).replace("<<TEXT>>", window)
        data = ask_json(prompt, model, MAX_TOKENS)
        raw_outputs.append(data)

        raw_entities, raw_relations = data.get("entities"), data.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            raise ValueError('The model output must contain "entities" and "relations" lists')

        for raw in raw_entities:
            entity = _clean_entity(raw)
            if entity is None:
                dropped += 1
            elif entity["key"] not in entities and len(entities) < MAX_ENTITIES:
                entities[entity["key"]] = entity    # same name in two windows: keep the first

        for raw in raw_relations:
            relation = _clean_relation(raw, entities, norm_doc)
            if relation is None:
                dropped += 1
                continue
            key = (relation["source_key"], relation["relation"], relation["target_key"])
            if key not in relations and len(relations) < MAX_RELATIONS:
                relations[key] = relation

    save_branch_output(doc_id, "graph", {
        "windows": raw_outputs, "entities": len(entities),
        "relations": len(relations), "dropped": dropped,
    })
    n_entities, n_relations = save_graph(doc_id, list(entities.values()), list(relations.values()))
    logger.info("doc_id=%s graph: %d entities, %d relations, dropped %d",
                doc_id, n_entities, n_relations, dropped)