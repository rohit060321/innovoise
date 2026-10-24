"""Document ingestion, chunking, embedding, and initial graph construction."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans

from .config import MorpheConfig
from .embedding.embedder import BaseEmbedder
from .graph.memory_graph import MemoryGraph
from .graph.index import LocalVectorIndex
from .storage.db import SQLiteStorage, MemoryRecord


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Split text into overlapping token/word chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def ingest_documents(
    docs: List[Dict[str, Any]],
    graph: MemoryGraph,
    storage: SQLiteStorage,
    embedder: BaseEmbedder,
    config: MorpheConfig,
) -> List[MemoryRecord]:
    """
    Ingest raw documents, generate embeddings, cluster into initial modules,
    construct graph topology, and persist to SQLite.
    """
    all_chunks: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []

    for doc in docs:
        text = doc.get("text", "")
        doc_id = doc.get("id", "doc")
        category = doc.get("category", "general")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metas.append({
                "doc_id": doc_id,
                "chunk_idx": i,
                "category": category,
                "title": doc.get("title", ""),
            })

    if not all_chunks:
        return []

    # Compute embeddings
    embeddings = embedder.encode(all_chunks)

    # Determine initial cluster count
    num_samples = len(all_chunks)
    n_clusters = min(config.graph.initial_clusters, max(1, num_samples // config.graph.min_module_size))
    n_clusters = max(1, n_clusters)

    # Cluster using KMeans with fixed seed
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.seed, n_init=5)
    cluster_labels = kmeans.fit_predict(embeddings)

    records: List[MemoryRecord] = []
    module_records: Dict[str, List[Tuple[MemoryRecord, np.ndarray]]] = {
        f"M{c}": [] for c in range(n_clusters)
    }

    for chunk, meta, label in zip(all_chunks, chunk_metas, cluster_labels):
        rec_id = f"rec_{meta['doc_id']}_{meta['chunk_idx']}"
        mod_id = f"M{label}"
        rec = MemoryRecord(
            id=rec_id,
            text=chunk,
            metadata=meta,
            module_id=mod_id,
        )
        records.append(rec)
        module_records[mod_id].append((rec, embedder.encode(chunk)))

    # Construct modules in MemoryGraph
    for mod_id, items in module_records.items():
        local_idx = LocalVectorIndex(dimension=embedder.dimension)
        for rec, vec in items:
            local_idx.add(rec.id, vec)

        centroid = local_idx.compute_centroid()
        graph.add_module(
            module_id=mod_id,
            centroid=centroid,
            capacity=config.graph.max_module_capacity,
            local_index=local_idx,
            metadata={"initial_cluster": mod_id},
        )

    # Build initial structural edges between module centroids based on proximity
    mod_ids = list(module_records.keys())
    for i in range(len(mod_ids)):
        for j in range(i + 1, len(mod_ids)):
            u, v = mod_ids[i], mod_ids[j]
            c_u = graph.get_module(u).centroid
            c_v = graph.get_module(v).centroid
            sim = float(np.dot(c_u, c_v))
            if sim > 0.3 or len(mod_ids) <= 3:
                graph.add_edge(u, v, edge_type="structural", weight=round(sim, 3))

    # Ensure graph is connected: if any node has degree 0, connect to closest module
    for mod_id in mod_ids:
        if graph.graph.degree(mod_id) == 0 and len(mod_ids) > 1:
            best_other = None
            best_sim = -1.0
            c_u = graph.get_module(mod_id).centroid
            for other_id in mod_ids:
                if other_id != mod_id:
                    sim = float(np.dot(c_u, graph.get_module(other_id).centroid))
                    if sim > best_sim:
                        best_sim = sim
                        best_other = other_id
            if best_other:
                graph.add_edge(mod_id, best_other, edge_type="structural", weight=round(best_sim, 3))

    # Persist records to SQLite
    storage.insert_records(records)
    return records


def ingest_incremental_documents(
    docs: List[Dict[str, Any]],
    graph: MemoryGraph,
    storage: SQLiteStorage,
    embedder: BaseEmbedder,
    config: MorpheConfig,
) -> List[MemoryRecord]:
    """
    Incrementally ingest documents into an existing MemoryGraph.
    Assigns each chunk to the best-matching module or creates a new one,
    updates centroids, and persists to SQLite.
    """
    if not graph.modules:
        return ingest_documents(docs, graph, storage, embedder, config)

    new_records: List[MemoryRecord] = []
    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue
        doc_id = doc.get("id", f"doc_{int(np.random.randint(1000, 9999))}")
        category = doc.get("category", "custom")
        title = doc.get("title", doc_id)

        chunks = chunk_text(text)
        chunk_embeddings = embedder.encode(chunks)
        if len(chunks) == 1 and chunk_embeddings.ndim == 1:
            chunk_embeddings = np.expand_dims(chunk_embeddings, axis=0)

        for i, (chunk, vec) in enumerate(zip(chunks, chunk_embeddings)):
            rec_id = f"rec_{doc_id}_{i}"

            # Route to highest similarity module centroid
            best_mod_id = None
            best_sim = -1e9
            for mod in graph.get_all_modules():
                norm_c = np.linalg.norm(mod.centroid)
                sim = float(np.dot(vec, mod.centroid / norm_c)) if norm_c > 1e-9 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_mod_id = mod.module_id

            if best_mod_id is None:
                best_mod_id = list(graph.modules.keys())[0]

            rec = MemoryRecord(
                id=rec_id,
                text=chunk,
                metadata={"doc_id": doc_id, "chunk_idx": i, "category": category, "title": title},
                module_id=best_mod_id,
            )
            new_records.append(rec)

            target_mod = graph.get_module(best_mod_id)
            if target_mod:
                target_mod.local_index.add(rec.id, vec)
                target_mod.update_centroid()

    if new_records:
        storage.insert_records(new_records)

    return new_records

