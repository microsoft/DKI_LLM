from __future__ import annotations

import os
from typing import Dict, List, Optional
from urllib.parse import quote

from neomodel import config as neomodel_config
from neomodel import db

from data_model import EvidenceNode, KnowledgeEdge, KnowledgeGraph, KnowledgeNode


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    cleaned = v.strip().strip("'\"")
    return cleaned if cleaned else default


def _build_database_url_from_env() -> Optional[str]:
    """
    Support both:
    - local style: NEO4J_BOLT_URL=bolt://neo4j:password@localhost:7687
    - Aura style:
        NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
        NEO4J_USERNAME=neo4j
        NEO4J_PASSWORD=...
        (optional) NEO4J_DATABASE=neo4j
    """
    # 1) Direct URL (highest priority)
    direct = (
        _env_str("NEO4J_BOLT_URL")
        or _env_str("NEOMODEL_DATABASE_URL")
        or _env_str("NEO4J_URL")
    )
    if direct:
        return direct

    # 2) Aura / split credentials
    uri = _env_str("NEO4J_URI")
    user = _env_str("NEO4J_USERNAME")
    pwd = _env_str("NEO4J_PASSWORD")
    if not uri:
        return None

    # If uri already includes credentials, keep it as-is.
    if "@" in uri:
        return uri

    if not user or not pwd:
        return uri

    scheme_sep = "://"
    if scheme_sep not in uri:
        uri = f"neo4j+s://{uri}"

    scheme, rest = uri.split(scheme_sep, 1)
    user_enc = quote(user, safe="")
    pwd_enc = quote(pwd, safe="")
    return f"{scheme}{scheme_sep}{user_enc}:{pwd_enc}@{rest}"


class Neo4jKGStore:
    """
    Minimal Neo4j persistence layer for baselines/data_model.py KnowledgeGraph.
    """

    def __init__(self, database_url: str, report_id: int):
        if not database_url:
            raise ValueError(
                "database_url is required. Set env var NEO4J_BOLT_URL/NEOMODEL_DATABASE_URL "
                "or Aura vars NEO4J_URI+NEO4J_USERNAME+NEO4J_PASSWORD."
            )
        self.database_url = database_url
        self.report_id = int(report_id)
        neomodel_config.DATABASE_URL = database_url

        # Optional: Aura exposes a database name (usually "neo4j").
        db_name = _env_str("NEO4J_DATABASE")
        if db_name and hasattr(neomodel_config, "DATABASE_NAME"):
            try:
                setattr(neomodel_config, "DATABASE_NAME", db_name)
            except Exception:
                pass

    @classmethod
    def from_env(cls, report_id: int) -> "Neo4jKGStore":
        url = _build_database_url_from_env()
        if not url:
            raise ValueError(
                "Neo4j database URL not found in environment. Please set one of:\n"
                "- NEO4J_BOLT_URL / NEOMODEL_DATABASE_URL / NEO4J_URL (e.g. bolt://neo4j:<password>@localhost:7687)\n"
                "- Aura: NEO4J_URI=neo4j+s://<id>.databases.neo4j.io + NEO4J_USERNAME + NEO4J_PASSWORD"
            )
        return cls(database_url=url, report_id=report_id)

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT kg_node_unique IF NOT EXISTS FOR (n:KGNode) REQUIRE (n.report_id, n.node_id) IS UNIQUE",
            "CREATE CONSTRAINT kg_ev_unique IF NOT EXISTS FOR (e:KGEvidence) REQUIRE (e.report_id, e.evidence_id) IS UNIQUE",
            "CREATE INDEX kg_rel_report_id IF NOT EXISTS FOR ()-[r:KG_REL]-() ON (r.report_id)",
        ]
        for cypher in statements:
            try:
                db.cypher_query(cypher)
            except Exception:
                continue

    def clear_report(self) -> None:
        db.cypher_query(
            """
            MATCH (n)
            WHERE n.report_id = $report_id AND (n:KGNode OR n:KGEvidence)
            DETACH DELETE n
            """,
            {"report_id": self.report_id},
        )

    def replace_graph(self, kg: KnowledgeGraph, iter_idx: Optional[int] = None) -> None:
        self.ensure_schema()
        self.clear_report()

        # 1) Nodes
        node_rows: List[Dict] = []
        for n in kg.knowledge_nodes:
            node_rows.append(
                {
                    "report_id": self.report_id,
                    "node_id": int(n.id),
                    "knowledge": n.knowledge,
                    "name": n.knowledge,  # for Neo4j Browser caption
                    "is_core_entity": bool(getattr(n, "is_core_entity", False)),
                    "cluster_id": getattr(n, "cluster_id", None),
                }
            )

        if node_rows:
            db.cypher_query(
                """
                UNWIND $rows AS row
                MERGE (n:KGNode {report_id: row.report_id, node_id: row.node_id})
                SET n.knowledge = row.knowledge,
                    n.name = row.name,
                    n.is_core_entity = row.is_core_entity,
                    n.cluster_id = row.cluster_id
                """,
                {"rows": node_rows},
            )

        # 2) Evidences
        ev_rows: List[Dict] = []
        for ev in getattr(kg, "evidence_nodes", []) or []:
            if ev.id is None:
                continue
            ev_rows.append(
                {
                    "report_id": self.report_id,
                    "evidence_id": int(ev.id),
                    "source_title": ev.source_title or "",
                    "source_url": ev.source_url or "",
                    "content": ev.content or "",
                }
            )

        if ev_rows:
            db.cypher_query(
                """
                UNWIND $rows AS row
                MERGE (e:KGEvidence {report_id: row.report_id, evidence_id: row.evidence_id})
                SET e.source_title = row.source_title,
                    e.source_url = row.source_url,
                    e.content = row.content
                """,
                {"rows": ev_rows},
            )

        # 3) Relationships
        rel_rows: List[Dict] = []
        for edge in kg.knowledge_edges:
            evidence_ids: List[int] = []
            for ev in edge.evidence_nodes or []:
                if ev and ev.id is not None:
                    evidence_ids.append(int(ev.id))
            rel_rows.append(
                {
                    "report_id": self.report_id,
                    "edge_id": int(edge.id),
                    "source_id": int(edge.source_id),
                    "target_id": int(edge.target_id),
                    "relation_name": edge.relation_name or "",
                    "evidence_ids": evidence_ids,
                    "iter": int(iter_idx) if iter_idx is not None else None,
                }
            )

        if rel_rows:
            db.cypher_query(
                """
                UNWIND $rows AS row
                MATCH (s:KGNode {report_id: row.report_id, node_id: row.source_id})
                MATCH (t:KGNode {report_id: row.report_id, node_id: row.target_id})
                MERGE (s)-[r:KG_REL {report_id: row.report_id, edge_id: row.edge_id}]->(t)
                SET r.relation_name = row.relation_name,
                    r.name = row.relation_name,
                    r.evidence_ids = row.evidence_ids,
                    r.iter = row.iter
                """,
                {"rows": rel_rows},
            )

    def load_graph(self) -> KnowledgeGraph:
        rows, _ = db.cypher_query(
            """
            MATCH (n:KGNode {report_id: $report_id})
            RETURN n.node_id AS node_id, n.knowledge AS knowledge, n.is_core_entity AS is_core_entity, n.cluster_id AS cluster_id
            ORDER BY n.node_id ASC
            """,
            {"report_id": self.report_id},
        )
        nodes: List[KnowledgeNode] = [
            KnowledgeNode(
                id=int(r[0]), knowledge=r[1], is_core_entity=bool(r[2]), cluster_id=r[3]
            )
            for r in rows
        ]

        ev_rows, _ = db.cypher_query(
            """
            MATCH (e:KGEvidence {report_id: $report_id})
            RETURN e.evidence_id AS evidence_id, e.source_title AS source_title, e.source_url AS source_url, e.content AS content
            ORDER BY e.evidence_id ASC
            """,
            {"report_id": self.report_id},
        )
        evidences: List[EvidenceNode] = []
        ev_by_id: Dict[int, EvidenceNode] = {}
        for r in ev_rows:
            ev = EvidenceNode(
                id=int(r[0]),
                source_title=r[1] or "",
                source_url=r[2] or "",
                content=r[3] or "",
            )
            evidences.append(ev)
            ev_by_id[ev.id] = ev

        edge_rows, _ = db.cypher_query(
            """
            MATCH (s:KGNode {report_id: $report_id})-[r:KG_REL {report_id: $report_id}]->(t:KGNode {report_id: $report_id})
            RETURN r.edge_id AS edge_id, s.node_id AS source_id, t.node_id AS target_id, r.relation_name AS relation_name, r.evidence_ids AS evidence_ids
            ORDER BY r.edge_id ASC
            """,
            {"report_id": self.report_id},
        )
        edges: List[KnowledgeEdge] = []
        for r in edge_rows:
            evidence_ids = r[4] or []
            edge_evs = [
                ev_by_id[eid]
                for eid in evidence_ids
                if isinstance(eid, int) and eid in ev_by_id
            ]
            edges.append(
                KnowledgeEdge(
                    id=int(r[0]),
                    source_id=int(r[1]),
                    target_id=int(r[2]),
                    relation_name=r[3] or "",
                    evidence_nodes=edge_evs,
                )
            )

        kg = KnowledgeGraph(
            knowledge_nodes=nodes, knowledge_edges=edges, evidence_nodes=evidences
        )
        if nodes:
            kg.last_knowledge_node_id = max(n.id for n in nodes)
        if edges:
            kg.last_knowledge_edge_id = max(e.id for e in edges)
        if evidences:
            kg.last_evidence_id = max(e.id for e in evidences if e.id is not None)
        return kg

    def write_back_node_clusters(
        self,
        node_id_to_cluster_id: Dict[int, str],
        *,
        cluster_property: str = "cluster_id",
        algorithm: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Write back clustering result to Neo4j node properties (non-destructive).
        """
        if not node_id_to_cluster_id:
            return

        rows = [
            {
                "report_id": self.report_id,
                "node_id": int(node_id),
                "cluster_id": str(cluster_id),
                "algorithm": algorithm,
                "run_id": run_id,
            }
            for node_id, cluster_id in node_id_to_cluster_id.items()
        ]

        allowed = {"cluster_id", "community_id", "partition_id"}
        prop = cluster_property if cluster_property in allowed else "cluster_id"

        db.cypher_query(
            f"""
            UNWIND $rows AS row
            MATCH (n:KGNode {{report_id: row.report_id, node_id: row.node_id}})
            SET n.{prop} = row.cluster_id,
                n.cluster_algorithm = row.algorithm,
                n.cluster_run_id = row.run_id
            """,
            {"rows": rows},
        )

    def fetch_node_clusters(
        self, *, cluster_property: str = "cluster_id"
    ) -> Dict[int, str]:
        allowed = {"cluster_id", "community_id", "partition_id"}
        prop = cluster_property if cluster_property in allowed else "cluster_id"

        rows, _ = db.cypher_query(
            f"""
            MATCH (n:KGNode {{report_id: $report_id}})
            WHERE n.{prop} IS NOT NULL
            RETURN n.node_id AS node_id, n.{prop} AS cluster_id
            ORDER BY n.node_id ASC
            """,
            {"report_id": self.report_id},
        )
        out: Dict[int, str] = {}
        for r in rows:
            try:
                out[int(r[0])] = str(r[1])
            except Exception:
                continue
        return out

    def run_gds_leiden_writeback(
        self,
        *,
        graph_name: Optional[str] = None,
        write_property: str = "community_id",
    ) -> Dict[int, str]:
        """
        Run GDS Leiden community detection and write back to KGNode.<write_property>.
        Requires Neo4j GDS plugin/service.
        """
        allowed = {"community_id", "partition_id", "cluster_id"}
        prop = write_property if write_property in allowed else "community_id"
        gname = graph_name or f"kg_report_{self.report_id}"

        try:
            db.cypher_query(
                "CALL gds.graph.drop($gname, false) YIELD graphName", {"gname": gname}
            )
        except Exception:
            pass

        db.cypher_query(
            """
            CALL gds.graph.project.cypher(
              $gname,
              'MATCH (n:KGNode {report_id: $rid}) RETURN id(n) AS id',
              'MATCH (s:KGNode {report_id: $rid})-[:KG_REL {report_id: $rid}]->(t:KGNode {report_id: $rid}) RETURN id(s) AS source, id(t) AS target'
            )
            YIELD graphName, nodeCount, relationshipCount
            """,
            {"gname": gname, "rid": self.report_id},
        )

        db.cypher_query(
            f"CALL gds.leiden.write($gname, {{ writeProperty: '{prop}' }}) YIELD communityCount",
            {"gname": gname},
        )

        return self.fetch_node_clusters(cluster_property=prop)
