from pydantic import BaseModel, Field
from typing import List, Optional, Set, Dict, Literal, Tuple
from collections import defaultdict
import json
try:
    from pytoony import json2toon, toon2json
except ImportError:
    json2toon = None
    toon2json = None

class EvidenceNode(BaseModel):
    id: Optional[int] = None
    source_title: str = ""
    source_url: str = ""
    content: str = ""
    
class EvidenceNodeList(BaseModel):
    evidence_nodes: List[EvidenceNode] = Field([], description="List of evidence nodes")
    last_evidence_id: int = 0
    
    def add_evidence_node(self, source_title: str, source_url: str, content: str) -> EvidenceNode:
        evidence_node = EvidenceNode(
            id=self.last_evidence_id+1,
            source_title=source_title,
            source_url=source_url,
            content=content
        )
        self.evidence_nodes.append(evidence_node)
        self.last_evidence_id += 1
        return evidence_node
      
    def get_evidence_node_by_id(self, id: int) -> Optional[EvidenceNode]:
        for node in self.evidence_nodes:
            if node.id == id:
                return node
        return None

class KnowledgeNode(BaseModel):
    id: int = Field(..., description="Unique identifier for the knowledge node")
    # type: str = Field(..., description="Entity-knowledge node or Attribute-knowledge node")
    knowledge: str = Field(..., description="Content of the knowledge node")
    is_core_entity: bool = Field(False, description="Whether this node is a core entity in the research topic")
    # Optional clustering label (for non-destructive clustering; stored/persisted to Neo4j if available)
    cluster_id: Optional[str] = Field(default=None, description="Optional cluster id for semantic clustering (e.g., HDBSCAN)")
    # Optional community label (for community detection algorithms like Leiden)
    community_id: Optional[str] = Field(default=None, description="Optional community id for community detection (e.g., Leiden)")
    # evidence_nodes: List[EvidenceNode] = Field([])
    # relation: Optional[Dict[int, str]] = Field(None, description="Relation of the knowledge node to other nodes")

    
class KnowledgeEdge(BaseModel):
    id: int
    source_id: int
    target_id: int
    relation_name: str
    # relation_type: str = Field(..., description="Type of the relation between nodes")
    evidence_nodes: List[EvidenceNode] = Field([]) 
    
      
class KnowledgeGraph(BaseModel):
    knowledge_nodes: List[KnowledgeNode] = Field([], description="List of knowledge nodes in the graph")
    knowledge_edges: List[KnowledgeEdge] = Field([], description="List of relations between nodes")
    evidence_nodes: List[EvidenceNode] = Field([], description="List of all evidence nodes in the graph")
    merged_knowledge_nodes: List[Tuple[int, str, int]] = Field([], description="[(original_node_id, original_name, merged_node_id), ...]")
    # Non-destructive semantic clustering results (kept so LLM can see clustering info later)
    semantic_clusters: List[Dict] = Field(default_factory=list, description="Semantic clustering metadata (non-destructive)")
    # Non-destructive community detection results (kept so LLM can see community info later)
    community_list: List[Dict] = Field(default_factory=list, description="Community detection metadata (non-destructive, e.g., Leiden)")
    # Cache for node embeddings to avoid redundant computation
    embedding_cache: Dict[Tuple[str, str], List[float]] = Field(default_factory=dict, description="Cache for node embeddings")
    last_evidence_id: int = 0
    last_knowledge_node_id: int = 0
    last_knowledge_edge_id: int = 0
    
    def get_knowledge_edge_by_id(self, id: int) -> Optional[KnowledgeEdge]:
      for edge in self.knowledge_edges:
          if edge.id == id:
              return edge
      return None
    def get_knowledge_node_by_id(self, id: int) -> Optional[KnowledgeNode]:
      for node in self.knowledge_nodes:
          if node.id == id:
              return node
      return None
    def get_evidence_node_by_id(self, id: int) -> Optional[EvidenceNode]:
      for evidence_node in self.evidence_nodes:
          if evidence_node.id == id:
              return evidence_node
      return None
    
    def add_evidence_node(self, source_title: str, source_url: str, content: str) -> EvidenceNode:
        evidence_node = EvidenceNode(
            id=self.last_evidence_id+1,
            source_title=source_title,
            source_url=source_url,
            content=content
        )
        self.evidence_nodes.append(evidence_node)
        self.last_evidence_id += 1
        return evidence_node
      
    def to_text(self) -> str:
      lines = []
      # === [KNOWLEDGE NODES] ===
      lines.append("[KNOWLEDGE NODES]")
      # Sort nodes by id to ensure KN1, KN2, ...
      sorted_nodes = sorted(self.knowledge_nodes, key=lambda n: n.id)
      
      for node in sorted_nodes:
          if node.is_core_entity==True:
              lines.append(f'KN{node.id}: {node.knowledge} [type: entity]')
          else:
              # Attribute node: no [desc: ...], no [source: ...]
              lines.append(f'KN{node.id}: {node.knowledge} [type: attribute]')
      
      lines.append("")  # blank line

      # === [RELATIONSHIPS] ===
      lines.append("[RELATIONSHIPS]")
      
      # Sort edges for deterministic output (optional but recommended)
      sorted_edges = sorted(self.knowledge_edges, key=lambda e: (e.source_id, e.target_id, e.relation_name))
      for edge in sorted_edges:
          lines.append(f'KN{edge.source_id} → "{edge.relation_name}" → KN{edge.target_id}')
      
      lines.append("")  # blank line

      # === [SEMANTIC CLUSTERS] ===
      if self.semantic_clusters:
          lines.append("[SEMANTIC CLUSTERS]")
          for i, cluster in enumerate(self.semantic_clusters, 1):
              cluster_id = cluster.get("cluster_id", f"cluster_{i}")
              representative = cluster.get("representative_concept", "")
              source_nodes = cluster.get("source_node_ids", [])
              justification = cluster.get("similarity_justification", "")
              
              lines.append(f"Cluster {cluster_id}: {representative}")
              lines.append(f"  Nodes: {', '.join(source_nodes)}")
              if justification:
                  lines.append(f"  Justification: {justification}")
          lines.append("")  # blank line
      
      # === [COMMUNITIES] ===
      if self.community_list:
          lines.append("[COMMUNITIES]")
          for i, community in enumerate(self.community_list, 1):
              community_id = community.get("community_id", f"community_{i}")
              representative = community.get("representative_node", "")
              source_nodes = community.get("source_node_ids", [])
              size = community.get("size", len(source_nodes))
              method = community.get("detection_method", "Unknown")
              
              lines.append(f"Community {community_id}: {representative} [method: {method}, size: {size}]")
              lines.append(f"  Nodes: {', '.join(source_nodes)}")
          lines.append("")  # blank line
      
      return "\n".join(lines)
    
    
    def to_json(self) -> Dict:
      id2node = {node.id: node for node in self.knowledge_nodes}
      return {
        "knowledge_nodes": [
          {
            "node_id": f"n{node.id}",
            "knowledge": node.knowledge,
            "is_core_entity": node.is_core_entity,
            # "cluster_id": node.cluster_id,
            # "community_id": node.community_id,
          }
          for node in self.knowledge_nodes],
        "knowledge_edges": [
          {
            "edge_id": f"e{edge.id}",
            # "relation_name": edge.relation_name,
            "representation": f"{id2node[edge.source_id].knowledge} - {edge.relation_name} -> {id2node[edge.target_id].knowledge}",
          }
        for edge in self.knowledge_edges],
        # "semantic_clusters": self.semantic_clusters,  # 可选：包含语义聚类信息
        # "community_list": self.community_list,  # 可选：包含社区发现信息
      }
    
    def to_toon(self) -> str:
      """
      Convert the knowledge graph to TOON (Token-Oriented Object Notation) format.
      TOON is a compact, human-readable data format, particularly suited for LLM prompts.

      Returns:
          str: TOON-formatted string representation.
      """
      if json2toon is None:
        raise ImportError("pytoony library is required. Please install it with: pip install pytoony")
      
      # 先获取 JSON 字典
      json_dict = self.to_json()
      
      # 将字典转换为 JSON 字符串
      json_str = json.dumps(json_dict, ensure_ascii=False)
      
      # 使用 json2toon 转换为 TOON 格式
      toon_str = json2toon(json_str)
      
      return toon_str
    
    def to_text_for_writer(self, evidence_node_ids: List[int]) -> Dict:
      '''
      Returns a text representation for the report writer:
      1. Only core entity nodes
      2. Relationships in "source - relation -> target" format per line
      3. Only KnowledgeEdges that reference the given evidence_node_ids
      '''
      
      id2node = {node.id: node for node in self.knowledge_nodes}
      lines = []
      # === [KNOWLEDGE NODES] ===
      lines.append("[Core Entity Nodes]")
      for node in self.knowledge_nodes:
          if node.is_core_entity:
              lines.append(f'{node.knowledge}')
      lines.append("")  # blank line
      # === [RELATIONSHIPS] ===
      lines.append("[Relationships]")
      for edge in self.knowledge_edges:
          # Only keep edges that reference the given evidence_node_ids
          edge_evidence_ids = [en.id for en in edge.evidence_nodes]
          if any(eid in evidence_node_ids for eid in edge_evidence_ids):
              lines.append(f'{id2node[edge.source_id].knowledge} - {edge.relation_name} -> {id2node[edge.target_id].knowledge}')
      return "\n".join(lines)
    
    def append_evidence_node_list(self) -> None:
      self.last_evidence_id = 0
      evidence_node_list = []
      for knowledge_node in self.knowledge_nodes:
        for evidence_node in knowledge_node.evidence_nodes:
          evidence_node_list.append(evidence_node)
          if self.last_evidence_id < evidence_node.id:
            self.last_evidence_id = evidence_node.id
      self.evidence_nodes.extend(evidence_node_list)
    
    def get_cluster_map(self) -> Dict[int, int]:
      """
      Get the cluster mapping for knowledge nodes.

      Returns:
          Dict[int, int]: Mapping where key=original node ID, value=merged node ID.
      """
      cluster_map = {}
      for original_id, original_name, merged_id in self.merged_knowledge_nodes:
          cluster_map[original_id] = merged_id
      return cluster_map
    
    # Helper: convert string ID to integer (n1 -> 1, e3 -> 3)
    def _extract_id(self, id_str: str) -> int:
        return int(''.join(filter(str.isdigit, id_str)))
    
    def add_llm_generated_knowledge(self, llm_output: dict, evidence_nodes: List[EvidenceNode]) -> None:
      """
      Integrate LLM-generated knowledge into the existing knowledge graph.

      Args:
          llm_output: LLM-generated JSON with keys: new_nodes, new_edges, evidences_map.
          evidence_nodes: Evidence node list for this batch, ordered as EN1, EN2, ...

      Processing logic:
      1. Convert ID format (n1 -> 1, e1 -> 1)
      2. Node deduplication and merging
      3. Edge deduplication and evidence mapping
      4. Update ID counters
      """
      # 解析LLM输出
      new_nodes_data = llm_output.get("new_nodes", [])
      new_edges_data = llm_output.get("new_edges", [])
      evidences_map = llm_output.get("evidences_map", {})
      
      # 1. Process new nodes - deduplicate and add
      existing_node_names = {node.knowledge.lower(): node.id for node in self.knowledge_nodes}
      node_id_mapping = {f"n{knowledge_node.id}": knowledge_node.id for knowledge_node in self.knowledge_nodes}  # Map n1 -> 1 (LLM outputs n-prefixed IDs)
      
      for node_data in new_nodes_data:
          raw_id = node_data["id"]
          node_name = node_data["node_name"].strip()
          is_core = node_data["is_core_entity"]
          
          # Check for existing node with same name (case-insensitive)
          existing_id = existing_node_names.get(node_name.lower())
          
          if existing_id is not None:
              # Reuse existing node
              node_id_mapping[raw_id] = existing_id
              # If existing node is not core entity but new one is, update the flag
              if is_core:
                  for node in self.knowledge_nodes:
                      if node.id == existing_id:
                          node.is_core_entity = True
                          break
          else:
              # Create new node
              new_id = self.last_knowledge_node_id + 1
              self.last_knowledge_node_id = new_id
              node_id_mapping[raw_id] = new_id
              
              # 添加到知识节点列表
              self.knowledge_nodes.append(KnowledgeNode(
                  id=new_id,
                  knowledge=node_name,
                  is_core_entity=is_core
              ))
              existing_node_names[node_name.lower()] = new_id

      # 2. Process new edges - deduplicate and add
      existing_edges = {(edge.source_id, edge.target_id, edge.relation_name.lower()): edge.id 
                      for edge in self.knowledge_edges}
      edge_id_mapping = {f"e{edge.id}": edge.id for edge in self.knowledge_edges}  # Map e1 -> 1 (LLM outputs e-prefixed IDs)
      
      for edge_data in new_edges_data:
          raw_edge_id = edge_data["id"]
          raw_source_id = edge_data["source_id"]
          raw_target_id = edge_data["target_id"]
          relation = edge_data["relation_name"].strip()
          
          # Convert source/target IDs
          source_id = node_id_mapping.get(raw_source_id)
          target_id = node_id_mapping.get(raw_target_id)
          
          # Validate ID mapping
          if source_id is None or target_id is None:
              continue

          # Check if edge already exists (semantic triple match)
          edge_key = (source_id, target_id, relation.lower())
          existing_edge_id = existing_edges.get(edge_key)
          
          if existing_edge_id is not None:
              # Reuse existing edge
              edge_id_mapping[raw_edge_id] = existing_edge_id
          else:
              # Create new edge
              new_edge_id = self.last_knowledge_edge_id + 1
              self.last_knowledge_edge_id = new_edge_id
              edge_id_mapping[raw_edge_id] = new_edge_id
              
              # Add to edge list (initially no evidence)
              self.knowledge_edges.append(KnowledgeEdge(
                  id=new_edge_id,
                  source_id=source_id,
                  target_id=target_id,
                  relation_name=relation,
                  evidence_nodes=[]
              ))
              existing_edges[edge_key] = new_edge_id
      # 3. Process evidence mapping
      # Build EN number -> actual EvidenceNode object mapping
      en_to_evidence = {}


      # Extract EN numbers (e.g. EN14, EN15) from evidences_map
      for en_id in evidences_map.keys():
          if en_id.startswith("EN"):
              try:
                  num = int(en_id[2:])
                  matched_node = [e for e in evidence_nodes if e.id == num]
                  if matched_node and len(matched_node) > 0:
                    en_to_evidence[en_id] = matched_node[0]
              except ValueError:
                  continue

      
      # 4. Add evidence support to each edge
      for en_id, edge_ids in evidences_map.items():
          evidence_node = en_to_evidence.get(en_id)
          if not evidence_node or evidence_node.id is None:
              continue
          
          for raw_edge_id in edge_ids:
              edge_id = edge_id_mapping.get(raw_edge_id)
              if not edge_id:
                  continue
              
              # Find the edge and add evidence
              for edge in self.knowledge_edges:
                  if edge.id == edge_id:
                      # Avoid duplicate evidence
                      if not any(e.id == evidence_node.id for e in edge.evidence_nodes):
                          edge.evidence_nodes.append(evidence_node)
                      break
      
      # 5. Clean up orphaned nodes (concept nodes with no connections)
      self._remove_orphaned_concept_nodes()

    def _remove_orphaned_concept_nodes(self) -> None:
        """
        Remove orphaned nodes with no connections.
        Rule: keep a node if it appears in any edge's source_id or target_id.
        """
        # 1. 收集所有在边中出现过的节点 ID
        active_node_ids = set()
        for edge in self.knowledge_edges:
            active_node_ids.add(edge.source_id)
            active_node_ids.add(edge.target_id)
        
        # 2. 只保留：是核心实体 OR 存在于边中的节点
        self.knowledge_nodes = [
            node for node in self.knowledge_nodes
            if node.is_core_entity or node.id in active_node_ids
        ]
      
    
    def apply_merge_node_results(self, clustering_results: dict) -> None:
      """
      Apply clustering results to update the knowledge graph - CORRECTED VERSION
      
      Critical fixes from previous version:
      1. Properly removes source nodes after clustering
      2. Correctly merges edges with same source-target (ignoring relation_name)
      3. Eliminates self-loop edges (X->X)
      4. Fully migrates all evidence to new edges
      
      Args:
          clustering_results: JSON output from LLM clustering with format:
          {
            "clusters": [
              {
                "cluster_id": "c1",
                "representative_concept": "New concept name",
                "source_node_ids": ["2", "3", ...],  # String IDs of nodes to merge
                "similarity_justification": "Justification text"
              },
              ...
            ]
          }
      """
      if not clustering_results or not clustering_results.get("clusters"):
          return
      
      # Step 1: Create new merged nodes and build node replacement map
      node_replacement_map = {}  # old_node_id -> new_node_id
      nodes_to_remove = set()    # Collect all node IDs to remove
      
      for cluster in clustering_results["clusters"]:
          # Parse source node IDs (handle both "n2" and "2" formats)
          source_node_ids = []
          for node_id_str in cluster["source_node_ids"]:
              if node_id_str.startswith('n'):
                  source_node_ids.append(int(node_id_str[1:]))
              else:
                  source_node_ids.append(int(node_id_str))
          
          if len(source_node_ids) < 2:
              continue
          
          # Validate source nodes exist and are concept nodes (not core entities)
          valid_source_nodes = []
          for node_id in source_node_ids:
              node = self.get_knowledge_node_by_id(node_id)
              if node and not node.is_core_entity:
                  valid_source_nodes.append(node)
                  nodes_to_remove.add(node_id)  # Mark for removal
          
          if len(valid_source_nodes) < 2:
              continue
          
          # Create new merged node
          new_node_id = self.last_knowledge_node_id + 1
          self.last_knowledge_node_id = new_node_id
          
          new_node = KnowledgeNode(
              id=new_node_id,
              knowledge=cluster["representative_concept"],
              is_core_entity=False
          )
          self.knowledge_nodes.append(new_node)
          # Record merging history BEFORE building replacement map
          for node in valid_source_nodes:
              self.merged_knowledge_nodes.append((node.id, node.knowledge, new_node_id))
          # Map all source node IDs to the new merged node ID
          for node_id in source_node_ids:
              node_replacement_map[node_id] = new_node_id
      
      if not node_replacement_map:
          return
      
      # Step 2: Process edges - merge and redirect
      new_edge_map = {}          # (new_source_id, new_target_id) -> new_edge, for merging edges
      edges_to_remove = set()    # Collect edge IDs to remove
      
      for edge in self.knowledge_edges:
          # For each existing edge in the KG
          # Determine new source and target IDs
          new_source_id = node_replacement_map.get(edge.source_id, edge.source_id)
          new_target_id = node_replacement_map.get(edge.target_id, edge.target_id)
          
          # Rule 1: Skip self-loop edges (X->X)
          if new_source_id == new_target_id:
              edges_to_remove.add(edge.id)
              continue
          
          # Rule 2: Only process edges connected to nodes being replaced
          if edge.source_id in node_replacement_map or edge.target_id in node_replacement_map:
              edges_to_remove.add(edge.id)
              
              edge_key = (new_source_id, new_target_id)
              
              if edge_key in new_edge_map:
                  # Existing edge: only migrate evidence
                  existing_edge = new_edge_map[edge_key]
                  for evidence in edge.evidence_nodes:
                      if not any(e.id == evidence.id for e in existing_edge.evidence_nodes):
                          existing_edge.evidence_nodes.append(evidence)
              else:
                  # Create new edge with first relation name encountered
                  new_edge_id = self.last_knowledge_edge_id + 1
                  self.last_knowledge_edge_id = new_edge_id
                  
                  new_edge = KnowledgeEdge(
                      id=new_edge_id,
                      source_id=new_source_id,
                      target_id=new_target_id,
                      relation_name=edge.relation_name,  # Keep the first relation name
                      evidence_nodes=edge.evidence_nodes.copy()  # Copy all evidence
                  )
                  new_edge_map[edge_key] = new_edge
      
      # Step 3: Update edges list
      updated_edges = []
      for edge in self.knowledge_edges:
          if edge.id not in edges_to_remove:
              updated_edges.append(edge)
      
      # Add all new merged edges
      updated_edges.extend(new_edge_map.values())
      self.knowledge_edges = updated_edges
      
      # Step 4: REMOVE SOURCE NODES
      self.knowledge_nodes = [
          node for node in self.knowledge_nodes
          if node.id not in nodes_to_remove
      ]
      
      # Step 5: Clean up orphaned edges
      valid_node_ids = {node.id for node in self.knowledge_nodes}
      self.knowledge_edges = [
          edge for edge in self.knowledge_edges
          if edge.source_id in valid_node_ids and edge.target_id in valid_node_ids
      ]

    def apply_semantic_clustering_results(self, clustering_results: dict) -> None:
      """
      Entry point for semantic clustering (embedding/HDBSCAN): uses non-destructive strategy.
      - Tags nodes in the same cluster with node.cluster_id
      - Writes cluster summaries to self.semantic_clusters for downstream LLM consumption

      For destructive merge, explicitly call apply_merge_node_results instead.
      """
      if not clustering_results or not clustering_results.get("clusters"):
          return

      # Normalize & store clusters
      normalized_clusters: List[Dict] = []
      node_to_cluster: Dict[int, str] = {}

      for cluster in clustering_results.get("clusters", []):
          cluster_id = str(cluster.get("cluster_id", "")).strip() or "semantic_cluster"
          rep = str(cluster.get("representative_concept", "")).strip()
          justification = str(cluster.get("similarity_justification", "")).strip()

          src_ids_raw = cluster.get("source_node_ids", []) or []
          src_node_ids: List[str] = []
          src_node_ints: List[int] = []
          for s in src_ids_raw:
              s = str(s).strip()
              if not s:
                  continue
              src_node_ids.append(s if s.startswith("n") else f"n{s}")
              try:
                  src_node_ints.append(int(''.join(filter(str.isdigit, s))))
              except ValueError:
                  continue

          if len(src_node_ints) < 2:
              continue

          normalized_clusters.append(
              {
                  "cluster_id": cluster_id,
                  "representative_concept": rep,
                  "source_node_ids": src_node_ids,
                  "similarity_justification": justification,
              }
          )
          for nid in src_node_ints:
              node_to_cluster[nid] = cluster_id

      # Apply node.cluster_id (do not touch graph topology)
      for node in self.knowledge_nodes:
          if node.is_core_entity:
              continue
          if node.id in node_to_cluster:
              node.cluster_id = node_to_cluster[node.id]

      # Persist clusters for later LLM consumption (replace existing semantic_clusters)
      self.semantic_clusters = normalized_clusters

    def apply_community_detection_results(self, community_results: Dict[int, str]) -> None:
      """
      Entry point for community detection (Leiden, etc.): uses non-destructive strategy.
      - Tags nodes assigned to the same community with node.community_id
      - Writes community summaries to self.community_list for downstream LLM consumption

      Args:
        community_results: Dict[int, str] - node_id -> community_id mapping.
      """
      if not community_results:
          return

      # Build reverse mapping: community_id -> [node_ids]
      community_to_nodes: Dict[str, List[int]] = defaultdict(list)
      for node_id, community_id in community_results.items():
          community_to_nodes[community_id].append(node_id)

      normalized_communities: List[Dict] = []

      for community_id, node_ids in sorted(community_to_nodes.items(), key=lambda kv: kv[0]):
          if len(node_ids) < 1:
              continue

          # Get node knowledge content for representative description
          node_knowledges = []
          for node in self.knowledge_nodes:
              if node.id in node_ids:
                  node_knowledges.append(node.knowledge)

          # Use first node as representative
          representative = node_knowledges[0] if node_knowledges else ""
          
          normalized_communities.append(
              {
                  "community_id": community_id,
                  "representative_node": representative,
                  "source_node_ids": [f"n{nid}" for nid in sorted(node_ids)],
                  "size": len(node_ids),
                  "detection_method": "Leiden",
              }
          )
          
          # Apply node.community_id
          for node in self.knowledge_nodes:
              if node.id in node_ids:
                  node.community_id = community_id

      # Persist communities for later LLM consumption (replace existing community_list)
      self.community_list = normalized_communities

    def _remove_orphaned_edges(self) -> None:
      """Remove edges that reference non-existent nodes"""
      valid_node_ids = {node.id for node in self.knowledge_nodes}
      self.knowledge_edges = [
          edge for edge in self.knowledge_edges
          if edge.source_id in valid_node_ids and edge.target_id in valid_node_ids
      ]
    
class SectionNode(BaseModel):
    id: Optional[int] = None
    section_title: str
    # section_description: str
    subsection_nodes: List['SectionNode'] = Field(default_factory=list)
    search_goals: List[str] = Field(default_factory=list, description="list of current required knowledges to accomplish this section which haven't been searched")
    evidences: List[EvidenceNode] = Field(default_factory=list, description="list of evidence nodes supporting this section")
    
class SkeletonGraph(BaseModel):
    root_query: str = Field(
        description="root query of deepresearch topic",
        default=""
    )
    title: str = Field(
        description="title of the research report",
        default=""
    )
    section_nodes: List[SectionNode] = Field(
        default_factory=list
    )
    
    
ChainType = Literal["enrich", "explore_entity_attribute", "explore_attribute_attribute", "explore_cross_community", "explore_cross_cluster", "explore_sbm_probability", "explore_sbm_entropy", "explore_entity_concept", "explore_entity_concept_coverage_gap", "explore_entity_concept_similarity"]

class Chain(BaseModel):
    id: int
    type: ChainType
    nodes: List[int]
    content: Optional[str] = None
    reason: str
    is_visited: bool = False
    
    
from typing import List
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None
