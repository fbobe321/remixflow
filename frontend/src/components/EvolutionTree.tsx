import type { TreeNode } from "../types";

interface Props {
  tree: TreeNode | null;
  selectedId: string | null;
  parentId: string | null;
  onSelect: (variantId: string) => void;
  onBranchFrom: (variantId: string) => void;
}

/** Branching evolution history (PRD §5). Click to audition; ↳ to branch. */
export function EvolutionTree({ tree, selectedId, parentId, onSelect, onBranchFrom }: Props) {
  if (!tree) return <p className="subtle">No evolution yet.</p>;
  return (
    <div className="evolution-tree">
      <Node
        node={tree}
        depth={0}
        selectedId={selectedId}
        parentId={parentId}
        onSelect={onSelect}
        onBranchFrom={onBranchFrom}
      />
    </div>
  );
}

function Node({
  node,
  depth,
  selectedId,
  parentId,
  onSelect,
  onBranchFrom,
}: {
  node: TreeNode;
  depth: number;
} & Omit<Props, "tree">) {
  const v = node.variant;
  const sim = v.similarity != null ? Math.round(v.similarity * 100) : null;
  const rating = v.rating === 1 ? "👍" : v.rating === -1 ? "👎" : "";
  return (
    <div className="tree-node" style={{ marginLeft: depth ? 18 : 0 }}>
      <div
        className={`node-card ${selectedId === v.id ? "selected" : ""} ${
          parentId === v.id ? "is-parent" : ""
        } ${v.is_original ? "original" : ""}`}
      >
        <button className="node-main" onClick={() => onSelect(v.id)}>
          <span className="node-label">
            {v.is_original ? "★ " : ""}
            {v.label} {rating}
          </span>
          {sim != null && <span className="node-sim">{sim}% match</span>}
        </button>
        <button
          className="node-branch"
          title="Branch a new variation from here"
          onClick={() => onBranchFrom(v.id)}
        >
          ↳
        </button>
      </div>
      {node.children.map((child) => (
        <Node
          key={child.variant.id}
          node={child}
          depth={depth + 1}
          selectedId={selectedId}
          parentId={parentId}
          onSelect={onSelect}
          onBranchFrom={onBranchFrom}
        />
      ))}
    </div>
  );
}
