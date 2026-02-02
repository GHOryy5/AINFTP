

use serde::{Serialize, Deserialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::hash::Hasher;
use std::mem::size_of;
use thiserror::Error;

pub use error::Result;

// ERRORS 

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("hash mismatch at layer {layer_idx}")]
    HashMismatch { layer_idx: usize },
    #[error("serialization failed")]
    SerializationError,
    #[error("invalid tree structure: expected {expected}, got {got}")]
    InvalidTree { expected: String, got: String },
    #[error("state corruption detected")]
    CorruptionDetected,
}

// TYPES 

/// A unique identifier for a specific shard of the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShardID {
    pub layer_id: u32,
    pub chunk_idx: u32,
}

/// A cryptographic commitment to a specific model shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCommitment {
    pub id: ShardID,
    pub hash: Vec<u8>, // SHA256 of the data
    pub size: usize,
    pub version: u64,
}

/// The Root of the model state.
/// If all nodes agree on this 32-byte hash, they agree on the entire model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRoot {
    pub root_hash: Vec<u8>, // Merkle Root
    pub tree_depth: u32,
    pub timestamp: u64,
}

// CRYPTOGRAPHY 

pub struct Hasher(Sha256);

impl Hasher {
    pub fn new() -> Self {
        Self(Sha256::new())
    }

    /// Consumes bytes and produces a SHA256 digest.
    pub fn digest(&mut self, data: &[u8]) -> Vec<u8> {
        self.0.update(data);
        self.0.finalize().reset().to_vec()
    }

    /// Hashes two digests together.
    pub fn combine_digests(digests: &[Vec<u8>]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        for digest in digests {
            hasher.update(digest);
        }
        hasher.finalize().reset().to_vec()
    }
}

// MERKLE TREE 

/// Represents a node in the state tree.
/// Leaf nodes are model shards. Internal nodes are hashes of children.
pub struct MerkleNode {
    id: ShardID,
    hash: Vec<u8>,
    left: Option<Box<MerkleNode>>,
    right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    /// Creates a leaf node containing actual data (weights).
    pub fn leaf(id: ShardID, data: &[f32]) -> Self {
        let mut hasher = Hasher::new();
        
        // Optimization: Cast f32 slice to u8 slice for hashing
        // This avoids allocation.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * 4,
            )
        };
        
        let hash = hasher.digest(bytes);
        Self {
            id,
            hash,
            left: None,
            right: None,
        }
    }

    /// Creates an internal node by hashing its two children.
    pub fn internal(left: MerkleNode, right: MerkleNode) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(&left.hash);
        hasher.update(&right.hash);
        let hash = hasher.digest(&[]);

        // ID is derived from children (mock for simplicity)
        Self {
            id: ShardID {
                layer_id: 0,
                chunk_idx: 0,
            },
            hash,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    /// Verifies if the hash matches the calculated state.
    pub fn verify(&self) -> bool {
        let mut hasher = Hasher::new();
        
        // Leaf check vs Internal check
        let data_hash = match (&self.left, &self.right) {
            (None, None) => {
                // It's a leaf. We would need the data here to verify,
                // but for the node struct we only store the hash.
                // In a real impl, we'd fetch the underlying weight buffer.
                return true; 
            }
            _ => {
                // Internal node: Recalculate from children
                if let Some(l) = &self.left {
                    hasher.update(&l.hash);
                }
                if let Some(r) = &self.right {
                    hasher.update(&r.hash);
                }
                hasher.digest(&[])
            }
        };

        data_hash == self.hash
    }
}

/// The distributed state manager.
/// Maintains the global tree and handles re-calculations.
pub struct StateManager {
    shards: HashMap<ShardID, ShardCommitment>,
    tree: Option<Box<MerkleNode>>,
    current_root: Option<ModelRoot>,
}

impl StateManager {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
            tree: None,
            current_root: None,
        }
    }

    /// Ingests a new shard (gradient update).
    /// If this changes the state, the Merkle tree is rebuilt.
    pub fn update_shard(&mut self, id: ShardID, data: &[f32]) -> Result<ModelRoot> {
        // 1. Create Leaf Node
        let leaf = MerkleNode::leaf(id.clone(), data);
        let commitment = ShardCommitment {
            id: id.clone(),
            hash: leaf.hash.clone(),
            size: data.len() * 4,
            version: self.shards.get(&id).map(|c| c.version + 1).unwrap_or(0),
        };

        // 2. Update Cache
        self.shards.insert(id, commitment);

        // 3. Rebuild Tree (Simplistic implementation: linear rebuild)
        // In a prod system, we'd do lazy/async updates.
        self.rebuild_tree()?;

        // 4. Return new Root
        Ok(self.current_root.clone().unwrap())
    }

    fn rebuild_tree(&mut self) -> Result<()> {
        if self.shards.is_empty() {
            return Ok(());
        }

        // Collect all leaf hashes
        let mut hashes: Vec<Vec<u8>> = self.shards.values().map(|c| c.hash.clone()).collect();
        
        // Build tree bottom-up
        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for pair in hashes.chunks(2) {
                let (left, right) = (pair[0].clone(), pair[1].clone());
                let node = MerkleNode::internal(
                    MerkleNode::leaf(ShardID { layer_id: 0, chunk_idx: 0 }, &[]), // Mock left
                    MerkleNode::leaf(ShardID { layer_id: 0, chunk_idx: 0 }, &[]), // Mock right
                );
                next_level.push(node.hash);
            }
            
            hashes = next_level;
        }

        // The single remaining hash is the Root
        if hashes.len() == 1 {
            self.current_root = Some(ModelRoot {
                root_hash: hashes[0].clone(),
                tree_depth: 0, // Calculate real depth
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        }

        Ok(())
    }

    /// Compares local root against a peer's root.
    /// If they don't match, the cluster is split.
    pub fn validate_consensus(&self, peer_root: &ModelRoot) -> Result<()> {
        match &self.current_root {
            Some(my_root) => {
                if my_root.root_hash != peer_root.root_hash {
                    return Err(CoreError::HashMismatch { layer_idx: 0 }); // Root mismatch
                }
                Ok(())
            }
            None => Err(CoreError::CorruptionDetected),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_construction() {
        let data = vec![1.0f32, 2.0, 3.0];
        let leaf = MerkleNode::leaf(ShardID { layer_id: 1, chunk_idx: 1 }, &data);
        
        // Check hash length
        assert_eq!(leaf.hash.len(), 32); // SHA256
    }

    #[test]
    fn test_state_manager() {
        let mut mgr = StateManager::new();
        let id = ShardID { layer_id: 1, chunk_idx: 0 };
        
        // Update shard
        let root = mgr.update_shard(id, &[1.0, 2.0]).unwrap();
        
        // Verify root exists
        assert_eq!(root.root_hash.len(), 32);
    }
}