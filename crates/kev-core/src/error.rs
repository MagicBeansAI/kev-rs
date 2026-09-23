use thiserror::Error;

#[derive(Debug, Error)]
pub enum KevError {
    #[error("context overflow: {0}")]
    ContextOverflow(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("model load: {0}")]
    Load(String),
    #[error("inference: {0}")]
    Inference(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, KevError>;
