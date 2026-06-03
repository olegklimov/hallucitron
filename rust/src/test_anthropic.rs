use super::test_data::{run_pdf, run_image, run_toolcall, run_toolcall_streaming, run_toolcall_streaming_thinking};

const MODEL: &str = "claude-sonnet-4-6";
const PROV: &str = "anthropic";

#[tokio::test]
async fn test_pdf_anthropic() { run_pdf(MODEL, PROV).await; }

#[tokio::test]
async fn test_image_anthropic() { run_image(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_anthropic() { run_toolcall(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_anthropic_stream() { run_toolcall_streaming(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_anthropic_stream_thinking() { run_toolcall_streaming_thinking(MODEL, PROV).await; }
