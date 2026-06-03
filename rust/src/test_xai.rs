use super::test_data::{run_pdf, run_image, run_toolcall, run_toolcall_streaming};

// https://docs.x.ai/llms.txt             (text only good for models)
// https://docs.x.ai/developers/models    (prices, including tool call prices)

const MODEL: &str = "grok-4-1-fast-reasoning";
// const MODEL: &str = "grok-4-1-fast-non-reasoning";
const PROV: &str = "xai";

#[tokio::test]
async fn test_pdf_xai() { run_pdf(MODEL, PROV).await; }

#[tokio::test]
async fn test_image_xai() { run_image(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_xai() { run_toolcall(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_xai_stream() { run_toolcall_streaming(MODEL, PROV).await; }
