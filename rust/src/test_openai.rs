use super::test_data::{run_pdf, run_image, run_toolcall, run_toolcall_streaming};

const MODEL: &str = "gpt-5.4";
const PROV: &str = "openai";

#[tokio::test]
async fn test_pdf_openai() { run_pdf(MODEL, PROV).await; }

#[tokio::test]
async fn test_image_openai() { run_image(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_openai() { run_toolcall(MODEL, PROV).await; }

#[tokio::test]
async fn test_toolcall_openai_stream() { run_toolcall_streaming(MODEL, PROV).await; }
