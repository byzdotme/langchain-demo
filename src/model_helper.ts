import { OllamaEmbeddings } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";

export function newDeepseekChat(
  temperature: number = 0.7,
  reasoner: boolean = true
): ChatOpenAI {
  let model = reasoner ? "deepseek-chat" : "deepseek-reasoner";
  return new ChatOpenAI({
    temperature,
    model,
    configuration: {
      baseURL: "https://api.deepseek.com",
      apiKey: process.env["DEEPSEEK_API_KEY"] || "",
    },
  });
}

export const deepseekChat: ChatOpenAI = newDeepseekChat(0.7, false);

const OLLAMA_BASE_URL: string = "http://localhost:11434"; // Ollama 服务地址
const EMBEDDINGS_MODEL: string = "nomic-embed-text"; // 用于生成嵌入的向量模型

export const ollamaEmbeddings: OllamaEmbeddings = new OllamaEmbeddings({
  model: EMBEDDINGS_MODEL,
  baseUrl: OLLAMA_BASE_URL,
});
