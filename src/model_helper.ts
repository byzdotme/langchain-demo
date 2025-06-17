import { OllamaEmbeddings } from '@langchain/ollama';
import { ChatOpenAI } from '@langchain/openai';

export function newDeepseekChat(reasoner = true, temperature = 0.5): ChatOpenAI {
  const model = reasoner ? 'deepseek-reasoner' : 'deepseek-chat';
  return new ChatOpenAI({
    temperature,
    model,
    configuration: {
      baseURL: 'https://api.deepseek.com',
      apiKey: process.env['DEEPSEEK_API_KEY'] || '',
    },
  });
}

export const deepseekChat: ChatOpenAI = newDeepseekChat(false);

const OLLAMA_BASE_URL = 'http://localhost:11434'; // Ollama 服务地址
const EMBEDDINGS_MODEL = 'nomic-embed-text'; // 用于生成嵌入的向量模型

export const ollamaEmbeddings: OllamaEmbeddings = new OllamaEmbeddings({
  model: EMBEDDINGS_MODEL,
  baseUrl: OLLAMA_BASE_URL,
});
