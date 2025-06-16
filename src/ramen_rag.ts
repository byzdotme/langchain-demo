// 导入 LangChain 的核心模块和类型
import { OllamaEmbeddings } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "langchain/document"; // 导入 Document 类型

// --- 配置信息 ---
const RAMEN_REVIEWS_FILE: string = "./ramen_reviews.txt"; // 文档路径
const OLLAMA_BASE_URL: string = "http://localhost:11434"; // Ollama 服务地址
const CHAT_MODEL: string = "qwen:7b"; // 用于聊天的模型
const EMBEDDINGS_MODEL: string = "nomic-embed-text"; // 用于生成嵌入的向量模型

/**
 * 主函数，运行整个 RAG 流程
 */
async function main(): Promise<void> {

  console.log("🍜 开始构建拉面店顾问 RAG 应用 (TypeScript 版本)...");

  // 1. 加载文档 (Load)
  // ------------------------------------------
  console.log(`\n[步骤 1] 从 '${RAMEN_REVIEWS_FILE}' 加载文档...`);
  const loader: TextLoader = new TextLoader(RAMEN_REVIEWS_FILE);
  const docs: Document[] = await loader.load();
  console.log(`  加载了 ${docs.length} 篇文档。`);

  // 2. 分割文档 (Split)
  // ------------------------------------------
  console.log("\n[步骤 2] 将文档分割成小块...");
  const splitter: RecursiveCharacterTextSplitter =
    new RecursiveCharacterTextSplitter({
      chunkSize: 500, // 每个块的最大字符数
      chunkOverlap: 50, // 相邻块之间的重叠字符数，以保证语义连续性
    });
  const splitDocs: Document[] = await splitter.splitDocuments(docs);
  console.log(`  文档被分割成 ${splitDocs.length} 个小块。`);

  // 3. 初始化模型 (Initialize Models)
  // ------------------------------------------
  console.log("\n[步骤 3] 初始化 Ollama 模型...");
  const embeddings: OllamaEmbeddings = new OllamaEmbeddings({
    model: EMBEDDINGS_MODEL,
    baseUrl: OLLAMA_BASE_URL,
  });
  console.log(`  嵌入模型: ${EMBEDDINGS_MODEL}`);

  const llm: ChatOpenAI = new ChatOpenAI({
    temperature: 0.7,
    model: "deepseek-chat",
    configuration: {
      baseURL: "https://api.deepseek.com",
      apiKey: process.env["DEEPSEEK_API_KEY"] || "",
    },
  });
  console.log(`  聊天模型: ${CHAT_MODEL}`);

  // 4. 创建向量存储和检索器 (Embed & Store & Retrieve)
  // ------------------------------------------
  console.log("\n[步骤 4] 创建向量存储和检索器...");
  const vectorstore: MemoryVectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  console.log("  内存向量存储创建成功。");

  const retriever = vectorstore.asRetriever({
    k: 3, // 设置为返回最相关的3个文档块
  });
  console.log("  检索器创建成功，将返回最相关的3个文档块。");

  // 5. 构建 RAG 链 (Chain)
  // ------------------------------------------
  console.log("\n[步骤 5] 构建 RAG 链...");

  const promptTemplate: string = `
你是一个专业的拉面店顾问。请根据下面提供的“上下文信息”，用中文简洁地回答用户的问题。
你的回答必须完全基于上下文信息，不要编造任何内容。
如果上下文中没有足够的信息来回答问题，请直接说“根据我所知的信息，无法回答这个问题”。

上下文信息:
{context}

用户问题:
{question}

你的回答:
`;
  const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);

  // 定义如何将检索到的文档格式化为字符串，并显式声明类型
  const formatDocs = (docs: Document[]): string => {
    return docs
      .map((doc, i) => `--- 文档 ${i + 1} ---\n${doc.pageContent}`)
      .join("\n\n");
  };

  // 使用 LCEL 构建链，并为链指定输入和输出类型
  // RunnableSequence<string, string> 表示这个链接收一个字符串输入，并返回一个字符串输出
  const chain: RunnableSequence<string, string> = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);
  console.log("  RAG 链构建完成！");

  // 6. 执行链并提问 (Invoke)
  // ------------------------------------------
  console.log("\n[步骤 6] 执行链并开始提问...");

  const questions: string[] = [
    "一乐拉面有没有素食选项？",
    "哪家店以沾面出名，价格大概多少？",
    "我想吃点辣的，有什么推荐吗？",
    "秋叶原附近有什么推荐的拉面店吗？",
    "面屋武藏的汤底是什么样的？",
  ];

  for (const question of questions) {
    console.log("\n==============================================");
    console.log(`🤔 提问: ${question}`);
    const result: string = await chain.invoke(question);
    console.log(`🍜 回答: ${result}`);
  }
  console.log("\n==============================================");
  console.log("\n所有问题回答完毕！");
}

// 运行主函数
main().catch(console.error);
