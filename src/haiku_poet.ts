import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { deepseekChat } from "./model_helper.js";

async function main(): Promise<void> {
  // 读取命令行参数
  const topic = process.argv[2];
  if (!topic) {
    throw "用法: haiku_poet <主题>";
  }

  // 构建 prompt
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一位擅长创作俳句的诗人。请根据用户给定的主题，创作一首符合 5-7-5 音节格式的中文俳句。只输出俳句正文，不要输出其他内容。",
    ],
    ["human", "主题：{topic}"],
  ]);

  // 初始化模型
  const model = deepseekChat;

  // 串联 prompt -> model -> output parser
  const chain = prompt.pipe(model).pipe(new StringOutputParser());

  // 执行链
  const haiku = await chain.invoke({ topic });
  console.log(haiku.trim());
}

main().catch(console.error);
