import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import dotenv from 'dotenv';

dotenv.config();

// 读取命令行参数
const topic = process.argv[2];
if (!topic) {
  console.error("用法: node haiku_poet.js <主题>");
  process.exit(1);
}

// 构建 prompt
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "你是一位擅长创作俳句的诗人。请根据用户给定的主题，创作一首符合 5-7-5 音节格式的中文俳句。只输出俳句正文，不要输出其他内容。"
  ],
  ["human", "主题：{topic}"]
]);

// 初始化模型
const model = new ChatOpenAI({
  temperature: 0.7,
  model: 'deepseek-chat',
  configuration: {
    baseURL: 'https://api.deepseek.com',
    apiKey: process.env["DEEPSEEK_API_KEY"] || '',
  }
});

// 串联 prompt -> model -> output parser
const chain = prompt.pipe(model).pipe(new StringOutputParser());

// 执行链
(async () => {
  try {
    const haiku = await chain.invoke({ topic });
    console.log(haiku.trim());
  } catch (err) {
    console.error("生成俳句失败:", err);
    process.exit(2);
  }
})();
