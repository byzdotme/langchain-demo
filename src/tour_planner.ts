/*
 * 旅行日程规划器 - TypeScript 实现
 * 
 * 本脚本展示了如何使用 Zod 和 LangChain.js 的 .withStructuredOutput() 方法
 * 来创建一个结构化输出的旅行规划工具。
 * 
 * 依赖安装：
 * npm install zod
 * 
 * 注意：项目已包含 @langchain/openai langchain 等必要的 LangChain 依赖
 */

import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { deepseekChat } from "./model_helper.js";

// 使用 Zod 定义旅行计划的结构化 Schema
// Zod 是一个强类型的 Schema 验证库，它提供了以下优势：
// 1. 编译时类型安全：确保数据结构的正确性
// 2. 运行时验证：验证 LLM 返回的数据是否符合预期格式
// 3. 自动类型推断：从 Schema 自动生成 TypeScript 类型
// 4. 详细的错误信息：当数据不符合 Schema 时提供清晰的错误描述
const itinerarySchema = z.object({
  destination: z.string().describe("旅行目的地"),
  duration_days: z.number().describe("总旅行天数"),
  daily_plans: z.array(
    z.object({
      day: z.number().describe("第几天"),
      theme: z.string().describe("当天的主题，例如：'城市探索'或'自然风光'"),
      activities: z.array(z.string()).describe("当天具体的活动安排列表"),
    })
  ).describe("每日的详细计划列表"),
  estimated_cost_usd: z.number().describe("预估的总花费（美元）")
});

// 从 Zod Schema 自动推断 TypeScript 类型
// 这确保了我们的代码在编译时就有正确的类型信息
type TravelItinerary = z.infer<typeof itinerarySchema>;

/**
 * 创建旅行规划链
 * 
 * 为什么使用 .withStructuredOutput() 而不是传统方法？
 * 
 * 1. 相比于 prompt engineering：
 *    - prompt engineering 依赖于精心设计的提示词来"引导"模型输出特定格式
 *    - 但模型仍可能产生格式不一致或包含额外文本的输出
 *    - .withStructuredOutput() 在模型层面强制执行结构化输出
 * 
 * 2. 相比于 JsonOutputParser：
 *    - JsonOutputParser 只是后处理步骤，尝试从模型输出中解析 JSON
 *    - 如果模型输出不是有效 JSON，解析会失败
 *    - .withStructuredOutput() 确保模型直接生成符合 Schema 的结构化数据
 * 
 * 3. 现代化优势：
 *    - 利用了最新 LLM 的结构化输出能力（function calling）
 *    - 提供更好的可靠性和一致性
 *    - 减少了后处理的复杂性
 */
function createTravelPlannerChain() {
  // 创建聊天提示模板
  // 这个模板定义了我们如何向 LLM 描述任务
  const prompt = ChatPromptTemplate.fromTemplate(`
    你是一个专业的旅行规划师。请根据用户的需求，制定一个详细的旅行计划。

    用户需求：{request}

    请提供一个结构化的旅行计划，包括：
    1. 明确的目的地
    2. 旅行天数
    3. 每日的详细安排，包括主题和具体活动
    4. 预估的总费用（美元）

    请确保计划实际可行，活动安排合理，时间分配恰当。
    成本估算应该包括住宿、交通、餐饮、门票等主要开支。
  `);

  // 使用 .withStructuredOutput() 方法将 Zod Schema 绑定到模型
  // 这是 LangChain.js 中实现结构化输出的现代化、推荐方式
  // 它利用了底层模型的 function calling 或 structured output 能力
  const structuredModel = deepseekChat.withStructuredOutput(itinerarySchema);

  // 使用 .pipe() 方法将提示模板和结构化模型连接起来
  // 这创建了一个完整的处理链：输入 -> 格式化提示 -> 模型推理 -> 结构化输出
  const chain = prompt.pipe(structuredModel);

  return chain;
}

/**
 * 主函数：演示旅行规划器的使用
 */
async function main() {
  try {
    console.log("🌍 旅行日程规划器启动中...\n");

    // 创建旅行规划链
    const plannerChain = createTravelPlannerChain();

    // 测试用例：为期 3 天的京都深度游
    const testRequest = "请帮我规划一个为期3天的京都深度游，主题是寺庙与庭园";
    
    console.log(`📝 用户需求：${testRequest}\n`);
    console.log("🤖 正在生成旅行计划...\n");

    // 调用链并获取结构化输出
    const result: TravelItinerary = await plannerChain.invoke({
      request: testRequest
    });

    // 输出结果
    console.log("✅ 旅行计划生成完成！\n");
    console.log("📋 结构化输出结果：");
    console.log("=".repeat(50));
    console.log(JSON.stringify(result, null, 2));
    console.log("=".repeat(50));

    // 验证结果的类型安全性
    console.log("\n🔍 数据验证：");
    console.log(`✓ 目的地：${result.destination}`);
    console.log(`✓ 旅行天数：${result.duration_days} 天`);
    console.log(`✓ 每日计划数量：${result.daily_plans.length} 天`);
    console.log(`✓ 预估费用：$${result.estimated_cost_usd} USD`);

    // 展示每日详细计划
    console.log("\n📅 每日详细计划：");
    result.daily_plans.forEach(plan => {
      console.log(`\n第 ${plan.day} 天 - ${plan.theme}:`);
      plan.activities.forEach((activity, index) => {
        console.log(`  ${index + 1}. ${activity}`);
      });
    });

  } catch (error) {
    console.error("❌ 规划过程中发生错误：", error);
    
    // 如果是 Zod 验证错误，提供详细信息
    if (error instanceof z.ZodError) {
      console.error("📋 Schema 验证错误详情：");
      error.errors.forEach(err => {
        console.error(`  - ${err.path.join('.')}: ${err.message}`);
      });
    }
  }
}

// 如果直接运行此文件，执行主函数
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

// 导出主要组件供其他模块使用
export { 
  itinerarySchema, 
  createTravelPlannerChain,
  type TravelItinerary 
};
