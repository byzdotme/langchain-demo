
import { StateGraph, END } from "@langchain/langgraph";
import { deepseekChat } from './model_helper.js';

/**
 * @fileoverview 这是一个使用 @langchain/langgraph 实现的自动化研究助手的示例。
 * 
 * ## 核心概念 ##
 * 
 * 1.  **图 (Graph):** 整个研究流程被定义为一个“图”，它由多个“节点”和连接它们的“边”组成。
 * 2.  **状态 (State):** 一个在图的执行过程中持续传递和更新的对象。它像一个“活的”数据容器，
 *     每个节点都可以读取它的当前值，并写入新的结果，从而影响后续节点的行为。
 *     我们在这里使用 `ResearchState` 接口来定义它的结构。
 * 3.  **节点 (Nodes):** 图中的基本执行单元，通常是一个函数或一个 Runnable。
 *     我们将研究流程的每个步骤（规划、搜索、总结、报告）都定义为一个节点。
 *     每个节点接收当前的状态对象，执行特定任务，然后返回一个包含更新信息的部分状态对象。
 * 4.  **边 (Edges):** 连接节点的路径，决定了工作流的走向。
 *     - **普通边 (addEdge):** 定义了固定的、从 A 到 B 的流程。
 *     - **条件边 (addConditionalEdges):** 这是实现循环和决策的关键。它连接一个“决策节点”，
 *       该节点根据当前状态的内容（例如，LLM的判断结果）来动态选择下一条路径（例如，是“继续研究”还是“结束并生成报告”）。
 * 
 * ## 工作流程 ##
 * 
 * [入口] -> planner -> searcher -> summarizer -> [should_continue (决策)]
 *                                                       |
 *      +------------------------------------------------+
 *      | (如果需要继续)                                  | (如果信息足够)
 *      v                                                v
 *   planner (循环回起点)                           report_writer -> [出口/END]
 * 
 */

// ================ 1. 定义状态接口 ================
/**
 * ResearchState 接口定义了在整个图执行过程中需要追踪的所有数据。
 * 它就像一个共享的记忆板，每个节点都能读写。
 * - topic: 研究主题，由用户输入，全程不变。
 * - plan: 当前的研究计划或搜索关键词。
 * - searchResults: 模拟的搜索结果列表。
 * - summary: 对当前所有信息的综合总结。
 * - report: 最终生成的完整报告。
 * - iterationCount: 迭代次数，用于防止无限循环。
 */
interface ResearchState {
  topic: string;
  plan: string;
  searchResults?: string[];
  summary: string;
  report?: string;
  iterationCount: number;
}

// ================ 2. 定义图的节点 ================
// 每个节点都是一个异步函数，接收当前状态，返回一个包含更新字段的部分状态。

/**
 * 规划节点 (Planner Node)
 * @param state 当前状态
 * @returns 返回更新后的 plan 和 iterationCount
 */
async function plannerNode(state: ResearchState): Promise<ResearchState> {
  console.log("--- 步骤: 规划 (Planner) ---");
  const planPrompt = `
作为专业研究分析师，请为以下研究项目制定下一步的搜索计划：

研究主题: ${state.topic}
当前轮次: ${state.iterationCount + 1}
已有总结: ${state.summary || '暂无'}

请生成3-5个具体的搜索关键词或研究方向，用于下一轮信息收集。
要求：
1. 关键词应该具体且有针对性，能够补充现有信息的不足。
2. 覆盖技术、应用、市场等不同维度。
3. 以逗号分隔的格式返回关键词。
示例格式：人工智能医疗诊断,机器学习疾病检测,AI辅助诊断系统,医疗AI应用案例
`;

  const response = await deepseekChat.invoke([{ role: 'user', content: planPrompt }]);
  const plan = response.content as string;

  console.log(`[规划完成] 新的研究计划: ${plan.trim()}`);
  return {
    ...state,
    plan: plan.trim(),
    iterationCount: state.iterationCount + 1
  };
}

/**
 * 搜索节点 (Searcher Node) - 模拟
 * @param state 当前状态
 * @returns 返回模拟的 searchResults
 */
async function searcherNode(state: ResearchState): Promise<ResearchState> {
  console.log("--- 步骤: 搜索 (Searcher) ---");
  console.log(`[正在搜索] 根据计划: "${state.plan}"`);
  
  // 模拟搜索延迟
  await new Promise(resolve => setTimeout(resolve, 1000));

  // 生成模拟搜索结果
  const mockResults = state.plan.split(',').map(keyword => 
    `关于"${keyword.trim()}"的模拟搜索结果：最新研究论文、市场分析报告和行业新闻。`
  );
  
  console.log(`[搜索完成] 找到 ${mockResults.length} 条模拟信息`);
  return { ...state, searchResults: mockResults };
}

/**
 * 总结节点 (Summarizer Node)
 * @param state 当前状态
 * @returns 返回更新后的 summary
 */
async function summarizerNode(state: ResearchState): Promise<ResearchState> {
  console.log("--- 步骤: 总结 (Summarizer) ---");
  
  const summarizePrompt = `
作为专业研究分析师，请对以下信息进行深度分析和总结：

研究主题: ${state.topic}
已有总结: ${state.summary || '暂无'}
新搜索结果:
${(state.searchResults || []).map((result, index) => `${index + 1}. ${result}`).join('\n')}

请提供一个综合性的分析总结，整合新旧信息，突出重点。
要求：
- 信息整合度高，逻辑清晰。
- 与已有信息形成有机补充。
`;

  const response = await deepseekChat.invoke([{ role: 'user', content: summarizePrompt }]);
  const newSummary = response.content as string;
  
  console.log("[总结完成] 已更新研究摘要。");
  return { ...state, summary: newSummary.trim() };
}

/**
 * 报告生成节点 (Report Writer Node)
 * @param state 当前状态
 * @returns 返回最终的 report
 */
async function reportWriterNode(state: ResearchState): Promise<ResearchState> {
  console.log("--- 步骤: 生成报告 (Report Writer) ---");

  const reportPrompt = `
作为资深研究分析师，请基于以下研究信息生成一份完整的专业研究报告：

研究主题: ${state.topic}
最终总结: ${state.summary}
总研究轮次: ${state.iterationCount}

请生成一份结构化的研究报告，包含执行摘要、技术分析、市场前景、挑战与建议等部分。
要求：报告内容专业、全面、有深度，逻辑清晰。
`;

  const response = await deepseekChat.invoke([{ role: 'user', content: reportPrompt }]);
  const report = response.content as string;
  
  console.log("[报告生成] 最终研究报告已创建。");
  return { ...state, report: report.trim() };
}

// ================ 3. 定义条件边逻辑 ================

/**
 * 决策函数 (shouldContinue)
 * 这是实现条件边的核心。它会调用 LLM 来判断当前信息是否足够。
 * @param state 当前状态
 * @returns 返回一个字符串 "continue" 或 "end"，这个字符串将决定图的下一跳路径。
 */
async function shouldContinueNode(state: ResearchState): Promise<"continue" | "end"> {
  console.log("--- 步骤: 决策 (Decision) ---");
  
  // 简单的规则：防止无限循环
  if (state.iterationCount >= 3) {
    console.log("[决策] 已达到最大迭代次数(3)，将结束研究。");
    return "end";
  }

  const evaluationPrompt = `
作为研究质量评估专家，请评估当前研究信息的完整性：

研究主题: ${state.topic}
研究轮次: ${state.iterationCount}
当前总结: ${state.summary}

评估标准：信息的全面性、深度、是否覆盖了技术/市场/应用等关键方面。
根据以上标准，当前信息是否足够生成一份高质量的专业研究报告？

请只回答以下两个词之一：
- "continue": 如果信息还不够充分，需要继续研究。
- "finish": 如果信息已足够，可以生成报告。
`;
  
  const response = await deepseekChat.invoke([{ role: 'user', content: evaluationPrompt }]);
  const decision = response.content as string;
  
  if (decision.toLowerCase().includes('continue')) {
    console.log("[决策] 结论: 信息不足，继续研究。");
    return "continue";
  } else {
    console.log("[决策] 结论: 信息充足，准备生成报告。");
    return "end";
  }
}

// ================ 4. 构建图 ================

async function main() {
  console.log("🤖 欢迎使用 LangGraph 自动化研究助手！\n");
  
  // 研究主题
  const researchTopic = "AI 在医疗诊断领域的应用前景";
  
  // 创建一个 StateGraph 实例。由于我们将让每个节点返回完整的状态，
  // 我们可以使用更简单的构造函数，这通常可以避免复杂的类型问题。
  const workflow = new StateGraph<ResearchState>();

  // **添加节点 (addNode)**
  // 将我们定义的函数作为节点添加到图中，并为每个节点指定一个唯一的名称。
  workflow.addNode("planner", plannerNode);
  workflow.addNode("searcher", searcherNode);
  workflow.addNode("summarizer", summarizerNode);
  workflow.addNode("report_writer", reportWriterNode);
  workflow.addNode("should_continue", shouldContinueNode);

  // **设置入口点 (setEntryPoint)**
  // 指定图从哪个节点开始执行。
  workflow.setEntryPoint("planner");
  
  // **添加普通边 (addEdge)**
  // 连接具有确定性流程的节点。
  workflow.addEdge("planner", "searcher");
  workflow.addEdge("searcher", "summarizer");
  workflow.addEdge("summarizer", "should_continue");
  
  // **添加条件边 (addConditionalEdges)**
  // 这是图的核心决策点。
  // - source: "should_continue" 是我们的决策节点。
  // - path: (decision: "continue" | "end") => decision 这是一个函数，它接收决策节点的输出（"continue"或"end"），
  //   并直接将其作为下一跳节点的名称返回。
  // - pathMap: 这是一个映射，将 `path` 函数返回的字符串值映射到具体的节点名称。
  //   - 如果返回 "continue"，工作流跳转到 "planner" 节点，形成循环。
  //   - 如果返回 "end"，工作流跳转到 "report_writer" 节点。
  workflow.addConditionalEdges("should_continue", 
    (decision: "continue" | "end") => decision,
    {
      "continue": "planner",
      "end": "report_writer"
    }
  );

  // **设置出口点**
  // 当流程走到 "report_writer" 后，我们希望它结束。
  // `END` 是一个特殊的标记，表示图的执行到此为止。
  workflow.addEdge("report_writer", END);

  // **编译图 (compile)**
  // 将定义好的图结构编译成一个可执行的 Runnable 对象。
  const app = workflow.compile();

  // **执行图 (stream)**
  // 我们使用 `.stream()` 方法来执行图，这样可以实时观察每一步的状态变化。
  const initialState: ResearchState = {
    topic: researchTopic,
    plan: "",
    summary: "",
    iterationCount: 0,
  };

  console.log(`🚀 开始研究任务，主题: "${researchTopic}"`);
  console.log("========================================\n");
  
  const stream = await app.stream(initialState, {
    recursionLimit: 10, // 设置递归限制以允许循环
  });

  let fullState: ResearchState | undefined;

  for await (const chunk of stream) {
    const [nodeName] = Object.keys(chunk);
    // @ts-ignore
    const nodeOutput = chunk[nodeName];

    // 打印出当前是哪个节点在运行及其输出
    console.log(`\n▶️  节点 [${nodeName}] 已执行完毕`);
    console.log("----------------------------------------");
    console.log("📝 节点输出 (完整状态):");
    console.log(JSON.stringify(nodeOutput, null, 2));

    // 由于每个节点现在都返回完整状态，我们可以直接将其赋值给 fullState
    fullState = nodeOutput;
  }
  
  console.log("\n========================================");
  console.log("✅ 研究任务全部完成！");
  console.log("========================================");
  
  if (fullState && fullState.report) {
    console.log("\n📄 最终研究报告:\n");
    console.log(fullState.report);
  } else {
    console.log("\n⚠️ 未能生成最终报告。");
    console.log("最终状态:", fullState);
  }
}

// 执行主函数
main().catch(e => console.error("💥 程序执行失败:", e)); 