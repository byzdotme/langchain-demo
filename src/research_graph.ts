import { deepseekChat } from './model_helper.js';
import { DynamicTool } from '@langchain/community/tools/dynamic';
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents';
import { ChatPromptTemplate } from '@langchain/core/prompts';

/**
 * 自动化研究分析师实现
 * 
 * 这个项目演示如何使用 LangChain Agent 构建一个有状态、能循环、有决策能力的复杂研究助手。
 * 
 * 核心特性：
 * 1. 状态管理：维护研究过程中的各种信息
 * 2. 循环决策：根据当前信息质量决定是否继续研究
 * 3. 智能规划：动态制定搜索关键词和研究方向
 * 4. 自动总结：对收集的信息进行智能分析和整合
 * 5. 报告生成：基于所有信息生成专业研究报告
 */

// ================ 研究状态管理 ================

/**
 * 研究状态类 - 管理整个研究过程的状态信息
 * 这是实现有状态 Agent 的核心，包含了研究过程中需要持续追踪的所有信息
 */
class ResearchState {
  topic = '';                    // 研究主题
  plan = '';                     // 当前研究计划
  searchResults: string[] = [];          // 搜索结果集合
  summary = '';                  // 研究总结
  report = '';                   // 最终报告
  iterationCount = 0;            // 迭代计数
  isComplete = false;           // 研究是否完成

  constructor(topic: string) {
    this.topic = topic;
  }

  /**
   * 获取当前状态的字符串表示
   * 用于在工具调用之间传递状态信息
   */
  toString(): string {
    return JSON.stringify({
      topic: this.topic,
      plan: this.plan,
      searchResults: this.searchResults,
      summary: this.summary,
      report: this.report,
      iterationCount: this.iterationCount,
      isComplete: this.isComplete
    }, null, 2);
  }

  /**
   * 从字符串恢复状态
   */
  static fromString(stateStr: string): ResearchState {
    const data = JSON.parse(stateStr);
    const state = new ResearchState(data.topic);
    Object.assign(state, data);
    return state;
  }
}

// 全局状态实例
let currentResearchState: ResearchState;

// ================ 研究工具定义 ================

/**
 * 工具1：研究规划器
 * 根据当前研究状态制定下一步的搜索计划和关键词
 */
const plannerTool = new DynamicTool({
  name: "research_planner",
  description: `
制定研究计划和搜索关键词。
使用场景：
- 开始新的研究项目时
- 当前信息不足需要继续收集时
- 需要调整研究方向时
输入：当前研究状态的JSON字符串
输出：包含搜索关键词的研究计划
`,
  func: async (input: string) => {
    console.log('🔍 [规划工具] 正在制定研究计划...');
    
    // 处理传入的参数，可能是直接的JSON字符串或包装的对象
    let stateJson = input;
    try {
      const parsed = JSON.parse(input);
      if (parsed.input) {
        stateJson = parsed.input;
      }
    } catch {
      // 如果解析失败，使用原始输入
    }
    
    const state = ResearchState.fromString(stateJson);
    
    const planPrompt = `
作为专业研究分析师，请为以下研究项目制定下一步的搜索计划：

研究主题: ${state.topic}
当前轮次: ${state.iterationCount + 1}
已有总结: ${state.summary || '暂无'}

请生成3-5个具体的搜索关键词或研究方向，用于下一轮信息收集。
要求：
1. 关键词应该具体且有针对性
2. 能够补充现有信息的不足
3. 覆盖技术、应用、市场等不同维度

请以逗号分隔的格式返回关键词。
示例格式：人工智能医疗诊断,机器学习疾病检测,AI辅助诊断系统,医疗AI应用案例
`;

    try {
      const response = await deepseekChat.invoke([{ role: 'user', content: planPrompt }]);
      const plan = response.content as string;
      
      // 更新状态
      state.plan = plan.trim();
      state.iterationCount += 1;
      currentResearchState = state;
      
      console.log(`📋 [规划完成] ${plan.trim()}`);
      
      return `研究计划已制定。第${state.iterationCount}轮研究关键词：${plan.trim()}`;
      
    } catch (error) {
      console.error('规划工具执行出错:', error);
      const fallbackPlan = `关于"${state.topic}"的基础研究,相关技术分析,应用前景评估`;
      state.plan = fallbackPlan;
      state.iterationCount += 1;
      currentResearchState = state;
      
      return `研究计划已制定（备用方案）。关键词：${fallbackPlan}`;
    }
  }
});

/**
 * 工具2：信息搜索器
 * 根据研究计划执行信息搜索（模拟）
 */
const searcherTool = new DynamicTool({
  name: "information_searcher", 
  description: `
执行信息搜索任务。
使用场景：在制定了研究计划后，根据关键词搜索相关信息
输入：当前研究状态的JSON字符串
输出：搜索到的相关信息列表
`,
  func: async (input: string) => {
    console.log('🔎 [搜索工具] 正在执行信息搜索...');
    
    let stateJson = input;
    try {
      const parsed = JSON.parse(input);
      if (parsed.input) {
        stateJson = parsed.input;
      }
    } catch {
      // 如果解析失败，使用原始输入
    }
    
    const state = ResearchState.fromString(stateJson);
    
    // 模拟搜索延迟
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // 生成模拟搜索结果
    const mockResults = generateMockSearchResults(state.topic, state.plan);
    
    // 更新状态
    state.searchResults = mockResults;
    currentResearchState = state;
    
    console.log(`📄 [搜索完成] 找到 ${mockResults.length} 条相关信息`);
    
    return `搜索完成。找到${mockResults.length}条相关信息：\n${mockResults.map((item, idx) => `${idx + 1}. ${item}`).join('\n')}`;
  }
});

/**
 * 工具3：信息总结器
 * 对搜索结果进行智能分析和总结
 */
const summarizerTool = new DynamicTool({
  name: "information_summarizer",
  description: `
对搜索信息进行智能总结分析。
使用场景：在完成信息搜索后，需要对收集的信息进行整理和分析
输入：当前研究状态的JSON字符串
输出：整合后的研究总结
`,
  func: async (input: string) => {
    console.log('📝 [总结工具] 正在总结搜索结果...');
    
    let stateJson = input;
    try {
      const parsed = JSON.parse(input);
      if (parsed.input) {
        stateJson = parsed.input;
      }
    } catch {
      // 如果解析失败，使用原始输入
    }
    
    const state = ResearchState.fromString(stateJson);
    
    if (!state.searchResults || state.searchResults.length === 0) {
      return '没有搜索结果可以总结';
    }

    const summarizePrompt = `
作为专业研究分析师，请对以下搜索结果进行深度分析和总结：

研究主题: ${state.topic}
已有总结: ${state.summary || '暂无'}

新搜索结果:
${state.searchResults.map((result, index) => `${index + 1}. ${result}`).join('\n')}

请提供一个综合性分析总结，包含：
1. 主要技术发现和关键信息点
2. 当前技术发展趋势和方向
3. 实际应用场景和商业价值
4. 面临的主要挑战和技术难点
5. 未来发展前景和建议

要求：
- 信息整合度高，逻辑清晰
- 突出重点和亮点
- 与已有信息形成有机补充
- 为后续研究提供方向指引
`;

    try {
      const response = await deepseekChat.invoke([{ role: 'user', content: summarizePrompt }]);
      const newSummary = response.content as string;
      
      // 更新状态
      state.summary = newSummary.trim();
      currentResearchState = state;
      
      console.log('📋 [总结完成] 研究总结已更新');
      
      return `信息总结完成。更新后的研究总结：\n${newSummary.trim()}`;
      
    } catch (error) {
      console.error('总结工具执行出错:', error);
      const fallbackSummary = state.searchResults.join(' | ');
      const updatedSummary = state.summary ? 
        `${state.summary}\n\n新增信息: ${fallbackSummary}` : 
        fallbackSummary;
      
      state.summary = updatedSummary;
      currentResearchState = state;
      
      return `信息总结完成（简化版）：${updatedSummary}`;
    }
  }
});

/**
 * 工具4：研究质量评估器
 * 评估当前研究信息的完整性，决定是否继续研究
 */
const evaluatorTool = new DynamicTool({
  name: "research_evaluator",
  description: `
评估研究质量和完整性，决定是否需要继续研究。
使用场景：在完成一轮信息总结后，评估当前信息是否足够生成高质量报告
输入：当前研究状态的JSON字符串
输出：评估结果和建议（continue继续研究 或 finish完成研究）
`,
  func: async (input: string) => {
    console.log('🤔 [评估工具] 正在评估研究完整性...');
    
    let stateJson = input;
    try {
      const parsed = JSON.parse(input);
      if (parsed.input) {
        stateJson = parsed.input;
      }
    } catch {
      // 如果解析失败，使用原始输入
    }
    
    const state = ResearchState.fromString(stateJson);
    
    // 防止无限循环
    if (state.iterationCount >= 3) {
      console.log('⏰ [循环限制] 已达到最大迭代次数，建议完成研究');
      return 'finish - 已达到最大研究轮次，建议生成最终报告';
    }

    const evaluationPrompt = `
作为研究质量评估专家，请评估当前研究信息的完整性和质量：

研究主题: ${state.topic}
研究轮次: ${state.iterationCount}
当前总结: ${state.summary}

评估标准：
1. 信息的全面性和深度覆盖
2. 技术层面的详细程度
3. 应用场景和案例的丰富性
4. 市场前景和商业价值分析
5. 挑战和风险识别的完整性
6. 未来发展趋势的预测

请根据以上标准判断当前信息是否足够生成一份高质量的专业研究报告。

请只回答以下两个选项之一：
- "continue" - 需要更多信息，建议继续下一轮研究
- "finish" - 信息已足够充分，可以生成高质量报告

请在回答后简要说明理由。
`;

    try {
      const response = await deepseekChat.invoke([{ role: 'user', content: evaluationPrompt }]);
      const evaluation = response.content as string;
      
      console.log(`🔍 [评估结果] ${evaluation}`);
      
      if (evaluation.toLowerCase().includes('continue')) {
        console.log('🔄 决策：继续研究 - 需要更多信息');
        return 'continue - 信息还不够充分，建议继续下一轮研究';
      } else {
        console.log('✨ 决策：完成研究 - 信息已足够');
        return 'finish - 信息已足够充分，可以生成最终报告';
      }
      
    } catch (error) {
      console.error('评估工具执行出错:', error);
      const decision = state.iterationCount === 1 ? 'continue' : 'finish';
      return `${decision} - 评估过程出错，采用默认策略`;
    }
  }
});

/**
 * 工具5：报告生成器
 * 基于所有收集的信息生成最终的专业研究报告
 */
const reportGeneratorTool = new DynamicTool({
  name: "report_generator",
  description: `
生成最终的专业研究报告。
使用场景：在研究评估认为信息足够时，生成完整的研究报告
输入：当前研究状态的JSON字符串
输出：结构化的专业研究报告
`,
  func: async (input: string) => {
    console.log('📊 [报告工具] 正在生成最终研究报告...');
    
    let stateJson = input;
    try {
      const parsed = JSON.parse(input);
      if (parsed.input) {
        stateJson = parsed.input;
      }
    } catch {
      // 如果解析失败，使用原始输入
    }
    
    const state = ResearchState.fromString(stateJson);

    const reportPrompt = `
作为资深研究分析师，请基于以下研究信息生成一份完整的专业研究报告：

研究主题: ${state.topic}
研究总结: ${state.summary}
研究轮次: ${state.iterationCount}

请生成一份结构化的研究报告，严格按照以下格式：

# ${state.topic} - 专业研究报告

## 执行摘要
[简要概述主要发现和核心观点，3-4个要点]

## 技术现状分析  
[详细分析当前技术发展水平、主要技术路线、技术成熟度]

## 市场应用前景
[分析实际应用场景、市场规模、商业价值、成功案例]

## 发展趋势预测
[基于当前信息预测未来3-5年的发展趋势和方向]

## 挑战与风险分析
[识别技术挑战、市场风险、监管限制等阻碍因素]

## 投资与发展建议
[为投资者、企业、政策制定者提供具体建议]

## 结论
[总结性结论，重申核心观点和价值]

要求：
- 报告内容专业、全面、有深度
- 逻辑清晰，结构完整
- 基于事实和数据，避免空泛表述
- 具有实用价值和指导意义
`;

    try {
      const response = await deepseekChat.invoke([{ role: 'user', content: reportPrompt }]);
      const report = response.content as string;
      
      // 更新状态
      state.report = report.trim();
      state.isComplete = true;
      currentResearchState = state;
      
      console.log('✅ [报告完成] 专业研究报告已生成');
      
      return `研究报告生成完成！\n\n${report.trim()}`;
      
    } catch (error) {
      console.error('报告生成出错:', error);
      const fallbackReport = `# ${state.topic} - 研究报告\n\n基于 ${state.iterationCount} 轮深度研究，现将主要发现总结如下：\n\n${state.summary}`;
      
      state.report = fallbackReport;
      state.isComplete = true;
      currentResearchState = state;
      
      return `研究报告生成完成（简化版）：\n${fallbackReport}`;
    }
  }
});

// ================ 辅助函数 ================

/**
 * 生成模拟搜索结果
 * 在实际应用中，这里会调用真实的搜索API
 */
function generateMockSearchResults(topic: string, plan: string): string[] {
  const baseResults = [
    `${topic}相关技术的最新发展动态和突破性进展报告`,
    `业内专家对${topic}的深度分析和未来预测研究`,
    `${topic}在实际应用中的成功案例和实施经验总结`,
    `${topic}面临的技术挑战和创新解决方案探讨`,
    `${topic}的市场前景、投资价值和商业模式分析`,
    `${topic}的标准化进展和政策法规影响评估`
  ];

  // 根据搜索计划添加更具体的结果
  const planKeywords = plan.split(',').map(k => k.trim());
  const specificResults = planKeywords.slice(0, 4).map((keyword: string) => 
    `关于"${keyword}"的最新研究成果、技术进展和应用案例分析`
  );

  return [...baseResults, ...specificResults];
}

// ================ Agent 配置和执行 ================

/**
 * 创建研究Agent的核心函数
 */
async function createResearchAgent() {
  console.log('🔧 正在配置自动化研究分析师...');
  
  // 定义Agent使用的工具集
  const tools = [
    plannerTool,
    searcherTool, 
    summarizerTool,
    evaluatorTool,
    reportGeneratorTool
  ];

  // 创建Agent提示模板
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", `你是一个专业的自动化研究分析师，能够执行完整的研究工作流程。

你的核心能力包括：
1. 智能研究规划：根据研究主题制定详细的搜索计划和关键词
2. 信息收集：执行系统性的信息搜索和数据收集
3. 深度分析：对收集的信息进行专业的分析和总结
4. 质量评估：评估研究完整性，决定是否需要继续深入
5. 报告生成：基于所有信息生成专业的研究报告

工作流程说明：
1. 首先使用 research_planner 制定研究计划
2. 然后使用 information_searcher 收集相关信息  
3. 接着使用 information_summarizer 分析总结信息
4. 使用 research_evaluator 评估是否需要继续研究
5. 如果需要继续，返回步骤1；如果足够，使用 report_generator 生成报告

重要原则：
- 严格按照工作流程执行，不要跳过任何步骤
- 每次工具调用都要传递完整的状态信息
- 根据评估结果决定是否继续循环或结束研究
- 确保最终生成高质量的专业报告
- 保持研究的逻辑性和系统性

请始终以专业、严谨的态度完成研究任务。`],
    ["human", "{input}"],
    ["placeholder", "{agent_scratchpad}"]
  ]);

  // 创建Agent
  const agent = await createOpenAIFunctionsAgent({
    llm: deepseekChat,
    tools,
    prompt,
  });

  // 创建Agent执行器
  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: true,          // 显示详细的执行过程
    maxIterations: 15,      // 增加最大迭代次数以支持循环研究
    returnIntermediateSteps: true, // 返回中间步骤
  });

  return executor;
}

/**
 * 执行完整的研究任务
 */
async function runResearch(topic: string) {
  console.log('🚀 启动自动化研究分析师');
  console.log(`📋 研究主题: ${topic}`);
  console.log('⚡ 开始执行研究工作流程...\n');

  // 初始化研究状态
  currentResearchState = new ResearchState(topic);

  try {
    // 创建研究Agent
    const researchAgent = await createResearchAgent();

    // 构建初始指令
    const instruction = `
请对主题"${topic}"进行全面深入的研究分析。

具体要求：
1. 系统性收集和分析相关信息
2. 确保研究的全面性和深度
3. 根据信息质量决定是否需要多轮研究
4. 最终生成专业的研究报告

请开始执行研究任务，首先制定研究计划。

当前研究状态：
${currentResearchState.toString()}
`;

    console.log('🎯 开始执行研究任务...\n');

    // 执行研究任务
    const result = await researchAgent.invoke({
      input: instruction
    });

    // 输出最终结果
    console.log('\n' + '='.repeat(80));
    console.log('🎉 自动化研究任务完成！');
    console.log('='.repeat(80));
    console.log(`📊 总研究轮次: ${currentResearchState.iterationCount}`);
    console.log(`📝 研究状态: ${currentResearchState.isComplete ? '已完成' : '进行中'}`);
    
    if (currentResearchState.summary) {
      console.log(`💡 核心发现: ${currentResearchState.summary.substring(0, 200)}...`);
    }
    
    console.log('\n📋 Agent执行总结:');
    console.log('-'.repeat(80));
    console.log(result.output);
    
    if (currentResearchState.report) {
      console.log('\n📄 完整研究报告:');
      console.log('-'.repeat(80));
      console.log(currentResearchState.report);
      console.log('-'.repeat(80));
    }

    return {
      finalState: currentResearchState,
      agentOutput: result.output,
      intermediateSteps: result.intermediateSteps || []
    };

  } catch (error) {
    console.error('❌ 研究过程中出现错误:', error);
    throw error;
  }
}

// ================ 主执行函数 ================

async function main() {
  console.log('🏠 欢迎使用自动化研究分析师！');
  console.log('🤖 这是一个基于LangChain Agent的智能研究工具\n');
  
  // 研究主题（可自定义）
  const researchTopic = 'AI 在医疗诊断领域的应用前景';
  
  try {
    const result = await runResearch(researchTopic);
    
    console.log('\n✅ 研究任务执行完毕');
    console.log(`📈 执行效果: ${result.finalState.isComplete ? '成功完成' : '部分完成'}`);
    
  } catch (error) {
    console.error('💥 程序执行失败:', error);
    process.exit(1);
  }
}

// 如果直接运行此文件，则执行主函数
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

// 导出主要功能
export { 
  ResearchState, 
  runResearch, 
  createResearchAgent,
  main as runResearchDemo
}; 