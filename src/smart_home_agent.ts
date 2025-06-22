/*
 * 智能家居助手 Agent 实现
 * 
 * 所需依赖包（在 package.json 中已配置）：
 * - langchain: 核心 LangChain 库
 * - @langchain/ollama: Ollama 模型支持
 * - @langchain/openai: OpenAI 模型支持（备用）
 * - @langchain/community: 社区工具包
 * 
 * 本脚本演示如何创建一个能理解并执行复合指令的 Agent：
 * 1. 定义模拟的智能家居控制工具
 * 2. 创建基于工具的 Agent
 * 3. 使用 AgentExecutor 运行 ReAct（推理-行动）循环
 */

import { newOllamaChat } from './model_helper.js';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents';
import { DynamicTool } from '@langchain/community/tools/dynamic';
import { StructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

// ================== 工具定义部分 ==================

/**
 * 智能家居工具 1：获取天气信息
 * 这是一个模拟的天气查询工具，实际项目中可以调用真实的天气 API
 */
const getWeatherTool = new DynamicTool({
  name: "get_weather",
  description: "Get current weather information for a specific city. Use this when user asks about weather conditions.",
  func: async (city: string) => {
    console.log(`🌤️  正在获取 ${city} 的天气信息...`);
    // 模拟 API 调用延迟
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // 返回模拟的天气数据
    const mockWeather = `${city} 当前天气：晴天，温度 22°C，湿度 65%，微风`;
    console.log(`📡 天气查询完成: ${mockWeather}`);
    return mockWeather;
  }
});

/**
 * 智能家居工具 2：控制灯光状态
 * 这是一个模拟的智能灯光控制工具，实际项目中可以连接到 IoT 设备
 */
class LightControlTool extends StructuredTool {
  name = "set_light_status";
  description = "Control smart lights in different rooms. Use this to turn lights on or off in specific rooms.";
  
  schema = z.object({
    room: z.string().describe("The room where the light is located (e.g., living room, bedroom, kitchen)"),
    status: z.enum(["on", "off"]).describe("Whether to turn the light on or off")
  });

  async _call(args: { room: string; status: "on" | "off" }): Promise<string> {
    const { room, status } = args;
    console.log(`💡 正在${status === 'on' ? '打开' : '关闭'} ${room} 的灯光...`);
    
    // 模拟设备控制延迟
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const result = `${room} 的灯光已${status === 'on' ? '打开' : '关闭'}`;
    console.log(`🏠 灯光控制完成: ${result}`);
    return result;
  }
}

/**
 * 智能家居工具 3：查询设备状态
 * 这是一个模拟的设备状态查询工具
 */
const checkDeviceStatusTool = new DynamicTool({
  name: "check_device_status",
  description: "Check the current status of smart home devices like lights, air conditioner, etc. Use this to get current device states.",
  func: async (device: string) => {
    console.log(`🔍 正在查询 ${device} 的状态...`);
    await new Promise(resolve => setTimeout(resolve, 600));
    
    // 模拟随机的设备状态
    const isOn = Math.random() > 0.5;
    const status = `${device} 当前状态: ${isOn ? '开启' : '关闭'}`;
    console.log(`📊 状态查询完成: ${status}`);
    return status;
  }
});

// ================== Agent 配置部分 ==================

/**
 * 创建并配置 LLM 模型
 * 这里使用 Ollama 的本地模型，确保你已经安装了 Ollama 并下载了相应模型
 */
const model = newOllamaChat("qwen:7b", 0.1);

/**
 * 定义 Agent 使用的工具列表
 * 这些工具将提供给 Agent，Agent 会根据用户输入决定使用哪些工具
 */
const tools = [
  getWeatherTool,
  new LightControlTool(),
  checkDeviceStatusTool
];

/**
 * 创建 Chat Prompt Template
 * 这个模板定义了 Agent 的行为模式和上下文
 * 
 * 重要占位符说明：
 * - {input}: 用户的输入指令
 * - {agent_scratchpad}: Agent 的"草稿纸"，用于存储推理过程和工具调用历史
 */
const prompt = ChatPromptTemplate.fromMessages([
  ["system", `你是一个智能家居助手，可以帮助用户控制家居设备和获取相关信息。

你的能力包括：
1. 控制各个房间的灯光开关
2. 获取城市天气信息  
3. 查询智能设备状态

请按照以下原则工作：
- 理解用户的复合指令，将其分解为多个步骤
- 按逻辑顺序执行每个步骤
- 为每个操作提供清晰的反馈
- 如果不确定某个参数，请询问用户

使用可用的工具来完成任务，并提供友好的中文回复。`],
  ["human", "{input}"],
  ["placeholder", "{agent_scratchpad}"]
]);

// ================== Agent 创建和执行部分 ==================

/**
 * ReAct（Reasoning and Acting）循环说明：
 * 
 * ReAct 是一种重要的 Agent 工作模式，它将推理（Reasoning）和行动（Acting）结合：
 * 
 * 1. **Reasoning（推理）**: Agent 分析用户输入，理解需要完成什么任务
 * 2. **Acting（行动）**: Agent 选择并调用适当的工具来执行任务
 * 3. **Observation（观察）**: Agent 观察工具执行的结果
 * 4. **继续循环**: 基于观察结果，Agent 决定是否需要进一步的推理和行动
 * 
 * 这个循环持续进行，直到 Agent 认为任务已完成或无法继续。
 * 
 * AgentExecutor 的作用：
 * - 管理整个 ReAct 循环的执行流程
 * - 处理工具调用和结果返回
 * - 提供错误处理和安全限制（如最大迭代次数）
 * - 维护对话上下文和工具使用历史
 * 
 * 工具 description 的重要性：
 * - LLM 主要依靠工具的 description 来理解何时使用该工具
 * - 清晰、准确的 description 直接影响 Agent 的决策质量
 * - description 应该包含工具的功能、使用场景和参数说明
 */

async function runSmartHomeAgent() {
  try {
    console.log("🏡 智能家居助手启动中...\n");
    
    // 创建 OpenAI Functions Agent
    // 注意：虽然叫 OpenAI Functions Agent，但它也可以与其他支持函数调用的模型一起工作
    const agent = await createOpenAIFunctionsAgent({
      llm: model,
      tools,
      prompt,
    });

    // 创建 AgentExecutor
    // AgentExecutor 负责管理 Agent 的执行流程和工具调用
    const executor = new AgentExecutor({
      agent,
      tools,
      verbose: true, // 启用详细日志，可以看到完整的推理过程
      maxIterations: 10, // 最大迭代次数，防止无限循环
    });

    // 测试场景 1：复合指令 - 查询设备状态并控制
    console.log("📝 测试场景 1：复合设备控制指令");
    console.log("=" .repeat(50));
    
    const complexCommand1 = "请检查卧室灯的状态，如果是关闭的就打开它，然后告诉我东京的天气";
    
    console.log(`👤 用户指令: ${complexCommand1}\n`);
    
    const result1 = await executor.invoke({
      input: complexCommand1
    });
    
    console.log(`\n🤖 最终回复: ${result1.output}\n`);
    console.log("=" .repeat(50));

    // 测试场景 2：简单的天气查询
    console.log("\n📝 测试场景 2：简单天气查询");
    console.log("=" .repeat(50));
    
    const simpleCommand = "北京现在的天气怎么样？";
    console.log(`👤 用户指令: ${simpleCommand}\n`);
    
    const result2 = await executor.invoke({
      input: simpleCommand
    });
    
    console.log(`\n🤖 最终回复: ${result2.output}\n`);
    console.log("=" .repeat(50));

    // 测试场景 3：多房间灯光控制
    console.log("\n📝 测试场景 3：多房间灯光控制");
    console.log("=" .repeat(50));
    
    const lightCommand = "把客厅和厨房的灯都打开";
    console.log(`👤 用户指令: ${lightCommand}\n`);
    
    const result3 = await executor.invoke({
      input: lightCommand
    });
    
    console.log(`\n🤖 最终回复: ${result3.output}\n`);
    
  } catch (error) {
    console.error("❌ Agent 执行过程中出现错误:", error);
  }
}

// ================== 程序入口 ==================

/**
 * 主函数：启动智能家居助手
 * 
 * 使用方法：
 * npm run build && node -r dotenv/config dist/src/smart_home_agent.js
 * 或者直接使用 ts-node：
 * npx ts-node -r dotenv/config src/smart_home_agent.ts
 */
if (import.meta.url === new URL(import.meta.resolve(import.meta.url)).href) {
  runSmartHomeAgent().catch(console.error);
}

export { runSmartHomeAgent }; 