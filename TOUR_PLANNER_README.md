# 🌍 旅行日程规划器

一个使用 TypeScript、Zod 和 LangChain.js 构建的智能旅行规划工具，能够生成结构化的旅行计划 JSON 数据。

## 🎯 项目特点

- **结构化输出**：使用 Zod Schema 确保输出数据的类型安全
- **现代化技术栈**：采用 LangChain.js 的 `.withStructuredOutput()` 方法
- **智能规划**：基于 DeepSeek AI 模型生成详细的旅行计划
- **类型安全**：完整的 TypeScript 支持，编译时类型检查

## 🛠 技术架构

### 核心技术组合

1. **Zod + .withStructuredOutput()** 的优势：
   - 🔒 **编译时类型安全**：从 Schema 自动生成 TypeScript 类型
   - ✅ **运行时验证**：验证 AI 返回的数据结构
   - 🎯 **强制结构化**：在模型层面确保输出格式一致性
   - 📋 **详细错误信息**：Schema 验证失败时的清晰错误描述

2. **相比传统方法的优势**：
   - 比 prompt engineering 更可靠
   - 比 JsonOutputParser 更安全
   - 利用最新 LLM 的 function calling 能力

## 📦 安装与使用

### 1. 安装依赖
```bash
npm install zod
```

### 2. 设置环境变量
确保你的 `.env` 文件包含 DeepSeek API 密钥：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 运行旅行规划器
```bash
npm run start_tour
```

## 📊 输出格式

旅行规划器生成的 JSON 结构如下：

```typescript
{
  destination: string;           // 旅行目的地
  duration_days: number;         // 总旅行天数
  daily_plans: Array<{          // 每日详细计划
    day: number;                // 第几天
    theme: string;              // 当天主题
    activities: string[];       // 具体活动列表
  }>;
  estimated_cost_usd: number;   // 预估费用（美元）
}
```

## 🧩 代码结构

```typescript
// 1. 定义 Zod Schema
const itinerarySchema = z.object({
  destination: z.string().describe("旅行目的地"),
  duration_days: z.number().describe("总旅行天数"),
  // ... 其他字段
});

// 2. 创建结构化模型
const structuredModel = deepseekChat.withStructuredOutput(itinerarySchema);

// 3. 构建处理链
const chain = prompt.pipe(structuredModel);

// 4. 调用并获取结构化结果
const result: TravelItinerary = await chain.invoke({ request });
```

## 🔧 自定义使用

你可以轻松自定义旅行规划器：

```typescript
import { createTravelPlannerChain } from './src/tour_planner.js';

const planner = createTravelPlannerChain();
const result = await planner.invoke({
  request: "为期5天的北海道之旅，重点是滑雪和温泉"
});

console.log(result.destination);      // 类型安全访问
console.log(result.daily_plans);      // 自动补全支持
```

## 📝 示例输出

```json
{
  "destination": "京都",
  "duration_days": 3,
  "daily_plans": [
    {
      "day": 1,
      "theme": "经典寺庙巡礼",
      "activities": [
        "清水寺参观",
        "二年坂・三年坂漫步",
        "祇园角观看传统表演"
      ]
    }
  ],
  "estimated_cost_usd": 800
}
```

## 🎨 核心优势

1. **可靠性**：结构化输出确保数据格式一致
2. **类型安全**：完整的 TypeScript 支持
3. **现代化**：使用最新的 LangChain.js 技术
4. **易扩展**：清晰的模块化设计

---

*Built with ❤️ using TypeScript, Zod, and LangChain.js* 