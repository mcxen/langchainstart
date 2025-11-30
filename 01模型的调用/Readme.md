参考：

Entrypoint to using [chat models](https://docs.langchain.com/oss/python/langchain/models) in LangChain.



### 模型分类





| 类型                 | 说明                 | 例子                               |
| :------------------- | :------------------- | :--------------------------------- |
| **Chat Model**       | 基于消息格式的模型   | ChatOpenAI, ChatAnthropic          |
| **Text LLM**         | 旧式文本输入输出模型 | OpenAI (legacy)                    |
| **Embedding Model**  | 文本向量化模型       | OpenAIEmbeddings                   |
| **Local Model**      | 本地运行的模型       | ChatOllama, ChatHuggingFace        |
| **Cloud Model**      | 云端模型             | ChatOpenAI, ChatGoogleGenerativeAI |
| **Multimodal Model** | 支持多模态输入       | ChatOpenAI, ChatAnthropic          |

#### 参数设置分类

三种参数设置方式对比

```

方式一：初始化参数（推荐大多数场景）
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)
result = model.invoke("问题")
# ✅ 简单清晰
# ✅ 性能最好
# ❌ 不灵活

方式二：运行时参数（推荐需要变化的场景）
model = ChatOpenAI(model="gpt-4o")
result1 = model.invoke("问题1", config={"temperature": 0.5})
result2 = model.invoke("问题2", config={"temperature": 0.9})
# ✅ 灵活
# ⚠️ 每次调用要写参数
# ❌ 代码重复

方式三：中间件参数（推荐复杂逻辑的场景）
middleware = createMiddleware({
    wrapModelCall: (request, handler) => {
        # 根据复杂逻辑改参数
        request.model_kwargs = calculate_params(request.state)
        return handler(request)
    }
})
agent = create_agent(model="gpt-4o", middleware=[middleware])
# ✅ 最灵活
# ✅ 逻辑集中
# ❌ 学习曲线陡
```



## Init_chat_model

你的原始代码是直接使用 `ChatOpenAI`：

```python
# ❌ 你现在的方式
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="google/gemma-3n-e4b",
    openai_api_key="lm-studio",
    openai_api_base="http://localhost:1234/v1"
)
```

改进方案 1：用 `init_chat_model`（简单）

```python
from langchain.chat_models import init_chat_model
import os

# 设置环境变量（可选，init_chat_model 会用到）
os.environ["OPENAI_API_KEY"] = "lm-studio"

# 使用 init_chat_model
llm = init_chat_model(
    "openai:google/gemma-3n-e4b",  # 模型 ID
    api_key="lm-studio",
    base_url="http://localhost:1234/v1"
)
```