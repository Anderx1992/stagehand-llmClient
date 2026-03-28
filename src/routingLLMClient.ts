/**
 * RoutingLLMClient — 根据请求内容自动路由到不同模型：
 *   - 包含图片的请求 → Gemini（视觉模型），支持 SOCKS5 代理
 *   - 纯文本请求 → MiniMax（文本模型）
 *
 * 内置了两个 AISdkClient 的创建逻辑，使用时只需传入 API Key 等配置即可。
 * @see https://docs.stagehand.dev/v3/configuration/models#all-other-providers
 */

import {
  LLMClient,
  AISdkClient,
  type CreateChatCompletionOptions,
  type ChatCompletionOptions,
  type LLMResponse,
} from "@browserbasehq/stagehand";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { Socks5ProxyAgent, fetch as undiciFetch } from "undici";

export interface RoutingClientConfig {
  /** Gemini API Key，默认读取环境变量 GOOGLE_GENERATIVE_AI_API_KEY */
  geminiApiKey?: string;
  /** Gemini 模型 ID，默认 "gemini-2.5-flash" */
  geminiModelId?: string;

  /** MiniMax API Key，默认读取环境变量 MINIMAX_API_KEY */
  minimaxApiKey?: string;
  /** MiniMax 模型 ID，默认 "MiniMax-M1" */
  minimaxModelId?: string;
  /** MiniMax API 基础 URL，默认 "https://api.minimaxi.com/v1" */
  minimaxBaseURL?: string;

  /**
   * SOCKS5 代理地址，仅用于 Gemini 调用。
   * 格式：socks5://[user:pass@]host:port
   * 若未设置，回退读取环境变量 GEMINI_SOCKS5_PROXY
   */
  socks5ProxyUrl?: string;
}

const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";
const DEFAULT_MINIMAX_MODEL = "MiniMax-M1";
const DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1";

function createSocks5ProxiedFetch(proxyUrl: string) {
  const dispatcher = new Socks5ProxyAgent(proxyUrl);
  return (input: string | URL | Request, init?: RequestInit) =>
    undiciFetch(
      input as Parameters<typeof undiciFetch>[0],
      { ...(init as any), dispatcher },
    ) as unknown as Promise<Response>;
}

function buildVisionClient(config: RoutingClientConfig): AISdkClient {
  const google = createGoogleGenerativeAI({
    apiKey: config.geminiApiKey ?? process.env.GOOGLE_GENERATIVE_AI_API_KEY,
  });
  const modelId = config.geminiModelId ?? DEFAULT_GEMINI_MODEL;
  return new AISdkClient({ model: google(modelId) });
}

function buildTextClient(config: RoutingClientConfig): AISdkClient {
  const minimax = createOpenAICompatible({
    name: "minimax",
    baseURL: config.minimaxBaseURL ?? DEFAULT_MINIMAX_BASE_URL,
    apiKey: config.minimaxApiKey ?? process.env.MINIMAX_API_KEY,
  });
  const modelId = config.minimaxModelId ?? DEFAULT_MINIMAX_MODEL;
  return new AISdkClient({ model: minimax(modelId) });
}

export class RoutingLLMClient extends LLMClient {
  public type = "routing" as const;
  public hasVision = true;

  private visionClient: AISdkClient;
  private textClient: AISdkClient;
  private proxiedFetch?: ReturnType<typeof createSocks5ProxiedFetch>;

  constructor(config: RoutingClientConfig = {}) {
    const textModelId = config.minimaxModelId ?? DEFAULT_MINIMAX_MODEL;
    super(textModelId);

    this.visionClient = buildVisionClient(config);
    this.textClient = buildTextClient(config);

    const proxyUrl = config.socks5ProxyUrl || process.env.GEMINI_SOCKS5_PROXY;
    if (proxyUrl) {
      this.proxiedFetch = createSocks5ProxiedFetch(proxyUrl);
    }
  }

  private hasImageContent(options: ChatCompletionOptions): boolean {
    if (options.image) {
      return true;
    }
    for (const message of options.messages) {
      if (!Array.isArray(message.content)) continue;
      for (const part of message.content) {
        if (
          ("image_url" in part && part.image_url) ||
          ("source" in part && part.source)
        ) {
          return true;
        }
      }
    }
    return false;
  }

  async createChatCompletion<T = LLMResponse>(
    params: CreateChatCompletionOptions,
  ): Promise<T> {
    const useVision = this.hasImageContent(params.options);
    const delegate = useVision ? this.visionClient : this.textClient;

    params.logger({
      category: "routing-llm",
      message: `routing to ${useVision ? "vision (gemini)" : "text (minimax)"} model: ${delegate.modelName}`,
      level: 1,
      auxiliary: {
        modelName: { value: delegate.modelName, type: "string" },
        hasImage: { value: String(useVision), type: "string" },
      },
    });

    if (useVision && this.proxiedFetch) {
      params.logger({
        category: "routing-llm",
        message: "using SOCKS5 proxy for Gemini call",
        level: 1,
      });
      const originalFetch = globalThis.fetch;
      globalThis.fetch = this.proxiedFetch as typeof globalThis.fetch;
      try {
        return await delegate.createChatCompletion<T>(params);
      } finally {
        globalThis.fetch = originalFetch;
      }
    }

    return delegate.createChatCompletion<T>(params);
  }
}
