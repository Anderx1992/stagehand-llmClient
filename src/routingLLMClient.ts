/**
 * RoutingLLMClient — 根据请求内容自动路由到不同模型：
 *   - 包含图片的请求 → Gemini 3.0（视觉模型），支持 SOCKS5 代理
 *   - 纯文本请求 → MiniMax M2.7
 *
 * 基于 Stagehand 官方文档推荐的 AISdkClient + Vercel AI SDK provider 方式实现。
 * @see https://docs.stagehand.dev/v3/configuration/models#all-other-providers
 */

import {
  LLMClient,
  AISdkClient,
  type CreateChatCompletionOptions,
  type ChatCompletionOptions,
  type LLMResponse,
} from "@browserbasehq/stagehand";
import { Socks5ProxyAgent, fetch as undiciFetch } from "undici";

export interface RoutingClientConfig {
  /** Gemini 视觉模型的 AISdkClient 实例 */
  visionClient: AISdkClient;
  /** MiniMax 文本模型的 AISdkClient 实例 */
  textClient: AISdkClient;
  /**
   * SOCKS5 代理地址，仅用于 Gemini 调用。
   * 格式：socks5://[user:pass@]host:port
   * 若未设置，回退读取环境变量 GEMINI_SOCKS5_PROXY
   */
  socks5ProxyUrl?: string;
}

function createSocks5ProxiedFetch(proxyUrl: string) {
  const dispatcher = new Socks5ProxyAgent(proxyUrl);
  return (input: string | URL | Request, init?: RequestInit) =>
    undiciFetch(
      input as Parameters<typeof undiciFetch>[0],
      { ...(init as any), dispatcher },
    ) as unknown as Promise<Response>;
}

export class RoutingLLMClient extends LLMClient {
  public type = "routing" as const;
  public hasVision = true;

  private visionClient: AISdkClient;
  private textClient: AISdkClient;
  private proxiedFetch?: ReturnType<typeof createSocks5ProxiedFetch>;

  constructor(config: RoutingClientConfig) {
    super(config.textClient.modelName);
    this.visionClient = config.visionClient;
    this.textClient = config.textClient;

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
