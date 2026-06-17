# 语音电话呼叫系统 — GitHub 开源项目调研报告

> 调研日期：2026-06-11

---

## 一、需求概述

| 需求 | 说明 |
|---|---|
| 模拟电话通话 | 流式 TTS + ASR，模拟真实通话效果 |
| 打断 (barge-in) | 用户可随时插话中断 AI 语音输出 |
| 外接 API | 可接入自己的 TTS / ASR / LLM API |
| 流式波形 | 音频波形实时可视化 |
| 多语言 | 中文、英文、粤语 |
| 技术栈 | Python 优先 |

---

## 二、候选项目对比

### 1. Pipecat（⭐12.8k）— 生态最大

| 项目 | |
|---|---|
| GitHub | https://github.com/pipecat-ai/pipecat |
| Language | Python (98%) |
| License | BSD-2 |
| TTS/ASR | 30+ 集成 (Deepgram, Whisper, Azure, ElevenLabs, Cartesia, OpenAI 等) |
| LLM | 25+ 集成 (OpenAI, Anthropic, Gemini, 本地模型等) |
| 打断 | ✅ Silero VAD + 可中断 Pipeline, 支持 Krisp/Koala 降噪 |
| 多语言 | 取决于所选供应商 (Deepgram 30+ 语种) |
| 波形可视化 | 无内置, 依赖客户端 SDK (React/Swift/Kotlin) |
| 电话 | WebRTC + Daily + LiveKit, 可接 Twilio |
| 优劣 | ✅ 集成最多、社区最大、插件最丰富<br>❌ Pipeline 架构有学习曲线；依赖面大 |

### 结构非常完整，已经集成语音识别，文本转语音和对话处理。模块化，支持流式处理。支持sdk开发。

### 2. LiveKit Agents（⭐10.9k）— 最好用的 WebRTC 方案

| 项目 | |
|---|---|
| GitHub | https://github.com/livekit/agents |
| Language | Python (99%) |
| License | Apache 2.0 |
| TTS/ASR | 15+ 插件 (Deepgram, OpenAI, Cartesia, ElevenLabs, Azure, Google, Groq 等) |
| LLM | OpenAI, Anthropic, Gemini 等 |
| 打断 | ✅ **语义级 turn detection** + Silero VAD, 业界最强 |
| 多语言 | Deepgram `language="multi"`；ElevenLabs 多语言 TTS |
| 波形可视化 | ✅ LiveKit 客户端 SDK 自带 UI 组件 (React/Swift/Android/Flutter) |
| 电话 | ✅ **SIP + Twilio 外呼/呼入**, 不依赖付费 SaaS, 可自建 LiveKit Server |
| 优劣 | ✅ 中断检测最强、WebRTC 最成熟、文档优秀、客户端最全<br>❌ 需部署 LiveKit Server (自建免费) |
### 结构非常完整。可集成，模块化，支持流式处理。支持sdk开发。

### 3. Patter（⭐515）— 最快的电话接入

| 项目 | |
|---|---|
| GitHub | https://github.com/PatterAI/Patter |
| Language | Python (52%) + TypeScript (42%) |
| License | MIT |
| TTS/ASR | 27+ 集成 (OpenAI Realtime, Gemini Live, ElevenLabs ConvAI, Deepgram 等) |
| LLM | OpenAI, Anthropic, Gemini, Groq, Cerebras |
| 打断 | ✅ Silero VAD + Krisp + DeepFilterNet |
| 多语言 | 取决于供应商 |
| 波形可视化 | 无内置；Dashboard 有延迟/成本追踪 |
| 电话 | ✅ **Twilio + Telnyx + Plivo**, 4 行代码绑定号码, 支持外呼 + 语音信箱检测 |
| 优劣 | ✅ 明确对标 Vapi/Retell、接入最快、MIT 协议<br>❌ 社区小 (515 stars)、文档不完善、commit 少 (125) |
### 可AI 代理连接到真实的电话通话，支持sdk开发，集成。
### 4. Deepgram Voice Agent SDK（⭐441）— 一站式 API

| 项目 | |
|---|---|
| GitHub | https://github.com/deepgram/deepgram-python-sdk |
| Language | Python (100%) |
| License | MIT |
| TTS/ASR | Deepgram 自有 (Nova-3 STT + Aura-2 TTS, 一体式 Voice Agent 模式) |
| 打断 | ✅ Contextual turn detection (Listen v2) |
| 多语言 | ✅ **Nova-3 支持 30+ 语种, 含粤语 (yue) 和普通话 (zh)** |
| 波形可视化 | 无 |
| 电话 | ❌ 纯 SDK, 不内置电话功能 |
| 优劣 | ✅ 粤语/中文最成熟、集成最简单 (listen+think+speak 一条 API)<br>❌ 锁定 Deepgram 生态；LLM 选择有限；API 收费 |
### deepgram的子项目，不推荐。它将语音转文本、文本转语音和 LLM 编排统一到一个 API 中，从而降低了复杂性、延迟和成本。不适合我们的项目（非专门为电话开发的）
### 5. Bolna（⭐665）— 端到端电话平台

| 项目 | |
|---|---|
| GitHub | https://github.com/bolna-ai/bolna |
| Language | Python (99.7%) |
| License | MIT |
| TTS/ASR | Deepgram, Azure (STT)；AWS Polly, ElevenLabs, OpenAI, Cartesia, XTTS (TTS) |
| LLM | LiteLLM 统一接口 (OpenAI, DeepSeek, Llama, Mistral, Groq, Anthropic 等) |
| 打断 | ✅ WebSocket 双向流 + VAD |
| 多语言 | Deepgram 多语种 |
| 波形可视化 | 无 |
| 电话 | ✅ Twilio, Plivo |
| 优劣 | ✅ 端到端平台、Docker 部署、LiteLLM 统一接口<br>❌ 部分组件闭源；社区较小 |
### 部署要求多Telephony web server,Bolna server,ngrok, redis
### 6. 其他参考项目

| 项目 | Stars | 语言 | 要点 |
|---|---|---|---|
| Vocode | 3.8k | Python | ⚠️ 2024年6月停更, 寻求社区维护者 |
| Siphon | 41 | Python | 基于 LiveKit, AGPL 协议, 企业级扩展,可能需要自己搞定sdk开发|
| Enterprise Realtime Voice Agent | 36 | Python | 教程教你如何构建一个实时语音代理https://github.com/SalesforceAIResearch/enterprise-realtime-voice-agent |
| LangGraph Voice Call Agent | 43 | Python | LangGraph + LiveKit, 状态图流程化对话，没有维护的项目不推荐（10个月前） |
| SIP-to-AI | 59 | Python | SIP/Freeswitch/Asterisk → AI 语音代理, 低延迟，多实时 API，纯Python SIP + RTP（无C 依赖），不是很推荐。这个是给多模态LLM用的 |


---

## 三、最终推荐

| 场景 | 推荐 | 理由 |
|---|---|---|
| **快速原型 + 电话模拟** | **LiveKit Agents** | 中断最强、WebRTC 免费自建、客户端 SDK 带波形、SIP 外呼 |
| **最大灵活性 + API 集成** | **Pipecat** | 生态最大、30+ 供应商可任意切换 |
| **最快接入短信拨号** | **Patter** | 4 行代码绑号码、MIT 协议 |

> **综合推荐：LiveKit Agents**
>
> 自建 LiveKit Server 免费，中断业界最强（语义+ VAD），客户端 SDK 自带流式波形，可外接任何 TTS/ASR/LLM API，SIP 支持模拟通话。

### 可以考虑Pipecat，LiveKit Agents，Patter

### 取舍点：
- 要手机 App 直连通话（不走电话线路）→ LiveKit Agents 是唯一选择
- 要最大供应商灵活度和社区 → Pipecat
- 要最快原型验证、走电话 → Patter

