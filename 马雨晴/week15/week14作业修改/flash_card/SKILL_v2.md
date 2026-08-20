---
name: flash-card
description: >-
  为单个英语单词生成静态 HTML 学习闪卡。闪卡包含标准化单词、音标、词性、中文释义、
  恰好 3 条中英对照例句和 4-6 个近义词。Use when the user asks to create/make/generate
  an English-word flash card / 单词卡 / 闪卡, including Chinese or English requests such as
  "给我做 resilient 的闪卡", "make a flash card for meticulous", "生成 crazy 单词卡"。
---

# Flash Card 单词闪卡生成

为**一个英语单词**生成可直接预览的静态 HTML 学习闪卡。

## 适用范围

### 应触发
当请求明确要求为一个英语单词制作闪卡、单词卡或 flash card 时触发，例如：
- `给我做 resilient 的闪卡`
- `帮我生成 meticulous 的单词卡`
- `make a flash card for crazy`
- `做一张 AMBITIOUS 的英语单词卡`

### 不应触发
以下请求不属于本 Skill：
- 只问单词含义但没有要求制作闪卡；
- 要求生成多个单词的整套词表；
- 非英语单词；
- 句子、短语、语法题。

如果用户一次给出多个候选单词且没有明确目标，不要猜测，应要求用户指定一个单词。

## 输入解析

1. 从用户请求中提取**唯一目标英语单词**。
2. 将目标单词规范化为小写，并用该值作为 JSON 和 HTML 文件名。
3. 不要把 `flash card`、`单词卡`、`闪卡` 等指令词误识别为目标单词。
4. 如果无法可靠提取唯一英语单词，停止执行并说明需要用户提供一个明确的英语单词。

## 数据生成要求

生成以下 JSON 对象，字段名和类型必须完全一致：

```json
{
  "word": "resilient",
  "phonetic": "/rɪˈzɪliənt/",
  "pos": "adj.",
  "definition": "能迅速从困难、挫折中恢复过来的；有韧性的，适应力强的",
  "examples": [
    {
      "en": "She is a resilient child who bounces back quickly from setbacks.",
      "zh": "她是个有韧性的孩子，遇到挫折能很快恢复过来。"
    },
    {
      "en": "The economy proved remarkably resilient during the crisis.",
      "zh": "在危机期间，经济表现出了惊人的韧性。"
    },
    {
      "en": "A resilient mindset helps you cope with life's challenges.",
      "zh": "有韧性的心态能帮助你应对生活中的挑战。"
    }
  ],
  "synonyms": ["tough", "flexible", "strong", "hardy", "buoyant"]
}
```

### 字段约束

- `word`
  - 必须等于从请求中提取的目标单词；
  - 必须小写；
  - 不得添加解释、括号或其他文字。

- `phonetic`
  - 使用常见 IPA 音标；
  - 使用 `/.../` 包裹；
  - 不确定时仍应给出最常见的标准读音，不要留空。

- `pos`
  - 使用简洁英文缩写，如 `n.`、`v.`、`adj.`、`adv.`；
  - 如果单词有多种词性，选择与 `definition` 和例句对应的**一个核心词性**。

- `definition`
  - 使用简洁、自然的中文；
  - 与所选词性和例句中的词义保持一致；
  - 不堆砌无关义项。

- `examples`
  - 必须**恰好 3 条**；
  - 每条必须同时包含非空 `en` 和 `zh`；
  - 英文例句应自然、长度适中，并体现目标单词的典型搭配或语境；
  - 中文翻译必须与英文例句语义一致；
  - 三条例句尽量覆盖不同语境，避免仅替换主语形成近重复；
  - 英文例句中应实际出现目标单词，允许必要的大小写变化。

- `synonyms`
  - 必须为 **4-6 个**英文近义词；
  - 不得重复；
  - 尽量与 `definition` 指定的核心词义接近；
  - 不使用短语解释代替单词列表。

## 生成前自检

在写文件前检查：
1. `word` 是否与用户目标单词一致；
2. 所有必需字段是否存在且非空；
3. `examples` 是否恰好 3 条；
4. 每条例句是否同时有英文和中文；
5. `synonyms` 是否为 4-6 个且无重复；
6. JSON 是否可被标准 JSON 解析器直接解析。

如检查失败，先修正数据再进入下一步，不要把不合格数据交给生成脚本。

## 文件与脚本执行

设 `<skill_dir>` 为当前 `SKILL.md` 所在目录。

1. 将 JSON 保存为：

```text
<skill_dir>/data/<word>.json
```

2. 调用生成脚本：

```bash
python <skill_dir>/script/make_flashcard.py <skill_dir>/data/<word>.json
```

默认 HTML 输出到**当前工作目录**，文件名：

```text
./<word>.html
```

如脚本支持 `-o`，可以显式指定输出路径。

3. 检查脚本是否成功结束，并确认 HTML 文件存在且非空。
4. 使用默认浏览器打开生成的 HTML 文件进行预览。

## 失败处理

- JSON 无法生成或不符合约束：修正后再写入；
- `script/make_flashcard.py` 不存在：明确报告脚本缺失，不要伪造“生成成功”；
- 脚本执行失败：返回错误信息，不要声称 HTML 已生成；
- HTML 文件不存在或为空：视为生成失败；
- 不要静默吞掉异常。

## 输出原则

成功时只需简洁确认：
- 目标单词；
- JSON 保存位置；
- HTML 输出位置；
- 是否已打开预览。

不要输出冗长的中间推理过程。
