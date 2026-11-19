# ComfyUI 二次元实验室

> 在 6 GB 显存的笔记本上，把二次元出图这件事从「点按钮」做成「能复用的管线」。
> 记录了什么有效、什么无效，以及每一个坑是怎么查出来的。

**English TL;DR** — A field-tested ComfyUI pipeline for anime image generation on a
6 GB laptop GPU. Not a model, not a framework: a set of small scripts and workflow
presets that automate the boring parts (reference fetching, pose extraction,
matting, compositing) and document the failure modes. No image assets included.

---

## 这是什么

一套在**消费级小显存**上跑通的二次元出图环境，包含：

- **8 个 ComfyUI 工作流**（SDXL 立绘/动画、3DCG 手办质感、FLUX 单帧）
- **6 个画风档**（同一批底模派生出的参数配置，切画风不搬文件）
- **40 个 Python 脚本**，核心是 8 个：一条命令出图、开关面板、扒参考图、多图参考、造骨架、抠图合成、中文转 tag、颜文字转表情
- **一份 7 万字的技术日志**，记录所有故障的排查过程

## 这不是什么

先说清楚，免得你 clone 下来发现跑不起来：

- **不是一个能一键跑的项目。** 模型要自己下（清单见 `hq-downloads.txt` 等文件），
  脚本里有本机绝对路径（`BASE` 和 `COMFY` 两个常量），换机器要改。
- **不是模型也不是框架。** 没有训练代码、没有权重、没有推理服务。
- **不含任何图片素材。** `refs/` 和 `output/` 都不入库，理由见文末。

## 核心思路：两段式，边界在文件

这套东西最关键的设计不是某个参数，而是**分层**：

```
脚本层（本机 Python）               ComfyUI（节点图）
  联网 / 筛选 / 提骨架    →  文件  →  只读文件，纯确定性计算
  可能失败、每次不同                    同输入必然同输出
```

分界线画在「不确定」和「确定」之间。扒图会超时、会 503、会筛出垃圾图，
这些脏活全在脚本层做完，落盘成确定的结果；ComfyUI 那边永远只读已经存在的文件。

好处是：**参考图扒一次就停在盘上**，后面换底模、换尺寸、换场景都不用再联网，
同一个 seed 永远出同一张图，调参才调得下去。

## 能做什么

| 能力 | 脚本 | 说明 |
|---|---|---|
| 一条命令出图 | `make.py` | 骨架 → 生成 → 4x 超分 → 换背景，实测 161 秒 |
| 开关面板 | `switch.py` | 11 项，翻开关直接出图，不用记参数 |
| 按 tag 扒参考图 | `ref_search.py` | 多图源 + 分级/垃圾图过滤 + 自动提骨架 |
| 多图参考 | `multi_ref.py` | 角色/服装/质感走 IP-Adapter，动作走 ControlNet |
| 造骨架 | `skeleton_gen.py` | 18 点 OpenPose 骨架 + 手部线稿，姿势可手捏 |
| 抠图合成 | `matte_composite.py` | 软 alpha + 去色溢 + 接触阴影 + 色温匹配 |
| 中文转 tag | `nl2tags.py` | 174 条词典，零依赖，可单独用 |
| 颜文字转表情 | `kaomoji.py` | 42 条映射，`(╥﹏╥)` → crying, streaming tears |

## 几个实测结论（省你几天）

这些都是试出来的，不是看文档抄的：

**关于模型**
- `NOVA 3DCG` 是 diffusers 目录格式，必须用 `DiffusersLoader`，`CheckpointLoaderSimple` 会失败
- AnimateDiff v3 在 **16 帧以下必崩**，表现为噪点或霓虹条纹，**而且不报错**
- 6 GB 训不了 SDXL LoRA（至少 12 GB）；只能训 SD1.5 或 Textual Inversion

**关于参考图**
- IP-Adapter 拿到**非正方形**参考图时，CLIP 会中心裁切，把脸裁掉——这是"喂了图但角色不像"的头号原因
- `ipa_end` 收到 0.65 会在人物周围留一圈贴纸描边，要到 0.45
- 白底抠图的参考会导致生成图带描边

**关于姿态**
- 骨架不跟，**先怀疑提示词在打架**，不要急着加权重。`standing`、`looking at viewer`、
  `plain wall background` 这三个词会让模型把"举臂抬腿"解释成"扶墙摆拍"
- 骨架图必须是**黑底 + 标准 18 点**。画在照片上（133 点全画）ControlNet 读不懂

**关于扒图**
- 图站的 tag 是 AND 逻辑，**一个词写错整条查询归零**，而且返回空列表不报错
- `safebooru.org` 虽叫 safe，98% 的图被标成 q 级；它的 dapi **不支持服务端分级过滤**
- 国内直连：danbooru / konachan / yande 全超时，safebooru.com 被 Cloudflare 拦

完整版见 [`高质量模式说明.txt`](高质量模式说明.txt)（7 万字，按时间顺序）和
[`ComfyUI使用说明.html`](ComfyUI使用说明.html)（结构化手册，带搜索）。

## 目录

```
├── make.py / switch.py / ref_search.py …   40 个脚本
├── workflows/          01-02  SD1.5 动画（旧）
├── workflows_hq/       03-08  SDXL / 3DCG / FLUX
├── profiles/           6 个画风档 json
├── *.bat               启动器（GBK + CRLF，见下）
├── nl2tags.py          零依赖，可单独拿去用
└── 高质量模式说明.txt    技术日志
```

## 移植到你的机器

三步：

1. **改路径**——每个脚本头部有两行常量：
   ```python
   BASE = r"D:\AI-Video-Setup"
   COMFY = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI"
   ```
2. **下模型**——按 `hq-downloads.txt` / `model-downloads.txt` 里的文件名放进 `models/` 对应目录
3. **准备素材**——`refs/char/` 放你的角色参考图（`prep_refs.py` 会自动抠图并按脸裁方形）

> `.bat` 文件必须是 **GBK + CRLF + 不带 `chcp`**。改成 UTF-8 会让 cmd 把中文命令
> 拆错位，报 `'xxx' is not recognized`。这个坑踩过不止一次。

## Roadmap

这套基础设施（扒图 / 提骨架 / 抠图 / 分层）能复用到的地方，比现在做的多：

- [ ] Q 版动图 —— SD1.5 + AnimateDiff，`sprite` 档已有雏形
- [ ] Live2D / 皮套 —— 分层输出（前发/后发/身体/眼部），需要改 `layer_paint.py`
- [ ] 短视频 —— 现有动画工作流接音轨
- [ ] 像素帧 —— 出图后做调色板量化 + 降采样
- [ ] 三视图 —— 同一角色锁 + 三种骨架朝向
- [ ] TripoSG —— 单图转 3D，抠图那一步可以直接复用

## 关于素材与版权

**仓库不含任何图片。** `refs/`、`output/` 全部排除，原因很直接：
里面的图是别人的作品。

`ref_search.py` 只是一个检索工具——它帮你去公开图站按标签找图，
不附带任何图。用它扒到的图怎么用、能不能用，是使用者自己的事。
如果你要拿它建训练集，请先确认素材的授权。

## License

MIT，见 [LICENSE](LICENSE)。
随便用，出问题别找我。
