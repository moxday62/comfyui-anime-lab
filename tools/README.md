# tools —— 单点小工具

这一层的东西和主链路（出图）不耦合，什么时候想用就什么时候用。

## 像素小人.html

双击打开，或者用根目录的 `10_像素小人.bat`。

把图拖进去（也可以直接 `Ctrl+V` 粘一张截图），然后拉滑条：

| 滑条 | 作用 | 调的时候注意 |
|---|---|---|
| 像素块大小 | 原图宽度 ÷ 这个数 = 像素网格宽度 | 太小只是"糊"，太大连五官都没了。768 宽的人像，5~10 比较合适 |
| 颜色数 | 减到几种颜色（0 = 不减） | 16~32 是像素画的常见区间；4~8 是 GameBoy 那种味道 |
| 抖动 | Floyd–Steinberg | 开着渐变过渡自然，但会在暗部撒点噪；要"干净平涂"就关掉 |
| 亮度 / 对比度 / 饱和度 | 预处理 | 像素化之后细节会损失，**先在这里补对比度**比事后调有用 |
| 预览放大 | 只是看，不影响导出 | 放大必须用最近邻，否则像素块之间会出现过渡色 |

导出是 8 倍最近邻放大的 PNG（`pixel_宽x高.png`），贴到别处不会被浏览器二次插值糊掉。

**实现说明**：单文件、零依赖、离线可用。减色用的是 median cut（按颜色分布切盒子取平均），
比固定色板贴合原图；抖动按 7/3/5/1 把误差分给右、左下、下、右下。

## pixelate.py

同样的算法，命令行，用于批量。调参在网页里拉，拉好把数字抄过来：

```bash
python pixelate.py output/make/1119_202528/01_shot.png -s 6 -c 16
python pixelate.py output/make -o output/pixel --scale 6 --colors 16 -r 4
```

| 参数 | 含义 |
|---|---|
| `-s/--scale` | 像素块大小，对应网页第一根滑条 |
| `-c/--colors` | 颜色数，0 = 不减 |
| `--no-dither` | 关抖动 |
| `-r/--ratio` | 导出放大倍数（最近邻） |

缩小用 LANCZOS（面积平均），放大必须 NEAREST —— 反了就不像像素画。

`output/pixel/` 已经在 `.gitignore` 里，产物不会被带进仓库。

## pixelit/

第三方库，**不是我的代码**，原样收在这里：

- 上游：https://github.com/giventofly/pixelit
- 作者：José Moreira
- 许可证：MIT（见 `pixelit/LICENSE`）
- 保留内容：`dist/pixelit.js`、`dist/pixelit.min.js`、上游 README

它是"JS 像素化"里最成熟的一个（1.8k star），API 是链式的：

```js
const px = new pixelit({ scale: 8, palette: [[0,0,0],[255,255,255]] });
px.draw().pixelate().convertPalette();
```

和上面 `像素小人.html` 的区别：pixelit 走**指定色板**（你把颜色列给它，它做最近色匹配），
适合"我要严格只用这 8 个颜色"；我的页面走 **median cut 自动减色**，适合"我也不确定要几种颜色，先看效果"。
两边都留着，场景不同。

上游的演示页（带滑条的版本）在 `docs/` 里，为了控制体积没有收进来，
需要的话去仓库根目录 `git clone` 或看 https://giventofly.github.io/pixelit 。
