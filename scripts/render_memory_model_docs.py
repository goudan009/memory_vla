"""Render the code-derived V1/V2 teaching diagrams. No model or data is loaded.

Run with the existing V1 Python environment; needs Graphviz and Pillow.
DOT/SVG are editable diagram sources, PNG/GIF work in Markdown viewers.
"""
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "src/v1_groot_memoryvla/media/model"
V2 = ROOT / "src/v2_groot_memoryvla/media/model"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def render(folder, name, body, direction="TB"):
    folder.mkdir(parents=True, exist_ok=True)
    source = f'''digraph model {{
      graph [rankdir={direction}, bgcolor="#ffffff", pad="0.3", nodesep="0.34", ranksep="0.55",
             fontname="Noto Sans CJK SC", fontsize=20, compound=true];
      node [shape=box, style="rounded,filled", color="#cbd5e1", fillcolor="#f1f5f9",
            fontcolor="#172554", fontname="Noto Sans CJK SC", fontsize=17, margin="0.20,0.15", penwidth=1.3];
      edge [color="#64748b", penwidth=1.6, arrowsize=0.8, fontname="Noto Sans CJK SC", fontsize=13];
      {body}
    }}'''
    dot = folder / f"{name}.dot"
    dot.write_text(source)
    for kind in ("svg", "png"):
        subprocess.run(["dot", f"-T{kind}", "-Gdpi=125", str(dot), "-o", str(folder / f"{name}.{kind}")], check=True)
    png = folder / f"{name}.png"
    with Image.open(png) as im:
        if im.width > 1900:
            im = im.resize((1900, round(im.height * 1900 / im.width)), Image.Resampling.LANCZOS)
            im.save(png)


def diagrams():
    render(V1, "architecture", r'''
      obs [label="当前三视角 RGB + 任务指令\n每次只输入当前帧"];
      eagle [label="① 原 Eagle VLM\n正式训练冻结", fillcolor="#dbeafe", color="#60a5fa"];
      visual [label="LLM 前视觉投影 V\n[B,3,81,2048]", fillcolor="#dbeafe"];
      eos [label="LLM 后 H 的 user EOS\n[B,1,2048]", fillcolor="#dbeafe"];
      per [label="② 新增感知编码与记忆\nBottleneckSE → 检索 → 门控\n[B,243,256]", fillcolor="#ccfbf1", color="#14b8a6"];
      cog [label="③ 新增认知记忆\n检索 → 门控\n[B,1,2048]", fillcolor="#ccfbf1", color="#14b8a6"];
      ph [label="过去感知 bank\n严格早于当前决策", fillcolor="#f0fdfa"];
      ch [label="过去认知 bank\n严格早于当前决策", fillcolor="#f0fdfa"];
      projection [label="④ 新增线性接口\n256 → 2048", fillcolor="#ccfbf1", color="#14b8a6"];
      condition [label="⑤ direct 条件 Z\n243 个感知 Token + 1 个认知 Token\n[B,244,2048]", fillcolor="#fef3c7", color="#f59e0b"];
      state [label="当前 state [B,1,128]\n原 state_encoder → [B,1,1536]", fillcolor="#dbeafe"];
      noisy [label="噪声动作 / 中间动作 + flow time\n原 action_encoder", fillcolor="#dbeafe"];
      head [label="⑥ 原 vlln + 32 层动作专家 + 解码器\n训练：预测速度场；推理：4 次 Euler 更新", fillcolor="#dbeafe", color="#60a5fa"];
      action [label="模型动作 [B,50,128]\nprocessor 解码为 [B,32,23]", fillcolor="#fef3c7"];
      obs -> eagle; eagle -> visual; eagle -> eos;
      visual -> per; eos -> cog; ph -> per [style=dashed,label="读取过去"]; ch -> cog [style=dashed,label="读取过去"];
      per -> projection; projection -> condition; cog -> condition;
      condition -> head; state -> head; noisy -> head; head -> action;
      {rank=same; visual; eos;} {rank=same; per; cog;}
      {rank=same; state; head; noisy;}
    ''')
    render(V1, "conditioning", r'''
      subgraph cluster_base {label="A · 原版 Baseline"; color="#93c5fd";
        bh [label="Eagle 完整上下文 H\n图像 + 指令 + 特殊 Token", fillcolor="#dbeafe"];
        bz [label="[B,S,2048]\n保留原 image / attention masks"];
        ba [label="原动作专家", fillcolor="#dbeafe"]; bh -> bz -> ba; }
      subgraph cluster_old {label="B · 旧 V1：residual"; color="#cbd5e1";
        rh [label="Eagle 完整上下文 H", fillcolor="#dbeafe"];
        rd [label="图像位置加 Wr(P̂−P)\nuser EOS 加 Ĉ−C", fillcolor="#ccfbf1"];
        rz [label="[B,S,2048]\n无历史时残差 = 0"];
        ra [label="原动作专家", fillcolor="#dbeafe"]; rh -> rd -> rz -> ra; }
      subgraph cluster_new {label="C · 当前 V1：direct"; color="#14b8a6";
        dp [label="融合感知 P̂ → 投影\n243 × 2048", fillcolor="#ccfbf1"];
        dc [label="融合认知 Ĉ\n1 × 2048", fillcolor="#ccfbf1"];
        dz [label="拼接为 [B,244,2048]\nimage_mask = 243 个真 + 1 个假", fillcolor="#fef3c7"];
        da [label="原动作专家", fillcolor="#dbeafe"]; dp -> dz; dc -> dz; dz -> da; }
      {rank=same; bh; rh; dp; dc;} {rank=same; ba; ra; da;}
    ''')
    render(V1, "memory_bank", r'''
      current [label="当前 Token Xₜ\n感知：[1,243,256] / 认知：[1,1,2048]", fillcolor="#ccfbf1"];
      history [label="历史融合特征 Fᵢ（i < t）\n持久记录已 detach", fillcolor="#f0fdfa"];
      pe [label="历史决策索引 → sin/cos → MLP\n仅加入 Key"];
      kv [label="K = history + PE\nV = history", fillcolor="#f0fdfa"];
      retrieval [label="两层 CrossTransformerBlock\nQ 投影 / K,V 投影 → Attention\n残差 + LN → FFN → 残差 + LN", fillcolor="#ccfbf1", color="#14b8a6"];
      fusion [label="门控融合\ng = sigmoid(W[Xₜ ; Rₜ] + b)\nFₜ = g·Xₜ + (1−g)·Rₜ", fillcolor="#ccfbf1"];
      head [label="当前动作条件 → 动作 loss\n本步编码 / 检索 / 门控可训练", fillcolor="#fef3c7"];
      write [label="detach().clone() 后写入 Fₜ\n超容量：合并最相似相邻记录", fillcolor="#f0fdfa"];
      current -> retrieval [label="Q"]; history -> kv; pe -> kv; kv -> retrieval;
      retrieval -> fusion [label="Rₜ"]; current -> fusion [label="Xₜ"];
      fusion -> head; fusion -> write [style=dashed,label="切断跨历史梯度"];
      {rank=same; current; history; pe;} {rank=same; head; write;}
    ''')
    render(V1, "training", r'''
      data [label="已有示范 Episode：30 Hz\nRGB / 82维 state / language / 23维 action"];
      sampling [label="有序观测采样：原始帧 0,8,16,…\n每窗最多32决策，保留真实短尾窗", fillcolor="#ccfbf1"];
      vision [label="训练 processor：每次访问重新增强\n冻结 Eagle：no_grad 编码当前图像/指令", fillcolor="#dbeafe"];
      memory [label="可训练双路记忆 → direct 条件\nEpisode 内跨 microbatch/窗口延续", fillcolor="#ccfbf1"];
      labels [label="标签 a：当前帧起未来32步\n归一化并填充至 [B,50,128]\n超出Episode：action=0、mask=0", fillcolor="#fff7ed"];
      noise [label="训练 flow time τ + 随机噪声 ε\nxτ = (1−τ)ε + τa"];
      head [label="原 state/action 编码器 + 动作专家\n预测速度 vθ", fillcolor="#dbeafe"];
      loss [label="masked MSE：vθ 对比 a−ε\n按有效 action_mask 元素全局加权", fillcolor="#fef3c7"];
      grad [label="反向更新记忆 + 原动作头\nEagle冻结；不穿过旧历史计算图", fillcolor="#fef3c7"];
      data -> sampling; sampling -> vision; vision -> memory; memory -> head;
      sampling -> head [label="当前 state → 原状态编码器"];
      sampling -> labels; labels -> noise; noise -> head; head -> loss; labels -> loss [label="目标 / mask"];
      loss -> grad;
    ''')
    render(V2, "architecture", r'''
      vl [label="当前 RGB + Language\n原 Eagle → V、H", fillcolor="#dbeafe"];
      v1 [label="原 V1 双路记忆与条件接口\ndirect Z：[B,244,2048]\n旧 residual：[B,S,2048]", fillcolor="#ccfbf1", color="#14b8a6"];
      robot [label="URDF + 当前 q / qvel / 局部底盘速度"];
      body [label="① V2a/b/c：BodyEncoder\n关节 Token [B,N,128] → pooled Body bₜ", fillcolor="#ede9fe", color="#8b5cf6"];
      bodydelta [label="当前 Body 投影 + tanh gate\nΔbody ∈ 2048维", fillcolor="#ede9fe"];
      query [label="③ V2c：本体条件 Query\nf(当前视觉均值, 原 user EOS, bₜ)\nV2b 把 Body 输入置零", fillcolor="#ede9fe", color="#8b5cf6"];
      bank [label="② V2b/c：过去具身历史\nBody + 视觉 + EOS + 先前已执行命令\n保存 detached 输入，读取时重新编码", fillcolor="#f5f3ff"];
      retrieve [label="Query → K/V Attention\n检索过去，得到 256维向量", fillcolor="#ede9fe"];
      memdelta [label="历史投影 + tanh gate\nΔmemory ∈ 2048维", fillcolor="#ede9fe"];
      add [label="④ 仅在条件认知 Token 上加残差\nZ[cog] += Δbody + Δmemory", fillcolor="#fef3c7"];
      state [label="当前 state → 原 state_encoder\n动作 / 噪声 → 原 action_encoder", fillcolor="#dbeafe"];
      head [label="原 vlln → 原32层动作专家 → 动作\n状态/动作编解码与控制接口保留", fillcolor="#dbeafe"];
      vl -> v1; robot -> body; body -> bodydelta; body -> query;
      vl -> query [label="原 V / EOS"]; query -> retrieve [label="Q"]; bank -> retrieve [label="K,V"];
      retrieve -> memdelta; v1 -> add; bodydelta -> add; memdelta -> add; add -> head; state -> head;
      {rank=same; vl; robot;} {rank=same; v1; body;} {rank=same; query; bank;}
    ''')
    render(V2, "body_encoder", r'''
      urdf [label="URDF：父子边 / 类型 / 轴 / origin\nlimits / 质量 / 惯性 / 几何", fillcolor="#f5f3ff"];
      mapping [label="显式 mapping\n关节名称 ↔ OG索引 ↔ state / action维", fillcolor="#f5f3ff"];
      measured [label="当前 q / 可用 qvel\n当前局部底盘速度 + availability mask"];
      robotgraph [label="RobotGraph\n结构属性 [N,27] + 拓扑 mask", fillcolor="#ede9fe"];
      fk [label="base_link 系 FK（当前观测）\n位置3 + 旋转6 + 有效位", fillcolor="#ede9fe"];
      dyn [label="动态属性 [B,N,16]\nq编码 / qvel / 测量mask / FK / FK有效位", fillcolor="#ede9fe"];
      embed [label="结构MLP + 动态MLP + 底盘Linear\n统一到128维", fillcolor="#ede9fe"];
      attn [label="2层拓扑 Transformer（4 heads）\n仅允许图定义的局部交互", fillcolor="#ede9fe", color="#8b5cf6"];
      tokens [label="Body Tokens [B,N,128]\nN随图变化；Padding 与缺测不同"];
      pooled [label="排除 Padding 后均值池化\nBody bₜ：[B,128]", fillcolor="#fef3c7"];
      urdf -> robotgraph; mapping -> measured; robotgraph -> fk; measured -> fk;
      measured -> dyn; fk -> dyn; robotgraph -> embed; dyn -> embed; measured -> embed [label="局部速度6维"];
      embed -> attn; robotgraph -> attn [style=dashed,label="拓扑约束"]; attn -> tokens -> pooled;
    ''')
    render(V2, "retrieval", r'''
      subgraph cluster_write {label="历史内容：V2b / V2c 相同"; color="#c4b5fd";
        old [label="历史时刻 i 的 Body(128)\n视觉均值(2048) + 原EOS(2048)\n此前执行命令摘要(31)"];
        raw [label="4255维输入 detach 后保存\n不保存检索融合后的结果", fillcolor="#f5f3ff"];
        enc [label="读取时：历史MLP 4255→256→256\nK + 相对年龄编码；V 保留内容", fillcolor="#ede9fe"];
        old -> raw -> enc; }
      subgraph cluster_query {label="唯一对照：当前 Body 是否进入 Query"; color="#c4b5fd";
        current [label="当前视觉均值 + 原 user EOS"];
        b [label="V2b：Body 输入 = 0", fillcolor="#f1f5f9"];
        c [label="V2c：Body 输入 = bₜ", fillcolor="#ede9fe"];
        query [label="相同 QueryGenerator → Q ∈ 256维", fillcolor="#ede9fe"];
        current -> query; b -> query [style=dashed]; c -> query; }
      attention [label="Attention(Q,K,V) → 历史向量\n点积 / softmax / 加权和保留FP32", fillcolor="#ede9fe", color="#8b5cf6"];
      proj [label="投影256→2048 + Memory gate\n加到认知条件Token → 原动作专家", fillcolor="#fef3c7"];
      enc -> attention; query -> attention; attention -> proj;
    ''')


def memory_animation():
    """Conceptual event order, not a rollout or attention measurement."""
    V1.mkdir(parents=True, exist_ok=True)
    font = lambda size: ImageFont.truetype(FONT, size)
    phases = [
        ("1 / 6  接收新观测", "读取当前 RGB / Language / State；决策时钟 t=2", 0),
        ("2 / 6  读取严格过去", "用当前特征查询 t=0、t=1；此时库中没有 t=2", 1),
        ("3 / 6  融合为当前条件", "感知与认知分别检索、门控，再组成 244 个条件 Token", 2),
        ("4 / 6  写入一次记忆", "融合特征 detach 后写 t=2；不把动作预测块当历史观测", 3),
        ("5 / 6  生成动作块", "同一份条件用于 4 次 Euler 更新；不会写入 4 次记忆", 4),
        ("6 / 6  执行后再观测", "执行前 8 步；下一次请求带来最新观测并进入 t=3", 5),
    ]
    boxes = [(70,170,350,290), (460,170,740,290), (850,170,1130,290),
             (70,380,350,500), (460,380,740,500), (850,380,1130,500)]
    labels = [("当前观测", "RGB / Language / State"), ("双路检索", "当前 Q → 过去 K、V"),
              ("融合条件", "243 感知 + 1 认知"), ("历史 bank", "t=0   |   t=1"),
              ("原动作专家", "Euler × 4 → 动作块"), ("机器人执行", "只执行块的前 8 步")]
    frames=[]
    def arrow(draw, start, end, fill="#94a3b8"):
        draw.line([start,end],fill=fill,width=4)
        x,y=end;dx,dy=x-start[0],y-start[1]
        if abs(dx)>abs(dy): draw.polygon([(x,y),(x-12 if dx>0 else x+12,y-7),(x-12 if dx>0 else x+12,y+7)],fill=fill)
        else: draw.polygon([(x,y),(x-7,y-12 if dy>0 else y+12),(x+7,y-12 if dy>0 else y+12)],fill=fill)
    for title,caption,active in phases:
        im=Image.new("RGB",(1200,700),"#f8fafc");d=ImageDraw.Draw(im)
        d.text((55,28),"V1 记忆读写与动作执行：一次决策的先后顺序",font=font(30),fill="#0f172a")
        d.text((55,90),title,font=font(25),fill="#0f766e")
        selected=[0,1,2,3,4,5][active]
        arrow(d,(350,230),(460,230));arrow(d,(740,230),(850,230))
        d.line([(350,410),(410,410),(410,355),(510,355)],fill="#0f766e",width=4)
        arrow(d,(510,355),(510,290),"#0f766e")
        arrow(d,(990,290),(990,340));arrow(d,(990,340),(600,340));arrow(d,(600,340),(600,380))
        d.line([(990,290),(990,315),(210,315)],fill="#0f766e",width=4)
        arrow(d,(210,315),(210,380),"#0f766e")
        arrow(d,(740,440),(850,440))
        d.text((225,330),"融合后写一次",font=font(16),fill="#0f766e")
        for i,(box,label) in enumerate(zip(boxes,labels)):
            fill="#ccfbf1" if i==selected else "#ffffff"
            d.rounded_rectangle(box,16,fill=fill,outline="#0d9488" if i==selected else "#cbd5e1",width=4 if i==selected else 2)
            x,y,_,_=box
            d.text((x+20,y+18),label[0],font=font(26),fill="#172554")
            detail="t=0   |   t=1   |   t=2" if i==3 and active>=3 else label[1]
            d.text((x+20,y+70),detail,font=font(18),fill="#475569")
        d.rounded_rectangle((55,550,1145,620),14,fill="#e2e8f0")
        d.text((75,571),caption,font=font(22),fill="#0f172a")
        d.text((55,647),"流程示意：不是机器人实测视频；t 是策略决策索引，不是 flow time。",font=font(19),fill="#64748b")
        frames.append(im)
    frames[0].save(V1/"memory_cycle.gif",save_all=True,append_images=frames[1:],duration=[1700]*6,loop=0)
    frames[3].save(V1/"memory_cycle.png")


if __name__ == "__main__":
    diagrams()
    memory_animation()
    print("Rendered 7 architecture figures (DOT/SVG/PNG), one six-stage GIF and its static frame.")
