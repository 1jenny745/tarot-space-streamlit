"""塔罗心灵空间 · Streamlit 版（单文件应用）。"""

import os
import json
import requests

import streamlit as st

from tarot_data import (
    CARDS_BY_ID,
    SPREADS,
    QUESTION_TYPES,
    CONCERNS,
    get_safety_concern,
    get_safety_message,
    is_meaningful_question,
    recommend_spread,
    get_spread_reason,
    shuffled_deck,
)

st.set_page_config(page_title="塔罗心灵空间", page_icon="🔮", layout="centered")

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")

READING_SYSTEM_PROMPT = """你是「塔罗心灵空间」的AI塔罗咨询师，一位温暖、富有同理心的资深塔罗师。

【身份定位】
- 你是用户的陪伴者，帮助他们通过塔罗进行自我探索和情绪疏导
- 塔罗是自我探索的工具，帮助人们获得启发和行动建议
- 你不预测未来，而是帮助用户理解当下的状态和可能的心理模式

【核心原则】
1. 温暖共情：始终保持温暖、理解的语气
2. 不制造焦虑：避免负面恐吓性的解读
3. 理性客观：基于牌面提供平衡的分析
4. 引导行动：给出实用的建议而非空洞的理论
5. 不承诺确定：使用"可能""或许"等词语，保持开放性
6. 个性化：必须回应用户的具体处境、顾虑和补充背景，不能只罗列通用牌义
7. 有用性：每一条行动建议都要对应牌面或用户的现实约束，并说明为什么值得做
8. 信息边界：不得编造用户没有提供的经历、人物、情绪或事实。信息不足时明确指出需要确认什么。

【禁止事项】
- 宿命论表述（"你注定..." "必然..."）
- 医疗诊断或治疗建议
- 法律判断
- 投资建议
- 过度负面的解读

【解读结构】
请按以下结构进行解读：
1. 【你正在面对的核心矛盾】用1-2句话复述真正的取舍，不替用户下结论
2. 【牌面如何映照当下】依次解读每张牌、位置和正逆位，并联系用户背景
3. 【值得留意的盲点】说明还缺少什么信息或容易被什么情绪推动
4. 【本周行动计划】给出3件低风险、可验证、具体的行动；不要建议医疗、法律、金融决策
5. 【带着它继续探索】给出一个开放式追问

全文控制在500-700字。不要使用"亲爱的朋友"、冗长安慰或夸张修辞。请用中文回复，语气温暖、具体且克制。"""

CHAT_SYSTEM_PROMPT = """你是「塔罗心灵空间」的AI塔罗咨询师，一位温暖、富有同理心的资深塔罗师。

【身份定位】
- 你是用户的陪伴者，帮助他们通过塔罗进行自我探索和情绪疏导
- 塔罗是自我探索的工具，帮助人们理解当下的状态和可能的心理模式
- 你善于提出引导性的问题，帮助用户深入思考

【连续对话原则】
- 只围绕本轮已抽出的牌与已有上下文继续探索，不重新抽牌，也不声称新的牌面信息
- 优先帮助用户澄清选择、验证假设或安排一个低风险的下一步

【禁止事项】
- 宿命论表述
- 过度负面的预测
- 评判或说教

请用中文回复，保持温暖而有深度的对话风格。"""


def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def deepseek_key():
    return os.environ.get("DEEPSEEK_API_KEY") or get_secret("DEEPSEEK_API_KEY")


def llm_complete(messages: list) -> str:
    """非流式调用 DeepSeek，返回完整文本。"""
    key = deepseek_key()
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "stream": False,
            "temperature": 0.75,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 请求失败：{resp.status_code}")
    return resp.json()["choices"][0]["message"]["content"]


def llm_stream(messages: list):
    """流式调用 DeepSeek，逐段 yield 文本。"""
    key = deepseek_key()
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "stream": True,
            "temperature": 0.75,
        },
        stream=True,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 请求失败：{resp.status_code}")
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)["choices"][0]["delta"].get("content")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if chunk:
            yield chunk


def build_cards_description(drawn_cards, spread_type):
    positions = SPREADS[spread_type]["positions"]
    parts = []
    for i, (card, is_reversed) in enumerate(drawn_cards):
        _, name, _en, suit, up, rev = card
        meaning = rev if is_reversed else up
        pos_name, pos_desc = positions[i]
        parts.append(
            f"第{i + 1}张牌【{pos_name}】：{name}（{'逆位' if is_reversed else '正位'}）\n"
            f"位置含义：{pos_desc}\n牌义：{meaning}"
        )
    return "\n\n".join(parts)


# ---- 可选登录门 ----
def require_password():
    pwd = os.environ.get("APP_PASSWORD") or get_secret("APP_PASSWORD")
    if not pwd:
        return True
    if st.session_state.get("_authed"):
        return True
    st.title("🔮 塔罗心灵空间")
    user_input = st.text_input("请输入访问密码", type="password")
    if st.button("进入"):
        if user_input == pwd:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("密码错误")
    return False


# ---- 初始化会话状态 ----
def init_state():
    defaults = {
        "stage": "home",
        "question": "",
        "question_type": "career",
        "concern": "选择",
        "goal": "",
        "spread": "one_card",
        "drawn_cards": [],
        "deck": [],
        "picked": [],
        "reading": "",
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset():
    for k in ["question", "concern", "goal", "drawn_cards", "deck", "picked", "reading", "chat_history"]:
        st.session_state[k] = [] if k in ("drawn_cards", "deck", "picked", "chat_history") else ""
    st.session_state["question_type"] = "career"
    st.session_state["spread"] = "one_card"
    st.session_state["stage"] = "home"


def home_page():
    st.markdown("## 🔮 塔罗心灵空间")
    st.markdown("不是“AI 算命”，而是以塔罗为低压力交互媒介的 **AI 决策陪伴产品**。")
    st.markdown("把模糊的焦虑转成可讨论的问题，在牌义映照和 AI 引导中看清现实约束，并形成下一步行动。")
    st.markdown("---")
    if st.button("开始探索 →", type="primary", width="stretch"):
        st.session_state["stage"] = "question"
        st.rerun()


def question_page():
    st.markdown("## 告诉我，你最近在想什么")

    labels = [t["label"] for t in QUESTION_TYPES]
    current_type = st.session_state["question_type"]
    default_idx = next((i for i, t in enumerate(QUESTION_TYPES) if t["id"] == current_type), 1)
    sel_label = st.radio("选择领域", labels, index=default_idx, horizontal=True)
    sel_type = next(t for t in QUESTION_TYPES if t["label"] == sel_label)

    if sel_type["examples"]:
        st.caption("示例：" + " ／ ".join(sel_type["examples"]))

    question = st.text_input("你的问题", value=st.session_state["question"], placeholder=sel_type["placeholder"])

    concern = st.selectbox("你最在意什么", CONCERNS, index=CONCERNS.index(st.session_state["concern"]) if st.session_state["concern"] in CONCERNS else 0)
    goal = st.text_area("补充背景（可选）", value=st.session_state["goal"], placeholder="例如：我刚拿到实习 Offer，但担心影响毕业进度")

    if st.button("生成牌阵推荐 →", type="primary", width="stretch"):
        if not is_meaningful_question(question):
            st.warning("问题太短或信息不足，请写得更具体一些（至少 8 个字，例如“我该不该接受这个 offer”）。")
            return
        safety = get_safety_concern(f"{question} {goal}")
        if safety:
            st.session_state["stage"] = "safety"
            st.session_state["safety_msg"] = get_safety_message(safety)
            st.rerun()

        st.session_state["question"] = question
        st.session_state["question_type"] = sel_type["id"]
        st.session_state["concern"] = concern
        st.session_state["goal"] = goal
        st.session_state["spread"] = recommend_spread(question, concern, goal)
        st.session_state["drawn_cards"] = []
        st.session_state["deck"] = shuffled_deck()
        st.session_state["picked"] = []
        st.session_state["reading"] = ""
        st.session_state["chat_history"] = []
        st.session_state["stage"] = "draw"
        st.rerun()


def safety_page():
    st.markdown("## ⚠️ 安全提示")
    st.markdown(st.session_state.get("safety_msg", ""))
    if st.button("返回重新提问", type="primary", width="stretch"):
        st.session_state["stage"] = "question"
        st.rerun()


def draw_page():
    spread = st.session_state["spread"]
    info = SPREADS[spread]
    need = info["card_count"]
    positions = info["positions"]
    deck = st.session_state["deck"]
    picked = st.session_state["picked"]

    if not deck:
        st.session_state["deck"] = shuffled_deck()
        deck = st.session_state["deck"]

    st.markdown(f"## 请选择你的牌 · {info['name']}")
    st.caption(f"AI 推荐：{info['name']}，共需抽取 {need} 张。请在下方凭直觉点击你想要的牌，牌由你亲手翻开。")
    st.info(get_spread_reason(spread))

    remaining = need - len(picked)
    if remaining > 0:
        st.markdown(f"还需抽取 **{remaining}** 张")
    else:
        st.success("已抽满，可以查看解读了")

    # 已抽出的牌（按顺序对应牌阵位置）
    if picked:
        cols = st.columns(len(picked))
        for col, idx, pos in zip(cols, picked, positions):
            card, is_reversed = deck[idx]
            card_id, name, name_en, suit, up, rev = card
            with col:
                img_path = os.path.join(IMAGE_DIR, f"{card_id}.png")
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")
                st.markdown(f"**{pos[0]}** · {name}")
                st.caption(f"{'逆位' if is_reversed else '正位'} · {suit}")

    # 78 张背面牌网格
    st.markdown("#### 牌堆（点击选取）")
    full = len(picked) >= need
    ncols = 10
    cols = st.columns(ncols)
    for i in range(len(deck)):
        with cols[i % ncols]:
            if i in picked:
                card, is_reversed = deck[i]
                img_path = os.path.join(IMAGE_DIR, f"{card[0]}.png")
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")
            else:
                if st.button("✦", key=f"pick_{i}", disabled=full, width="stretch"):
                    st.session_state["picked"].append(i)
                    st.session_state["drawn_cards"] = [deck[j] for j in st.session_state["picked"]]
                    st.session_state["reading"] = ""
                    st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("重新洗牌", type="primary", width="stretch"):
            st.session_state["deck"] = shuffled_deck()
            st.session_state["picked"] = []
            st.session_state["drawn_cards"] = []
            st.rerun()
    with c2:
        if len(picked) >= need:
            if st.button("查看解读 →", type="primary", width="stretch"):
                st.session_state["stage"] = "reading"
                st.rerun()
        else:
            st.button("查看解读 →", width="stretch", disabled=True)


def reading_page():
    spread = st.session_state["spread"]
    info = SPREADS[spread]

    st.markdown("## 你的解读")

    if not st.session_state["reading"]:
        cards_desc = build_cards_description(st.session_state["drawn_cards"], spread)
        user_prompt = (
            f"用户问题：{st.session_state['question']}\n"
            f"话题类型：{st.session_state['question_type']}\n"
            f"用户最在意：{st.session_state['concern']}\n"
            f"补充背景：{st.session_state['goal'] or '未说明'}\n"
            f"牌阵类型：{info['name']}（{info['description']}）\n\n"
            f"抽到的牌：\n{cards_desc}\n\n请为用户解读这次塔罗占卜。"
        )
        messages = [
            {"role": "system", "content": READING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            st.session_state["reading"] = st.write_stream(llm_stream(messages))
        except Exception as e:
            st.error(f"AI 解读服务尚未连接：{e}")
            return
    else:
        st.markdown(st.session_state["reading"])

    # 牌面回顾
    with st.expander("查看本次牌面"):
        positions = info["positions"]
        cols = st.columns(len(st.session_state["drawn_cards"]))
        for col, (card, is_reversed), pos in zip(cols, st.session_state["drawn_cards"], positions):
            card_id, name, name_en, suit, up, rev = card
            with col:
                img_path = os.path.join(IMAGE_DIR, f"{card_id}.png")
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")
                st.caption(f"{pos[0]} · {name} · {'逆位' if is_reversed else '正位'}")

    st.markdown("---")
    st.markdown("### 继续追问")
    follow_up = st.text_input("还有想厘清的吗？围绕这一轮牌继续聊", key="follow_up_input")
    if st.button("发送追问", type="primary"):
        if not follow_up.strip():
            st.warning("请输入追问内容")
        else:
            safety = get_safety_concern(follow_up)
            if safety:
                st.info(get_safety_message(safety))
            else:
                cards_desc = build_cards_description(st.session_state["drawn_cards"], spread)
                context_prompt = (
                    f"这是之前的塔罗占卜信息：\n问题：{st.session_state['question']}\n"
                    f"用户最在意：{st.session_state['concern']}\n补充背景：{st.session_state['goal'] or '未说明'}\n"
                    f"牌阵：{info['name']}\n抽到的牌：\n{cards_desc}"
                )
                messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, {"role": "user", "content": context_prompt}]
                for m in st.session_state["chat_history"]:
                    messages.append(m)
                messages.append({"role": "user", "content": follow_up})
                try:
                    answer = llm_complete(messages)
                    st.session_state["chat_history"].append({"role": "user", "content": follow_up})
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"追问失败：{e}")

    # 历史追问展示
    for m in st.session_state["chat_history"]:
        role = "你" if m["role"] == "user" else "AI"
        with st.chat_message(role):
            st.markdown(m["content"])

    if st.button("开启一次新的探索", type="primary", width="stretch"):
        reset()
        st.rerun()


CARD_CSS = """
<style>
section.main button[data-testid="stBaseButton-secondary"],
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]{
  aspect-ratio:2/3;width:100%!important;min-height:0!important;
  background:linear-gradient(135deg,#241b45,#3b2a6b)!important;
  color:#d9b563!important;border:1px solid #b98b3566!important;
  border-radius:8px!important;font-size:16px!important;padding:0!important;
  transition:transform .15s ease, box-shadow .15s ease;
}
section.main button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover{
  border-color:#b98b35!important;box-shadow:0 0 12px #b98b3588;
  transform:scale(1.06);
}
section.main button[data-testid="stBaseButton-secondary"]:disabled,
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:disabled{
  opacity:.35;
}
</style>
"""


def main():
    if not require_password():
        return
    init_state()
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🔮 塔罗心灵空间")
        st.caption("AI 决策陪伴 · 已上线 MVP")
        if st.button("🏠 回到首页"):
            reset()
            st.rerun()
        st.markdown("---")
        st.caption("内容仅供自我探索与情绪梳理，不构成医疗、法律或投资建议。")

    stage = st.session_state["stage"]
    if stage == "home":
        home_page()
    elif stage == "question":
        question_page()
    elif stage == "safety":
        safety_page()
    elif stage == "draw":
        draw_page()
    elif stage == "reading":
        reading_page()


if __name__ == "__main__":
    main()
