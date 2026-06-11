# Type2 提交前未完成事项(非代码项)

> 依据《EXACT 2026 - Submission Guide》(IEEE IJCNN 2026)整理,只覆盖 **Type2** 相关事项,Type1 不在本清单范围内。
> **提交截止:2026 年 6 月 12 日**。提交方式:报名表单(Discord / 官网公布),备用邮箱 ura.hcmut@gmail.com。

## 状态总览

| # | 事项 | 状态 | 负责内容 |
|---|------|------|----------|
| 1 | 公网暴露 `/predict` 与 vLLM `/v1/models` | ✅ 完成(2026-06-11,Cloudflare Tunnel) | 部署 |
| 2 | `urls.txt` | ✅ 完成(`E:\LLM-vllm\urls.txt`) | 提交包 |
| 3 | `solution.pdf`(一页方案说明) | ✅ 完成(Team: superNB) | 提交包 |
| 4 | `source_code.zip` | ✅ 完成(18.2MB,87 个文件含 retained 模型权重) | 提交包 |
| 5 | Notation Mapping CSV 填写 | ✅ 本地版已填入包;委员会发正式模板后誊写并重打包 | 提交包 |
| 6 | Grading slot(评分时段)报名 | ❌ 未做(Discord 表单,人工) | 流程 |
| 7 | 打包 `<team_name>.zip` 并提交 | ✅ `E:\LLM-vllm\superNB.zip` 已就绪,**待提交**(表单/邮箱 ura.hcmut@gmail.com,截止 6/12) | 流程 |

### 公网部署信息(已生效)

- 域名:`exact2026tsupernb.xyz`(Cloudflare 托管,NS 已生效)
- 隧道:Cloudflare named tunnel `exact2026`(id `9a7c177c-4881-4c9c-8d72-e671b56003bd`),配置在 `C:\Users\Madao\.cloudflared\config.yml`
- 公网 URL(已端到端验证 type1/type2 都通):
  - `https://predict.exact2026tsupernb.xyz/predict` → 本机 8080(根 api.py)
  - `https://vllm.exact2026tsupernb.xyz/v1/models` → 本机 8002(vLLM 容器)
- **评测日开机检查清单**(三个进程缺一不可):
  1. `docker start exact-vllm`(vLLM,8002)
  2. `E:\LLM-vllm` 下 `python -m uvicorn api:app --host 0.0.0.0 --port 8080`
  3. `cloudflared tunnel run exact2026`(或先用管理员 PowerShell 跑一次 `cloudflared service install` 注册为自启服务)
  然后从外网(如手机流量)访问两个 URL 自检。
- 机器设置:关闭睡眠/休眠,保证 slot 全程在线。

---

## 1. 公网暴露 API 端点

评测服务器从公网调用,两个 URL 都必须可达,且在**整个 1 小时 grading slot 内保持在线**:

- **`/predict`(统一端点,2026-06-11 定稿)**:主路径 `E:\LLM-vllm` 下
  `python -m uvicorn api:app --host 0.0.0.0 --port 8080`(即根目录 `api.py`)。
  它按 `type` 字段分发:Type 1 → `type1/IJCNN-Qiwei` 管线(retained WM/SSM 模型答选择题、
  vLLM 答自由问答);Type 2 → `type2/` 完整物理管线。
  配置自动读取 `type1/IJCNN-Qiwei/.env`(已设为比赛配置:vLLM 8002 + qwen3-8b-awq + bge),
  进程环境变量优先;DSPY_* 自动从 VLLM_* 派生,无需手动设置。
  两个子目录里各自的 api 文件只是内部库/历史产物,不作为服务端点。
- **`/v1/models`**(vLLM 自带,勿自己实现):现有容器 `exact-vllm`,宿主端口 **8002**,服务模型 `qwen3-8b-awq`(Qwen/Qwen3-8B-AWQ)——Type 1 与 Type 2 共用这一个 8B 模型,满足"任一时刻 ≤ 8B"。

可选方案:cloudflared tunnel / ngrok / 云主机反代。注意:
- 评测是**串行单发、每题只调一次、无重试**,60 秒超时——隧道稳定性比带宽重要。
- 委员会可能在 slot 期间随时查 `/v1/models` 验证模型身份,并可能检查 GPU 显存占用(任一时刻加载的 LLM 总参数 ≤ 8B)。
- 启动后用 `GET /health` 自检:`type2_pipeline_loaded` 应为 `true`(启动时预热)。

## 2. urls.txt

纯文本文件,列出:

```
<公网 /predict 完整 URL>
<公网 /v1/models 完整 URL>
```

如有多个 vLLM 服务器,每个 `/v1/models` 都要列出(当前只有一个)。

## 3. solution.pdf(一页)

必须包含三部分:

1. **Datasets used**:每个数据集写名称、来源、使用样本数、若干样例条目。
   - 官方 EXACT 数据集:`type2/Dataset/Physics_Problems_Text_Only.csv`(1352 条,内部按 8:2 切分 train/test)
   - 如用过外部 / 爬取 / 合成数据,也要列出(没有就声明仅用官方数据)。
2. **Approach and method**:管线概述。Type2:parse → formula retrieval → deterministic symbolic solve(SymPy)→ step verification → diagnosis / guarded repair → final-answer validation → response;hybrid 模式下仅在确定性求解失败时调用本地 vLLM LLM。
3. **Model size calculation**:列出管线中每个 LLM 及参数量,证明任一时刻 ≤ 8B:
   - 唯一 LLM:Qwen3-8B-AWQ(8B,AWQ 量化),经 vLLM 服务,Stage 0 解析 fallback 与 Stage 2 求解 fallback 共用同一端点(已于 2026-06-11 改造完成,parser 不再加载本地 GGUF)。
   - 非 LLM 组件(SymPy、规则解析器、公式库检索)不计入限额。

参考数据(可写进 PDF):全量 1352 题评估 objective answer accuracy **97.58%**(见 `type2/README.md` Final Type2 Results 表);50 题模拟评测(无缓存、实时解析)100% 出答案,平均时延 1.79s/题,最大 10.48s,远低于 60s 限制。

## 4. source_code.zip

打包 `type2/` 仓库源码(建议 `git archive` 干净导出,排除 `outputs/`、`__pycache__/`、`.git/`、数据集大文件视要求而定)。

## 5. Notation Mapping CSV

**等委员会发模板**(评测阶段前发放)。拿到后:
- 三列:`canonical_latex` / `meaning` / `your_notation`,只填与官方记法不同的行,空行保持官方记法。
- 重点检查单位约定:我方输出为纯 ASCII(`uF`、`nC`、`ohm`、`V/m`、`degC`),需与 CSV 中声明一致。
- 答案数值的量级表示(如我方输出 `0.25 J` vs 官方 `250 mJ`)如有约定空间,在此声明。

## 6. Grading slot 报名

- 表单将在 Discord / 官网公布,每队 1 个时段、时长 1 小时。
- 错过时段需尽快在 Discord `#technical-support` 联系,改期视余量而定。

## 7. 提交包结构

```
<team_name>.zip
├── solution.pdf
├── source_code.zip
├── urls.txt
└── notation_mapping.csv
```

---

## 附:提交前自检(Type2 相关,代码侧现状)

| 检查项 | 状态 |
|--------|------|
| `/predict` 接受统一输入 schema(query_id/type/query/premises/options) | ✅ |
| 返回 JSON list,含 query_id/answer/unit/explanation/premises_used/reasoning | ✅ |
| Type2 unit 为 ASCII | ✅(2026-06-11 回归验证 0 违规) |
| Type2 premises_used 为 `[]` | ✅ |
| explanation 非空 | ✅ |
| reasoning 为 `{"type":"cot","steps":[字符串]}` 或 null | ✅ |
| 60 秒内响应 | ✅(50 题实测 max 10.48s) |
| 所有 LLM 经 vLLM 服务、总参数 ≤ 8B | ✅(parser fallback 已改走 vLLM 8002) |
| 已知遗留:多小问题目 answer 含散文(THCB110/THCB118 类) | ⚠️ 待修(已建后台任务) |
