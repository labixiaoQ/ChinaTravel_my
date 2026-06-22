
# 运行一条 UrbanTrip
python run_tpc2026.py \
  --agent UrbanTrip \
  --index 20250320174446059265

# 批量运行前 10 条
python run_tpc2026.py --agent UrbanTrip --limit 10 --skip

当前 UrbanTrip 默认使用 TPCLLM，但它实际是规则基线的空模型封装 EmptyLLM，不会调用外部大模型

python run_tpc2026.py \
  --agent LLMNeSy \
  --llm deepseek \
  --index 20250320174446059265




python run_tpc2026.py \
  --agent UrbanTrip \
  --skip
--skip 会跳过已有结果，支持断点续跑。完成后运行评分
python eval_tpc.py \
  --splits TPC_IJCAI_2026_phase1 \
  --method UrbanTrip_TPCLLM_en_oracletranslation \
  --lang en
  其中 --method 必须与 results/ 下的目录名一致：

测试代码 可中断
python eval_tpc_partial.py \
  --method UrbanTrip_TPCLLM_en_oracletranslation