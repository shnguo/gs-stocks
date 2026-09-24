# 重要研究产物 R2 留存

从本版本开始，重要实验只有在以下条件全部满足后才算完成：

1. 冻结输入清单、协议、配置和源码已纳入归档。
2. 模型与优化器检查点、预测路径、逐日排名、报告和独立核验材料已纳入归档。
3. 每个分包均上传到内容寻址的 R2 对象，随后从 R2 读回、展开并逐文件核验。
4. 汇总目录本身再次作为独立小包写入 R2，并生成恢复验证回执。

归档脚本会拒绝没有完成清单的输入；支持 completed.json、ARCHIVE-MANIFEST.json 和日常流程的 manifest.json。脚本忽略临时文件和运行锁，并按固定顺序分包。默认每包 1 GiB；超过该值的单个文件独立成包，单文件上限为 4 GiB。失败的上传或恢复不会生成 retention.json，因此不能被误报为已留存。

手动归档示例：

    uv run python scripts/archive_research_artifact.py \
      --archive-id token-retrained-early-stage-20260922-v1 \
      --source artifacts/token-retrained-early-stage-20260922-v1 \
      --output artifacts/r2-retention/token-retrained-early-stage-20260922-v1

该流程不创建定时任务，也不修改线上模型指针。归档清单中的 R2 key、压缩体哈希、分片数量和恢复状态保存在 retention.json。

现有手动收盘入口 scripts/run_daily_token.sh 已默认启用同一留存门禁。每次人工执行完成后，它会归档当日原始抓取与标准化数据、刷新输入、增量模型、预测路径、排名、报告、配置和源码；R2 上传、回读、恢复或哈希校验任一步失败，整次命令都会失败。该变更没有增加定时任务。
