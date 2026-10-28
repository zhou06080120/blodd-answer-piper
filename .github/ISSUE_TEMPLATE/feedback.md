name: badcase 反馈
about: Demo 检索错误 / 域外拦截误判时提交，将进入主动学习闭环（低置信队列 → 灰度转正）
title: "[badcase] "
labels: feedback
body:
  - type: textarea
    id: question
    attributes:
      label: 你的提问
      description: 在 Demo 里输入的原始问法（口语化表达尤其欢迎，长尾问法是检索引擎最需要的数据）
      placeholder: 例如：量出来比医院量的低不少是为啥
    validations:
      required: true
  - type: textarea
    id: answer
    attributes:
      label: 引擎的实际返回
      description: 粘贴返回的 FAQ 条目 / 置信度；若被判定为域外闲聊拦截也请说明
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: 期望命中的 FAQ（可选）
    validations:
      required: false
